"""Camera-ready LightningModule: the historical VitsModel with the protocol fixes.

Differences from the historical training (vits/lightning.py), all deliberate:
- group_semantics="strict": use_bigvgan also switches the Snake activations inside the decoder
  ResBlocks (historically they were always built);
- the split is read by stable ID from the split manifest; no random_split, and the training seed
  no longer influences the split; the test split is never loaded during training;
- training data order comes from an explicit DistributedSampler(seed=data_seed); model
  initialisation uses train_seed (set by the caller before construction);
- the torch/NumPy/Python RNGs are reseeded at every epoch start from (train_seed, rank, epoch), so a
  resume at an epoch boundary continues exactly like an uninterrupted run;
- validation is per utterance (batch size 1) with a fixed seed per utterance, without MAS noise,
  sharded across ranks without padding, and the monitored value is sum/count over all ranks;
- previews are synthesized from fixed validation IDs, never from test utterances;
- generator updates, discriminator updates and samples are counted and stored in checkpoints; the
  MAS noise decay is defined per generator update.
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ..vits.commons import slice_segments
from ..vits.dataset import PiperDataset, UtteranceCollate
from ..vits.lightning import VitsModel
from ..vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .distributed import ShardedSequentialSampler, gather_per_rank, reduce_sum_count
from .split import load_split_manifest

_LOGGER = logging.getLogger("camera_ready.model")

MONITOR = "val_mel_l1_global"
MAS_NOISE_INITIAL = 0.01
COUNTER_UNITS = {
    "g_updates": "generator optimizer steps (one per training batch per rank; equal on all ranks)",
    "d_updates": "discriminator optimizer steps (one per training batch per rank)",
    "samples_seen_global": "training utterances consumed, summed over ranks (updated at epoch end)",
    "lightning_global_step": "Lightning global_step; with manual optimization it counts both optimizer steps (2 per batch)",
    "mas_decay_per_g_update": "MAS noise scale = max(0.01 - g_updates * mas_decay_per_g_update, 0)",
    "lr_schedule": "ExponentialLR(gamma=lr_decay) stepped once per epoch, as in the historical source",
}


def _epoch_seed(train_seed: int, rank: int, epoch: int) -> int:
    return (train_seed * 1_000_003 + rank * 10_007 + epoch) % (2**32 - 1)


class CameraReadyVitsModel(VitsModel):
    def __init__(
        self,
        split_manifest: str,
        train_seed: int = 1234,
        data_seed: int = 1234,
        val_seed: int = 2026,
        train_shuffle: bool = False,
        mas_decay_per_g_update: float = 4e-6,
        val_batch_size: int = 1,
        preview_every_n_epochs: int = 1,
        diagnostics_path: Optional[str] = None,
        **kwargs,
    ):
        if val_batch_size != 1:
            raise ValueError("val_batch_size must be 1: the per-utterance validation seed assumes one utterance per batch")
        kwargs.pop("group_semantics", None)
        kwargs["seed"] = train_seed
        # The historical per-Lightning-step decay is not used; the decay is per generator update.
        kwargs["mas_noise_scale_decay"] = 0.0
        super().__init__(
            group_semantics="strict",
            split_manifest=split_manifest,
            train_seed=train_seed,
            data_seed=data_seed,
            val_seed=val_seed,
            train_shuffle=train_shuffle,
            mas_decay_per_g_update=mas_decay_per_g_update,
            val_batch_size=val_batch_size,
            preview_every_n_epochs=preview_every_n_epochs,
            diagnostics_path=diagnostics_path,
            **kwargs,
        )
        self.g_updates = 0
        self.d_updates = 0
        self.samples_seen_global = 0
        self._samples_epoch_local = 0
        self._val_total = torch.zeros((), dtype=torch.float64)
        self._val_count = torch.zeros((), dtype=torch.float64)
        self._val_rows: List[Tuple[str, float]] = []
        self._val_positions: List[int] = []

    # ------------------------------------------------------------------ data
    def _load_datasets(self, num_val_examples: int, num_test_examples: int, max_phoneme_ids: Optional[int] = None):
        self._preview_items = []
        self._val_ids: List[str] = []
        if self.hparams.dataset is None:
            return
        dataset_path = Path(self.hparams.dataset[0])
        manifest = load_split_manifest(self.hparams.split_manifest, dataset_path)
        if len(manifest.ids["validation"]) != num_val_examples or len(manifest.ids["test"]) != num_test_examples:
            raise ValueError("num_val_examples/num_test_examples differ from the split manifest")
        full = PiperDataset(self.hparams.dataset, max_phoneme_ids=max_phoneme_ids)
        lines = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(full) != len(lines):
            raise ValueError("PiperDataset dropped rows; row indices of the manifest would not match")
        for split in ("train", "validation"):
            for row in manifest.rows[split]:
                if str(full.utterances[row].audio_norm_path) != str(Path(lines[row]["audio_norm_path"])):
                    raise ValueError(f"row {row}: dataset row and utterance disagree")
        self._train_dataset = Subset(full, manifest.rows["train"])
        self._val_dataset = Subset(full, manifest.rows["validation"])
        self._test_dataset = None  # never touched during training
        self._val_ids = list(manifest.ids["validation"])
        for uid in manifest.preview_ids:
            row = manifest.rows["validation"][self._val_ids.index(uid)]
            self._preview_items.append((uid, torch.LongTensor(full.utterances[row].phoneme_ids)))
        self.hparams["split_manifest_sha256"] = manifest.sha256
        self.hparams["dataset_jsonl_sha256"] = manifest.dataset_sha256

    def _world_rank(self) -> Tuple[int, int]:
        if self._trainer is None:
            return 1, 0
        return self.trainer.world_size, self.trainer.global_rank

    def _collate(self):
        return UtteranceCollate(is_multispeaker=self.hparams.num_speakers > 1, segment_size=self.hparams.segment_size)

    def train_dataloader(self):
        world, rank = self._world_rank()
        sampler = DistributedSampler(self._train_dataset, num_replicas=world, rank=rank,
                                     shuffle=bool(self.hparams.train_shuffle), seed=int(self.hparams.data_seed), drop_last=False)
        return DataLoader(self._train_dataset, sampler=sampler, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def val_dataloader(self):
        world, rank = self._world_rank()
        sampler = ShardedSequentialSampler(len(self._val_dataset), world, rank)
        return DataLoader(self._val_dataset, sampler=sampler, collate_fn=self._collate(), num_workers=0,
                          batch_size=1, pin_memory=True)

    def test_dataloader(self):
        raise RuntimeError("the camera-ready protocol never evaluates the test split inside training")

    # ------------------------------------------------------------------ training
    def _set_mas_noise(self):
        if self.hparams.use_vits2 and self.hparams.mas_decay_per_g_update > 0:
            self.model_g.mas_noise_scale = max(MAS_NOISE_INITIAL - self.g_updates * self.hparams.mas_decay_per_g_update, 0.0)

    def on_train_start(self):
        self._set_mas_noise()

    def on_train_epoch_start(self):
        _, rank = self._world_rank()
        seed = _epoch_seed(int(self.hparams.train_seed), rank, int(self.current_epoch))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._samples_epoch_local = 0

    def training_step(self, batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        loss_gen_all, _loss_mel = self.training_step_g(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        opt_g.step()
        self.g_updates += 1

        loss_disc_all = self.training_step_d(batch)
        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(opt_d, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        opt_d.step()
        self.d_updates += 1

        self._samples_epoch_local += int(batch.phoneme_ids.size(0))
        self._set_mas_noise()
        self.log("train/g_updates", float(self.g_updates), on_step=True, on_epoch=False)
        self.log("train/mas_noise_scale", float(self.model_g.mas_noise_scale), on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()  # steps both ExponentialLR schedulers once per epoch
        total, _ = reduce_sum_count(torch.tensor(float(self._samples_epoch_local), device=self.device),
                                    torch.zeros((), device=self.device))
        self.samples_seen_global += int(total.item())

    # ------------------------------------------------------------------ validation
    def _mel_l1_per_sample(self, batch) -> torch.Tensor:
        x, x_lengths, spec, spec_lengths, f0 = (batch.phoneme_ids, batch.phoneme_lengths, batch.spectrograms,
                                                batch.spectrogram_lengths, batch.f0s)
        sid = batch.speaker_ids if batch.speaker_ids is not None else None
        y_hat, _l_length, _attn, ids_slice, *_ = self.model_g(x, x_lengths, spec, spec_lengths, sid, f0=f0)
        with autocast(self.device.type, enabled=False):
            mel = spec_to_mel_torch(spec.float(), self.hparams.filter_length, self.hparams.mel_channels,
                                    self.hparams.sample_rate, self.hparams.mel_fmin, self.hparams.mel_fmax)
            y_mel = slice_segments(mel, ids_slice, self.hparams.segment_size // self.hparams.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), self.hparams.filter_length, self.hparams.mel_channels,
                                              self.hparams.sample_rate, self.hparams.hop_length, self.hparams.win_length,
                                              self.hparams.mel_fmin, self.hparams.mel_fmax)
            return (y_mel - y_hat_mel).abs().mean(dim=(1, 2)) * self.hparams.c_mel

    def _rng_devices(self):
        return [self.device.index if self.device.index is not None else 0] if self.device.type == "cuda" else []

    def on_validation_epoch_start(self):
        world, rank = self._world_rank()
        self._val_positions = ShardedSequentialSampler(len(self._val_dataset), world, rank).positions()
        self._val_total = torch.zeros((), dtype=torch.float64, device=self.device)
        self._val_count = torch.zeros((), dtype=torch.float64, device=self.device)
        self._val_rows = []

    def validation_step(self, batch, batch_idx: int):
        pos = self._val_positions[batch_idx]
        uid = self._val_ids[pos]
        noised = self.model_g.use_noised_mas
        self.model_g.use_noised_mas = False  # MAS noise is a training regulariser; validation is measured without it
        try:
            with torch.random.fork_rng(devices=self._rng_devices()):
                torch.manual_seed((int(self.hparams.val_seed) * 1_000_003 + pos) % (2**31 - 1))
                value = self._mel_l1_per_sample(batch)
        finally:
            self.model_g.use_noised_mas = noised
        self._val_total = self._val_total + value.sum().double()
        self._val_count = self._val_count + value.numel()
        self._val_rows.append((uid, float(value.sum())))
        if batch_idx == 0 and self.global_rank == 0 and not self.trainer.sanity_checking:
            self._log_previews()

    def on_validation_epoch_end(self):
        local = torch.stack([self._val_total.double(), self._val_count.double()])
        total, count = reduce_sum_count(self._val_total, self._val_count)
        if count.item() == 0:
            raise RuntimeError("no validation utterance was scored")
        value = (total / count).float()
        # The value is already the exact global mean, identical on every rank; sync_dist averages identical values.
        self.log(MONITOR, value.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        per_rank = gather_per_rank(local)
        rows = self._gather_rows()
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            record = {
                "epoch": int(self.current_epoch), "g_updates": int(self.g_updates), "d_updates": int(self.d_updates),
                "lightning_global_step": int(self.global_step), "monitor": MONITOR, "value": float(value),
                "total": float(total), "count": int(count.item()),
                "per_rank": [{"rank": r, "total": float(v[0]), "count": int(v[1]),
                              "mean": float(v[0] / v[1]) if v[1] > 0 else None} for r, v in enumerate(per_rank)],
                "per_utterance": dict(sorted(rows)),
            }
            if self.hparams.diagnostics_path:
                with open(self.hparams.diagnostics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_scalar"):
                for entry in record["per_rank"]:
                    if entry["mean"] is not None:
                        exp.add_scalar(f"diag/val_mel_rank{entry['rank']}", entry["mean"], self.g_updates)

    def _gather_rows(self) -> List[Tuple[str, float]]:
        if dist.is_available() and dist.is_initialized():
            out = [None] * dist.get_world_size()
            dist.all_gather_object(out, self._val_rows)
            return [row for part in out for row in part]
        return list(self._val_rows)

    def _log_previews(self):
        every = int(self.hparams.preview_every_n_epochs)
        if every <= 0 or self.current_epoch % every:
            return
        exp = getattr(self.logger, "experiment", None)
        if exp is None or not hasattr(exp, "add_audio"):
            return
        for k, (uid, phoneme_ids) in enumerate(self._preview_items):
            with torch.random.fork_rng(devices=self._rng_devices()):
                torch.manual_seed((int(self.hparams.val_seed) * 7919 + k) % (2**31 - 1))
                text = phoneme_ids.unsqueeze(0).to(self.device)
                lengths = torch.LongTensor([phoneme_ids.numel()]).to(self.device)
                try:
                    audio = self(text, lengths, [0.667, 1.0, 0.8]).detach().float()
                except Exception as exc:  # reported, not hidden: a preview failure is a symptom worth seeing
                    _LOGGER.warning("preview %s failed at epoch %s: %r", uid, self.current_epoch, exc)
                    continue
            if not torch.isfinite(audio).all():
                _LOGGER.warning("preview %s is not finite at epoch %s", uid, self.current_epoch)
                continue
            audio = audio * (1.0 / max(0.01, float(audio.abs().max())))
            exp.add_audio(f"preview/{uid}", audio.squeeze().cpu(), global_step=self.g_updates,
                          sample_rate=self.hparams.sample_rate)

    # ------------------------------------------------------------------ checkpoint state
    def on_save_checkpoint(self, checkpoint):
        checkpoint["camera_ready"] = {
            "g_updates": int(self.g_updates), "d_updates": int(self.d_updates),
            "samples_seen_global": int(self.samples_seen_global), "units": COUNTER_UNITS,
            "train_seed": int(self.hparams.train_seed), "data_seed": int(self.hparams.data_seed),
            "val_seed": int(self.hparams.val_seed),
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("camera_ready")
        if state is None:
            raise ValueError("checkpoint lacks camera_ready counters; it was not written by this protocol")
        self.g_updates = int(state["g_updates"])
        self.d_updates = int(state["d_updates"])
        self.samples_seen_global = int(state["samples_seen_global"])
