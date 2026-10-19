"""End-to-end CPU tests of the camera-ready training entry point (real model, tiny split).

- the budget stops at the end of the epoch that reaches max_g_updates, and the counters are stored;
- a run interrupted after one epoch and resumed ends with weights identical to an uninterrupted run;
- the checkpoint ledger records saves/removals, and the best-checkpoint state survives the resume;
- new checkpoints load with weights_only=True and no allowlist;
- validation on 2 processes (uneven shards 2 + 1) gives the same metric and per-utterance values as 1 process.
"""
import json
import unittest

import torch

from _common import DATASET_DIR, WS, base_model_kwargs, dataset_config, new_run_dir, small_manifest
from piper_train.camera_ready import train as cr_train
from piper_train.camera_ready.model import MONITOR, CameraReadyVitsModel

torch.set_num_threads(4)
CONFIG = WS / "configs" / "cpu-smoke-H.json"


def run(run_dir, manifest, *extra):
    argv = ["--config", str(CONFIG), "--dataset-dir", str(DATASET_DIR), "--split-manifest", str(manifest),
            "--run-dir", str(run_dir), "--limit-train-batches", "2", *extra]
    return cr_train.main(argv)


def ledger(run_dir):
    return [json.loads(line) for line in (run_dir / "checkpoint_ledger.jsonl").read_text().splitlines()]


class TrainCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = small_manifest(n_train=4, n_val=3, n_test=2)
        cls.a = new_run_dir("uninterrupted")
        run(cls.a, cls.manifest)
        cls.b = new_run_dir("interrupted")
        run(cls.b, cls.manifest, "--stop-after-epochs", "1")
        cls.b_after_first = json.loads((cls.b / "run_manifest.json").read_text())
        run(cls.b, cls.manifest, "--resume")

    def test_budget_and_counters(self):
        m = json.loads((self.a / "run_manifest.json").read_text())
        att = m["attempts"][-1]
        self.assertEqual(att["status"], "completed")
        self.assertEqual((att["g_updates"], att["d_updates"]), (4, 4))      # 2 epochs x 2 batches; budget 4
        self.assertEqual(att["samples_seen_global"], 8)
        self.assertEqual(att["lightning_global_step"], 8)                     # Lightning counts both optimizer steps
        ck = torch.load(self.a / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)  # no allowlist
        self.assertEqual(ck["camera_ready"]["g_updates"], 4)
        self.assertEqual(ck["epoch"], 1)

    def test_resume_reproduces_uninterrupted_run(self):
        self.assertEqual(self.b_after_first["attempts"][0]["g_updates"], 2)
        a = torch.load(self.a / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)
        b = torch.load(self.b / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)
        self.assertEqual(a["camera_ready"]["g_updates"], b["camera_ready"]["g_updates"])
        self.assertEqual(set(a["state_dict"]), set(b["state_dict"]))
        diff = [k for k in a["state_dict"] if not torch.equal(a["state_dict"][k], b["state_dict"][k])]
        self.assertEqual(diff, [], f"{len(diff)} tensors differ, e.g. {diff[:3]}")
        for oa, ob in zip(a["optimizer_states"], b["optimizer_states"]):
            for sa, sb in zip(oa["state"].values(), ob["state"].values()):
                self.assertTrue(torch.equal(sa["exp_avg"], sb["exp_avg"]))
        self.assertEqual(a["lr_schedulers"], b["lr_schedulers"])

    def test_ledger_and_best_state_across_resume(self):
        rows_a, rows_b = ledger(self.a), ledger(self.b)
        best_a = [r for r in rows_a if r["role"] == "best" and r["event"] == "save"]
        self.assertTrue(best_a)
        self.assertTrue(all(r["monitor"] == MONITOR and r["current_score"] is not None for r in best_a))
        self.assertTrue(any(r["role"] == "rolling" and r["event"] == "remove" for r in rows_a))
        final_a = [r for r in rows_a if r["role"] == "best"][-1]
        final_b = [r for r in rows_b if r["role"] == "best"][-1]
        self.assertEqual(final_a["best_model_file"], final_b["best_model_file"])
        self.assertAlmostEqual(final_a["best_model_score"], final_b["best_model_score"], places=6)

    def test_validation_diagnostics_written(self):
        rows = [json.loads(line) for line in (self.a / "val_diagnostics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["count"] == 3 and len(r["per_utterance"]) == 3 for r in rows))

    def test_fresh_run_refuses_existing_directory(self):
        with self.assertRaises(SystemExit):
            run(self.a, self.manifest)


def _validate(devices, manifest, diag):
    import pytorch_lightning as pl
    pl.seed_everything(11, workers=False)
    cfg = json.loads(CONFIG.read_text())
    kwargs = {**base_model_kwargs(), **cfg["model"], "dataset": [str(DATASET_DIR / "dataset.jsonl")]}
    m = CameraReadyVitsModel(**kwargs, split_manifest=str(manifest), diagnostics_path=str(diag),
                             **{k: cfg["protocol"][k] for k in ("train_seed", "data_seed", "val_seed", "train_shuffle",
                                                                "mas_decay_per_g_update")})
    trainer = pl.Trainer(accelerator="cpu", devices=devices, strategy="ddp_spawn" if devices > 1 else "auto",
                         logger=False, enable_checkpointing=False, enable_progress_bar=False, use_distributed_sampler=False)
    trainer.validate(m)
    return json.loads(diag.read_text().splitlines()[-1])


class ValidationDDP(unittest.TestCase):
    def test_two_processes_equal_one(self):
        manifest = small_manifest(n_train=4, n_val=3, n_test=2)
        d1, d2 = new_run_dir("val1"), new_run_dir("val2")
        d1.mkdir(), d2.mkdir()
        one = _validate(1, manifest, d1 / "diag.jsonl")
        two = _validate(2, manifest, d2 / "diag.jsonl")
        self.assertEqual([p["count"] for p in two["per_rank"]], [2, 1])
        self.assertEqual(one["count"], two["count"])
        self.assertEqual(one["per_utterance"].keys(), two["per_utterance"].keys())
        for uid in one["per_utterance"]:
            self.assertAlmostEqual(one["per_utterance"][uid], two["per_utterance"][uid], places=5)
        self.assertAlmostEqual(one["value"], two["value"], places=5)


if __name__ == "__main__":
    unittest.main()
