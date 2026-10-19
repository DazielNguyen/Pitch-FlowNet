"""Callbacks: a training budget in generator updates and a ledger of every checkpoint save/removal."""
import json
import time
from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class GUpdateBudget(Callback):
    """Stop at the end of the first epoch whose cumulative generator updates reach the budget.

    The stop happens at an epoch boundary so that the last epoch is validated and checkpointed like
    every other one. Counters are equal on all ranks, so the decision is the same on every rank.
    """

    def __init__(self, max_g_updates: int):
        if max_g_updates <= 0:
            raise ValueError("max_g_updates must be positive")
        self.max_g_updates = int(max_g_updates)

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.g_updates >= self.max_g_updates:
            trainer.should_stop = True

    def state_dict(self):
        return {"max_g_updates": self.max_g_updates}

    def load_state_dict(self, state_dict):
        if int(state_dict["max_g_updates"]) != self.max_g_updates:
            raise ValueError("the resumed run was started with a different max_g_updates")


class LedgerModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that appends one JSON line per checkpoint save and removal (global rank 0)."""

    def __init__(self, *args, ledger_path: Optional[str] = None, role: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger_path = ledger_path
        self.role = role

    def _write(self, trainer, event: str, filepath: str):
        if not self.ledger_path or not trainer.is_global_zero:
            return
        module = trainer.lightning_module
        score = self.current_score
        row = {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "role": self.role, "event": event,
            "file": Path(filepath).name, "epoch": int(trainer.current_epoch),
            "lightning_global_step": int(trainer.global_step),
            "g_updates": int(getattr(module, "g_updates", -1)), "d_updates": int(getattr(module, "d_updates", -1)),
            "monitor": self.monitor, "current_score": float(score) if score is not None else None,
            "best_model_score": float(self.best_model_score) if self.best_model_score is not None else None,
            "best_model_file": Path(self.best_model_path).name if self.best_model_path else None,
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self._write(trainer, "save", filepath)

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        self._write(trainer, "remove", filepath)
