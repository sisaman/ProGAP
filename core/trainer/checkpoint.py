"""
In-Memory Checkpointing
===================

Automatically save in-memory model checkpoints during training.
"""

from copy import deepcopy
from typing import Optional

import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class InMemoryCheckpoint(Checkpoint):
    def __init__(
        self,
        monitor: str = 'val/acc',
        mode: str = 'max',
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.best_metrics = None
        self.best_state_dict = None

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer):
            monitor_candidates = trainer.callback_metrics
            self._save_topk_checkpoint(trainer, monitor_candidates)

    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
        )

    def _save_topk_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        # validate metric
        if self.monitor not in monitor_candidates:
            m = (
                f"`InMemoryCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                f" metrics: {list(monitor_candidates)}."
                f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
            )
            if trainer.fit_loop.epoch_loop.val_loop._has_run:
                raise MisconfigurationException(m)
        
        self._save_monitor_checkpoint(trainer, monitor_candidates)

    def _save_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            self._update_best_and_save(current, trainer, monitor_candidates)

    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[Tensor] = None) -> bool:
        if current is None:
            return False

        if self.best_metrics is None:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(current, self.best_metrics[self.monitor])

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save
    
    def _update_best_and_save(
        self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]
    ) -> None:
        self.best_metrics = deepcopy(monitor_candidates)
        self.best_state_dict = deepcopy(trainer.lightning_module.state_dict())
