import time
from typing import Any, Dict, Iterable, Optional, Union
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rich.console import Group
from rich.padding import Padding
from rich.table import Column, Table
from core.modules.base import Metrics
from core import console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, Task
from rich.highlighter import ReprHighlighter
from lightning.pytorch.callbacks import ProgressBar


class TrainerProgress(Progress):
    def __init__(self, 
                 **kwargs
                 ):

        progress_bar = [
            SpinnerColumn(),
            "{task.description}",
            "[cyan]{task.completed:>3}[/cyan]/[cyan]{task.total}[/cyan]",
            "{task.fields[unit]}",
            BarColumn(),
            "[cyan]{task.percentage:>3.0f}[/cyan]%",
            TimeElapsedColumn(),
            # "{task.fields[metrics]}"
        ]

        super().__init__(*progress_bar, console=console, **kwargs)

        self.trainer_tasks = {
            'epoch': self.add_task(metrics='', unit='epochs', description='overal progress'),
            'train': self.add_task(metrics='', unit='steps', description='training', visible=False),
            'val':   self.add_task(metrics='', unit='steps', description='validation', visible=False),
            'test':  self.add_task(metrics='', unit='steps', description='testing', visible=False),
        }

        self.max_rows = 0

    def update(self, task: Task, **kwargs):
        if 'metrics' in kwargs:
            kwargs['metrics'] = self.render_metrics(kwargs['metrics'])

        super().update(self.trainer_tasks[task], **kwargs)

    def reset(self, task: Task, **kwargs):
        super().reset(self.trainer_tasks[task], **kwargs)

    def render_metrics(self, metrics: Metrics) -> str:
        metric_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items())
        return metric_str

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Args:
            tasks (Iterable[Task]): An iterable of Task instances, one per row of the table.

        Returns:
            Table: A table instance.
        """
        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )

        highlighter = ReprHighlighter()
        table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

        if tasks:
            epoch_task = tasks[0]
            metrics = epoch_task.fields['metrics']

            for task in tasks:
                if task.visible:
                    table.add_row(
                        *(
                            (
                                column.format(task=task)
                                if isinstance(column, str)
                                else column(task)
                            )
                            for column in self.columns
                        )
                    )

            self.max_rows = max(self.max_rows, table.row_count)
            pad_top = 0 if epoch_task.finished else self.max_rows - table.row_count
            group = Group(table, Padding(highlighter(metrics), pad=(pad_top,0,0,2)))
            return Padding(group, pad=(0,0,1,18))

        else:
            return table
        

class ProgressCallback(ProgressBar):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.progress = TrainerProgress()
        self.progress.update('epoch', total=int(trainer.max_epochs))
        self.progress.start()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total = self.total_train_batches
        self.progress.update('train', total=total, visible=total > 1, completed=0)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        time.sleep(0.2)
        self.progress.update('train', advance=1)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total = self.total_val_batches
        self.progress.update('val', total=total, visible=total > 2, completed=0)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        time.sleep(0.2)
        self.progress.update('val', advance=1)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.progress.update('val',visible=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        self.progress.update('train',visible=False)
        self.progress.update('epoch', advance=1, metrics=metrics)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.progress.stop()

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        self.progress.stop()

    def print(self, *args: Any, **kwargs: Any) -> None:
        console.print(*args, **kwargs)

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, int | str | float | Dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        keys = list(metrics.keys())
        keys.sort(key=self.sort_metrics)
        metrics = {k: metrics[k] for k in keys}
        return metrics

    def sort_metrics(self, metric_key: str) -> int:
        if metric_key.startswith('train'):
            return 1
        elif metric_key.startswith('val'):
            return 2
        elif metric_key.startswith('test'):
            return 3
        else:
            return 4
