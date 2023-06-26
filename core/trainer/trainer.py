from copy import deepcopy
from typing import Annotated, Iterable, Literal, Optional
import torch
from torch.optim import Optimizer
from torch.types import Number
from torchmetrics import MeanMetric
from core.args.utils import ArgInfo
from core.loggers.dummy import DummyLogger
from core.loggers.logger import Logger
from core.trainer.progress import TrainerProgress
from core.modules.base import Metrics, TrainableModule
from core.typing import Phase


class Trainer:
    def __init__(self,
                 monitor:       str = 'val/acc',
                 monitor_mode:  Literal['min', 'max'] = 'max',
                 epochs:        Annotated[int,  ArgInfo(help='number of epochs for training')] = 100,
                 device:        Annotated[str,  ArgInfo(help='device to use for training', choices=['cpu', 'cuda', 'auto'])] = 'auto',
                 verbose:       Annotated[bool, ArgInfo(help='display progress')] = True,
                 logger:        Logger = None,
                 ):

        self.epochs = epochs
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.verbose = verbose
        self.logger = logger or DummyLogger()

        # setup device
        if device == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # trainer internal state
        self.model: TrainableModule = None
        self.metrics: dict[str, MeanMetric] = {}

    def reset(self) -> None:
        self.model = None
        self.metrics.clear()

    def update_metrics(self, metric_name: str, metric_value: object, batch_size: int = 1) -> None:
        # if this is a new metric, add it to self.metrics
        device = metric_value.device if torch.is_tensor(metric_value) else 'cpu'
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MeanMetric(compute_on_step=False).to(device)

        # update the metric
        self.metrics[metric_name].update(metric_value, weight=batch_size)

    def aggregate_metrics(self, phase: Phase='train') -> Metrics:
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if phase in metric_name.split('/'):
                value = metric_value.compute()
                metric_value.reset()
                metrics[metric_name] = value

        return metrics

    def is_better(self, current_metric: Number, previous_metric: Number) -> bool:
        assert self.monitor_mode in ['min', 'max'], f'Unknown metric mode: {self.monitor_mode}'
        if self.monitor_mode == 'max':
            return current_metric > previous_metric
        elif self.monitor_mode == 'min':
            return current_metric < previous_metric
        
    def fit(self, 
            model: TrainableModule, 
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            ) -> Metrics:
        
        self.model = model.to(self.device)
        self.optimizer: Optimizer = self.model.configure_optimizers()

        self.progress = TrainerProgress(
            num_epochs=self.epochs, 
            disable=not self.verbose,
        )

        best_state_dict = None
        best_metrics = None
        
        with self.progress:
            for epoch in range(1, self.epochs + 1):
                metrics = {f'epoch': epoch}

                # train loop
                train_metrics = self.loop(train_dataloader, phase='train')
                metrics.update(train_metrics)
                    
                # validation loop
                if val_dataloader:
                    val_metrics = self.loop(val_dataloader, phase='val')
                    metrics.update(val_metrics)

                    if best_metrics is None or self.is_better(
                        metrics[self.monitor], best_metrics[self.monitor]
                        ):
                        best_metrics = metrics
                        best_state_dict = deepcopy(self.model.state_dict())

                # test loop
                if test_dataloader:
                    test_metrics = self.loop(test_dataloader, phase='test')
                    metrics.update(test_metrics)

                # log and update progress
                self.progress.update(task='epoch', metrics=metrics, advance=1)
                self.logger.log(metrics)

        if best_metrics is None:
            best_metrics = metrics
        else:
            self.model.load_state_dict(best_state_dict)

        # log and return best metrics
        self.logger.log_summary(best_metrics)

        return best_metrics

    def test(self, dataloader: Iterable) -> Metrics:
        self.metrics.clear()
        metrics = self.loop(dataloader, phase='test')
        return metrics

    def predict(self, dataloader: Iterable, move_to_cpu: bool=False) -> Metrics:
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = self.to_device(batch)
                # out might be a tuple of predictions
                out = self.model.predict(batch)
                if move_to_cpu:
                    out = out.cpu()
                preds.append(out)
            
        # concatenate predictions, check if they are tuples
        if isinstance(preds[0], tuple):
            preds = tuple(torch.cat([p[i] for p in preds]) for i in range(len(preds[0])))
        else:
            preds = torch.cat(preds)

        return preds

    def loop(self, dataloader: Iterable, phase: Phase) -> Metrics:
        self.model.train(phase == 'train')
        grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(phase == 'train')
        self.progress.update(phase, visible=len(dataloader) > 1, total=len(dataloader))

        for batch in dataloader:
            batch = self.to_device(batch)
            metrics = self.step(batch, phase)
            for item in metrics:
                self.update_metrics(item, metrics[item], batch_size=batch.batch_nodes.size(0))
            self.progress.update(phase, advance=1)

        self.progress.reset(phase, visible=False)
        torch.set_grad_enabled(grad_state)
        return self.aggregate_metrics(phase)

    def step(self, batch, phase: Phase) -> Metrics:
        if phase == 'train':
            self.optimizer.zero_grad(set_to_none=True)

        loss, metrics = self.model.step(batch, phase=phase)
        
        if phase == 'train':
            loss.backward()
            self.optimizer.step()

        return metrics
    
    def to_device(self, batch):
        if isinstance(batch, tuple):
            return tuple(item.to(self.device) for item in batch)
        return batch.to(self.device)
