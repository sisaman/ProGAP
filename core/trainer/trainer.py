from copy import deepcopy
import torch
# from torch.types import Number
from torch.optim import Optimizer
from typing import Annotated, Iterable, Literal, Optional
from core.args.utils import ArgInfo
from core.trainer.checkpoint import InMemoryCheckpoint
# from torchmetrics import MeanMetric
from core.trainer.progress import TrainerProgress
from core.modules.base import Metrics, Phase, TrainableModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.loggers import Logger


class Trainer:
    def __init__(self,
                 patience:      int = 0,
                 monitor:       str = 'val/acc',
                 monitor_mode:  Literal['min', 'max'] = 'max',
                 val_interval:  Annotated[int, ArgInfo(help='interval of validation')] = 1,
                 verbose:       Annotated[bool, ArgInfo(help='display progress')] = True,
                 logger:        Optional[Logger] = None,
                 ):

        assert monitor_mode in ['min', 'max']

        self.patience = patience
        self.val_interval = val_interval
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.verbose = verbose
        self.logger = logger
        
        # trainer internal state
        # self.model: TrainableModule = None
        # self.metrics: dict[str, MeanMetric] = {}

        self.reset()

    def reset(self):
        # self.model = None
        # self.metrics = {}
        self.trainer = None
        

    # def update_metrics(self, metric_name: str, metric_value: object, weight: int = 1) -> None:
    #     # if this is a new metric, add it to self.metrics
    #     device = metric_value.device if torch.is_tensor(metric_value) else 'cpu'
    #     if metric_name not in self.metrics:
    #         self.metrics[metric_name] = MeanMetric(compute_on_step=False).to(device)

    #     # update the metric
    #     self.metrics[metric_name].update(metric_value, weight=weight)

    # def aggregate_metrics(self, phase: Phase='train') -> Metrics:
    #     metrics = {}

    #     for metric_name, metric_value in self.metrics.items():
    #         if phase in metric_name.split('/'):
    #             value = metric_value.compute()
    #             metric_value.reset()
    #             metrics[metric_name] = value

    #     return metrics

    # def is_better(self, current_metric: Number, previous_metric: Number) -> bool:
    #     if self.monitor_mode == 'max':
    #         return current_metric > previous_metric
    #     elif self.monitor_mode == 'min':
    #         return current_metric < previous_metric
    #     else:
    #         raise ValueError(f'Unknown metric mode: {self.monitor_mode}')

    def fit(self, 
            model: TrainableModule, 
            epochs: int,
            optimizer: Optimizer, 
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            ) -> Metrics:

        self.model = model
        self.optimizer = optimizer

        progress = TrainerProgress(
            # num_epochs=epochs, 
            # num_train_steps=len(train_dataloader), 
            # num_val_steps=len(val_dataloader), 
            # num_test_steps=len(test_dataloader),
            # disable=not self.verbose,
        )

        checkpoint = InMemoryCheckpoint(
            monitor=self.monitor, 
            mode=self.monitor_mode
        )

        self.trainer = LightningTrainer(
            accelerator='auto',
            callbacks=[progress, checkpoint],
            logger=self.logger,
            max_epochs=epochs,
            val_check_interval=self.val_interval,
            check_val_every_n_epoch=None,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=self.verbose,
            enable_model_summary=False,
            deterministic=True,
        )

        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            test_dataloaders=test_dataloader,
        )
        
        
        # test loop
        if test_dataloader:
            test_metrics = self.loop(test_dataloader, phase='test')
            metrics.update(test_metrics)

        
        if checkpoint.best_metrics is None:
            best_metrics = 
        else:
            # load best model if checkpointing is enabled
            if checkpoint and best_state_dict is not None:
                self.model.load_state_dict(best_state_dict)

        # log and return best metrics
        Logger.get_instance().log_summary(best_metrics)
        return best_metrics

    def test(self, dataloader: Iterable) -> Metrics:
        self.metrics.clear()
        metrics = self.loop(dataloader, phase='test')
        return metrics

    def loop(self, dataloader: Iterable, phase: Phase) -> Metrics:
        self.model.train(phase == 'train')
        self.progress.update(phase, visible=len(dataloader) > 1)

        for batch in dataloader:
            metrics = self.step(batch, phase)
            for item in metrics:
                self.update_metrics(item, metrics[item], weight=len(batch))
            self.progress.update(phase, advance=1)

        self.progress.reset(phase, visible=False)
        return self.aggregate_metrics(phase)

    def step(self, batch, phase: Phase) -> Metrics:
        if phase == 'train':
            self.optimizer.zero_grad(set_to_none=True)

        grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(phase == 'train')
        loss, metrics = self.model.step(batch, phase=phase)
        torch.set_grad_enabled(grad_state)
        
        if phase == 'train' and loss is not None:
            loss.backward()
            self.optimizer.step()

        return {f'{phase}/{key}': value for key, value in metrics.items()}
