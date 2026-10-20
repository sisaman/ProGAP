from typing import Annotated, ClassVar

from torch.nn import Module

from gnnprivtool.args.utils import ArgInfo
from gnnprivtool.loggers.base import LoggerBase
from gnnprivtool.loggers.dummy import DummyLogger
from gnnprivtool.loggers.wandb import WandbLogger


class Logger(LoggerBase):
    supported_loggers: ClassVar[dict[str, type[LoggerBase]]] = {
        'wandb': WandbLogger
    }

    def __init__(self,
        logger:     Annotated[str | None,  ArgInfo(help='select logger type', choices=supported_loggers)] = None,
        project:    Annotated[str,  ArgInfo(help="project name for logger")] = 'ProGAP',
        output_dir: Annotated[str,  ArgInfo(help="directory to store the results")] = './output',
        config:     dict | None = None,
        prefix:     str = '',
        ) -> None:
        
        if config is None:
            config = {}
        self.logger_name = logger
        self.project = project
        self.output_dir = output_dir
        self.config = config
        self.prefix = prefix

        if self.logger_name == 'wandb':
            self.logger = WandbLogger(
                project=project,
                output_dir=output_dir,
                config=config
            )
        else:
            self.logger = DummyLogger()
    
    @property
    def experiment(self):
        return self.logger.experiment

    def log(self, metrics: dict[str, object]):
        metrics = self._add_prefix(metrics)
        self.logger.log(metrics)
    
    def log_summary(self, metrics: dict[str, object]):
        metrics = self._add_prefix(metrics)
        self.logger.log_summary(metrics)

    def watch(self, model: Module, **kwargs):
        self.logger.watch(model, **kwargs)
    
    def finish(self):
        self.logger.finish()

    def set_prefix(self, prefix: str):
        self.prefix = prefix

    def _add_prefix(self, metrics: dict[str, object]) -> dict[str, object]:
        return {f'{self.prefix}{metric}': value for metric, value in metrics.items()}
