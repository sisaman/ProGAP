from argparse import Namespace
import os
from typing import Annotated, Any, Dict, Optional, Union
from uuid import uuid1
from core import console
from core.args.utils import ArgInfo
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger, Logger as LoggerBase


class Logger(LoggerBase):
    supported_loggers = {
        'wandb': WandbLogger
    }
    def __init__(self,
        logger:     Annotated[str,  ArgInfo(help='select logger type', choices=supported_loggers)] = None,
        project:    Annotated[str,  ArgInfo(help="project name for logger")] = 'ProGAP',
        output_dir: Annotated[str,  ArgInfo(help="directory to store the results")] = './output',
        version:    str = str(uuid1()), 
        config:     dict = {},
        debug:      bool = False,
        ) -> LoggerBase:
        
        if debug:
            logger = 'wandb'
            project += '-Test'
            console.warning(f'debug mode enabled! wandb logger is active for project {project}')

        self.logger_name = logger
        self.project = project
        self.output_dir = output_dir
        self._version = version
        self.config = config

        self._logger = None
    
    @property
    def logger(self):
        if self._logger is None:
            if self.logger_name == None:
                self._logger = DummyLogger()
            elif self.logger_name == 'wandb':
                import wandb
                os.environ["WANDB_SILENT"] = "true"
                settings = wandb.Settings(start_method="fork")
                self._logger = WandbLogger(
                    project=self.project,
                    save_dir=self.output_dir,
                    version=self._version,
                    log_model=False,
                    reinit=True, 
                    resume='allow', 
                    config=self.config, 
                    save_code=True,
                    settings=settings
                )
                self._logger.LOGGER_JOIN_CHAR = '/'
            
        return self._logger
    
    @property
    def name(self) -> str | None:
        return self.logger.name
    
    @property
    def version(self) -> int | str | None:
        return self.logger.version
    
    def log_summary(self, metrics: dict[str, object]):
        if self.logger_name == 'wandb':
            for metric, value in metrics.items():
                self.experiment.summary[metric] = value

    def log_hyperparams(self, params: Dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        return self.logger.log_hyperparams(params, *args, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        return self.logger.log_metrics(metrics, step)
    
    def set_prefix(self, prefix: str) -> None:
        self.logger._prefix = prefix
        
    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            return self.logger.__getattribute__(attr)
