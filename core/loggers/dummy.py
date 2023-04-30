from torch.nn import Module
from core.loggers.base import LoggerBase


class DummyLogger(LoggerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def experiment(self): pass
    def log(self, metrics: dict[str, object]): pass
    def log_summary(self, metrics: dict[str, object]): pass
    def watch(self, model: Module, **kwargs): pass
    def finish(self): pass