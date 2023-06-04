from abc import ABC, abstractmethod
from torch.nn import Module


class LoggerBase(ABC):
    @property
    @abstractmethod
    def experiment(self): pass

    @abstractmethod
    def log(self, metrics: dict[str, object]): pass
    
    @abstractmethod
    def log_summary(self, metrics: dict[str, object]): pass

    @abstractmethod
    def watch(self, model: Module, **kwargs): pass
    
    @abstractmethod
    def finish(self): pass
