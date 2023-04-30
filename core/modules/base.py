from typing import Optional
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from core.typing import Metrics, Phase
from class_resolver.contrib.torch import optimizer_resolver


class TrainableModule(Module, ABC):
    def __init__(self, optimizer: str, learning_rate: float, weight_decay: float):
        super().__init__()
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]: pass

    @abstractmethod
    def predict(self, data: Data) -> Tensor: pass

    @abstractmethod
    def reset_parameters(self): pass
    
    def configure_optimizers(self):
        optimizer = optimizer_resolver.make(
            query=self.optimizer_name, 
            params=self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer
