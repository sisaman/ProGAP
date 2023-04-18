from typing import Optional
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from core.typing import Metrics, Phase



class TrainableModule(Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]: pass

    @abstractmethod
    def predict(self, data: Data) -> Tensor: pass

    @abstractmethod
    def reset_parameters(self): pass
