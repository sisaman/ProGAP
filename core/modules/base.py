from typing import Optional
from torch import Tensor
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from core.typing import Metrics, Phase
from lightning import LightningModule


class TrainableModule(LightningModule, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]: pass

    @abstractmethod
    def predict(self, data: Data) -> Tensor: pass

    @abstractmethod
    def reset_parameters(self): pass

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, 'train')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, metrics = self.step(batch, 'val')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        _, metrics = self.step(batch, 'test')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
