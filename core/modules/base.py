from typing import Annotated, Optional
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from core.args.utils import ArgInfo
from core.typing import Metrics, Phase
from lightning import LightningModule
from class_resolver.contrib.torch import optimizer_resolver


class TrainableModule(LightningModule, ABC):
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

    def training_step(self, batch: Data, batch_idx):
        batch_size = batch.batch_nodes.size(0)
        loss, metrics = self.step(batch, 'train')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch.batch_nodes.size(0)
        _, metrics = self.step(batch, 'val' if dataloader_idx == 0 else 'test')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
    
    def test_step(self, batch, batch_idx):
        batch_size = batch.batch_nodes.size(0)
        _, metrics = self.step(batch, 'test')
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):
        return self.predict(batch)
    
    def configure_optimizers(self):
        optimizer = optimizer_resolver.make(
            query=self.optimizer_name, 
            params=self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
