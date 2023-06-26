from typing import Callable, Iterator, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Parameter
from core.nn.mlp import MLP
from torch_geometric.data import Data
from core.modules.base import Metrics, Phase, TrainableModule


class LPGNetModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 hops: int,
                 hidden_dim: int = 16,  
                 hidden_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = True,
                 **kwargs,
                 ):

        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm

        self.current_stage = 0

        self.mlps = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=num_classes,
                num_layers=hidden_layers+1,
                activation_fn=activation_fn,
                dropout=dropout,
                batch_norm=batch_norm,
                plain_last=True,
            ) for _ in range(hops + 1)
        )

    def set_stage(self, stage: int):
        self.current_stage = stage

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlps[self.current_stage](x)
        return x

    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]:
        x = data.x[data.batch_nodes]
        y = data.y[data.batch_nodes]
        
        preds: Tensor = self(x)
        acc = preds.detach().argmax(dim=1).eq(y).float().mean() * 100
        metrics = {f'{phase}/acc': acc}

        loss = None
        if phase != 'test':
            loss = F.cross_entropy(input=preds, target=y)
            metrics[f'{phase}/loss'] = loss.detach()

        return loss, metrics

    def predict(self, data: Data) -> tuple[Tensor, Tensor]:
        x = data.x[data.batch_nodes]
        x = self(x)                         # x is L in the paper
        y = torch.softmax(x, dim=-1)
        return x, torch.softmax(y, dim=-1)
        
    def reset_parameters(self):
        self.current_stage = 0
        for mlp in self.mlps:
            mlp.reset_parameters()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.mlps[self.current_stage].parameters(recurse=recurse)
        