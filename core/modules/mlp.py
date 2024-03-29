from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.nn.mlp import MLP
from torch_geometric.data import Data
from core.modules.base import TrainableModule, Phase, Metrics


class MLPNodeClassifier(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,  
                 num_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 **kwargs,
                 ):

        super().__init__(**kwargs)

        self.model = MLP(
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]:
        x, y = data.x[data.batch_nodes], data.y[data.batch_nodes]
        preds = F.log_softmax(self(x), dim=-1)
        acc = preds.detach().argmax(dim=1).eq(y).float().mean() * 100
        metrics = {f'{phase}/acc': acc}

        loss = None
        if phase != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{phase}/loss'] = loss.detach()

        return loss, metrics

    def predict(self, data: Data) -> Tensor:
        h = self(data.x[data.batch_nodes])
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        return self.model.reset_parameters()