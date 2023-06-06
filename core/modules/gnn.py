from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.nn.mlp import MLP
from torch_geometric.data import Data
from core.nn.gnn import GNN
from core.modules.base import Metrics, Phase, TrainableModule


class GNNNodeClassifier(TrainableModule):
    def __init__(self, *,
                 num_classes: int, 
                 hidden_dim: int = 16, 
                 base_layers: int = 1, 
                 mp_layers: int = 2, 
                 head_layers: int = 0, 
                 conv: str = 'sage',
                 conv_kwargs: dict = {},
                 jk: str = None,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 **kwargs
                 ):

        assert mp_layers > 0, 'Must have at least one message passing layer'

        super().__init__(**kwargs)

        self.base_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=base_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=False,
        )

        self.gnn = GNN(
            conv=conv,
            output_dim=num_classes if head_layers == 0 else hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mp_layers,
            dropout=dropout,
            jk=jk,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=head_layers == 0,
            **conv_kwargs,
        )

        self.head_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=head_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = self.base_mlp(x)
        x = self.gnn(x, adj_t)
        x = self.head_mlp(x)
        return x

    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]:
        h = self(data.x, data.adj_t)
        h, y = h[data.batch_nodes], data.y[data.batch_nodes]
        preds = F.log_softmax(h, dim=-1)
        acc = preds.detach().argmax(dim=1).eq(y).float().mean() * 100
        metrics = {f'{phase}/acc': acc}

        loss = None
        if phase != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{phase}/loss'] = loss.detach()

        return loss, metrics

    def predict(self, data: Data) -> Tensor:
        h = self(data.x, data.adj_t)[data.batch_nodes]
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        self.base_mlp.reset_parameters()
        self.gnn.reset_parameters()
        self.head_mlp.reset_parameters()