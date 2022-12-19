from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.models import MLP
from torch_geometric.data import Data
from torch_geometric.nn import JumpingKnowledge
from core.modules.base import Metrics, Stage, TrainableModule


# TODO
# - add support for lstm jk


class ProgressiveModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,  
                 encoder_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 jk: Optional[str] = None,
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__()

        self.normalize = normalize
        self.jk = JumpingKnowledge(mode=jk) if jk else None

        self.encoder = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=encoder_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=False,
        )

        self.head = MLP(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def forward(self, x: Tensor, h_stack: Tensor) -> tuple[Tensor, Tensor]:
        """forward propagation

        Args:
            x (Tensor): node features
            h_stack (Tensor): historical embeddings with size (num_nodes, hidden_dim, num_stages)

        Returns:
            tuple[Tensor, Tensor]: node embeddings, node unnormalized predictions
        """
        h = x = self.encoder(x)
        
        if self.jk:
            xs = h_stack.unbind(dim=-1)
            x = self.jk(xs + (x,))

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
            
        y = self.head(x)
        return h, y

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        mask = data[f'{stage}_mask']
        x, y = data.x[mask], data.y[mask]
        h = data.h[mask] if self.jk else None
        preds = F.log_softmax(self(x, h)[1], dim=-1)
        acc = preds.argmax(dim=1).eq(y).float().mean() * 100
        metrics = {'acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        x, y = self(data.x, data.h if self.jk else None)
        return x, torch.softmax(y, dim=-1)
        
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.head.reset_parameters()
