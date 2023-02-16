from typing import Callable, Optional
from typing_extensions import Self
import torch
from torch import Tensor
import torch.nn.functional as F
from core.models import MLP, JumpingKnowledge
from torch_geometric.data import Data
from core.modules.base import Metrics, Stage, TrainableModule


class ProgressiveModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,  
                 encoder_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 jk_mode: Optional[str] = None,
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__()

        self.normalize = normalize
        self.jk_mode = jk_mode
        if jk_mode is not None:
            self.jk = JumpingKnowledge(
                mode=jk_mode, 
                hidden_dim=hidden_dim, 
                channels=hidden_dim,
                num_layers=2, 
                num_heads=2
            )

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

    def forward(self, x: Tensor, xs: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """forward propagation

        Args:
            x (Tensor): node features
            xs (Tensor, optional): historical embeddings with size (num_nodes, hidden_dim, num_stages)

        Returns:
            tuple[Tensor, Tensor]: node embeddings, node unnormalized predictions
        """
        x = self.encoder(x)
        h = x

        if self.jk_mode is not None:
            xs = torch.cat([xs, x.unsqueeze(-1)], dim=-1)
            x = self.jk(xs)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
            
        y = self.head(x)
        return h, y

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        mask = data[f'{stage}_mask']
        x, y = data.x[mask], data.y[mask]
        xs = data.xs[mask] if self.jk_mode else None
        
        preds = F.log_softmax(self(x, xs)[1], dim=-1)
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
        x, y = self(data.x, getattr(data, 'xs', None))
        return x, torch.softmax(y, dim=-1)

    def transfer(self, pm: Self, encoder: bool = True, head: bool = True, jk: bool = True):
        if encoder:
            self.encoder.load_state_dict(pm.encoder.state_dict())

        if head:
            self.head.load_state_dict(pm.head.state_dict())

        if jk and self.jk_mode not in (None, 'cat'):
            self.jk.load_state_dict(pm.jk.state_dict())
        
    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()
