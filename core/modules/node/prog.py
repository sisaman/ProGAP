from typing import Callable, Iterator, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.nn import MLP, JumpingKnowledge, ModuleList, Parameter
from torch_geometric.data import Data
from core.modules.base import Metrics, Phase, TrainableModule


class ProgressiveModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 num_stages: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 jk_mode: str = 'cat',
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = True,
                 layerwise: bool = False,
                 ):

        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        self.base_layers = base_layers
        self.head_layers = head_layers
        self.normalize = normalize
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.layerwise = layerwise

        self.current_stage = 0

        self.base = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=base_layers,
                activation_fn=activation_fn,
                dropout=dropout,
                batch_norm=batch_norm,
                plain_last=False,
            ) for _ in range(num_stages)
        )

        self.jk = ModuleList(
            JumpingKnowledge(
                mode=jk_mode,
                hidden_dim=hidden_dim,
                channels=hidden_dim,
                num_layers=2,
                num_heads=2
            ) for _ in range(num_stages)
        )

        self.head = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=num_classes,
                num_layers=head_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                plain_last=True,
            ) for _ in range(num_stages)
        )

    def set_stage(self, stage: int):
        self.current_stage = stage
        if self.layerwise:
            # freeze previous layers
            for i in range(self.current_stage):
                for param in self.base[i].parameters():
                    param.requires_grad = False
                for param in self.jk[i].parameters():
                    param.requires_grad = False
                for param in self.head[i].parameters():
                    param.requires_grad = False

    def forward(self, xs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """forward propagation

        Args:
            xs (list[Tensor]): list of aggregate node embeddings

        Returns:
            tuple[Tensor, Tensor]: node embeddings, node unnormalized predictions
        """

        for i in range(self.current_stage + 1):
            xs[i] = self.base[i](xs[i])
        
        h = xs[-1]
        x = self.jk[self.current_stage](torch.stack(xs, dim=-1))

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        y = self.head[self.current_stage](x)
        return h, y

    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]:
        mask = data[f'{phase}_mask']
        xs = [data[f'x{i}'][mask] for i in range(self.current_stage + 1)]
        y = data.y[mask]
        
        preds: Tensor = self(xs)[1]
        acc = preds.argmax(dim=1).eq(y).float().mean() * 100
        metrics = {'acc': acc}

        loss = None
        if phase != 'test':
            loss = F.cross_entropy(input=preds, target=y)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        xs = [data[f'x{i}'] for i in range(self.current_stage + 1)]
        x, y = self(xs)
        return x, torch.softmax(y, dim=-1)
        
    def reset_parameters(self):
        self.current_stage = 0
        for encoder in self.base:
            encoder.reset_parameters()
        for jk in self.jk:
            jk.reset_parameters()
        for head in self.head:
            head.reset_parameters()
        for param in super().parameters():
            param.requires_grad = True

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.layerwise:
            for i in range(self.current_stage):
                yield from self.base[i].parameters(recurse=recurse)
        yield from self.base[self.current_stage].parameters(recurse=recurse)
        yield from self.jk[self.current_stage].parameters(recurse=recurse)
        yield from self.head[self.current_stage].parameters(recurse=recurse)
        