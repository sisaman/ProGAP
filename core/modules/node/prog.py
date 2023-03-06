from typing import Callable, Iterator, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.nn import MLP, JumpingKnowledge, ModuleList, Parameter
from torch_geometric.data import Data
from core.modules.base import Metrics, Stage, TrainableModule


class ProgressiveModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 num_phases: int,
                 hidden_dim: int = 16,  
                 encoder_layers: int = 2, 
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
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.head_layers = head_layers
        self.normalize = normalize
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.layerwise = layerwise

        self.current_phase = 0

        self.encoders = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=encoder_layers,
                activation_fn=activation_fn,
                dropout=dropout,
                batch_norm=batch_norm,
                plain_last=False,
            ) for _ in range(num_phases)
        )

        self.jks = ModuleList(
            JumpingKnowledge(
                mode=jk_mode,
                hidden_dim=hidden_dim,
                channels=hidden_dim,
                num_layers=2,
                num_heads=2
            ) for _ in range(num_phases)
        )

        self.heads = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=num_classes,
                num_layers=head_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                plain_last=True,
            ) for _ in range(num_phases)
        )

    def set_phase(self, phase: int):
        self.current_phase = phase

    def forward(self, xs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """forward propagation

        Args:
            xs (list[Tensor]): list of aggregate node embeddings

        Returns:
            tuple[Tensor, Tensor]: node embeddings, node unnormalized predictions
        """

        for i in range(self.current_phase + 1):
            xs[i] = self.encoders[i](xs[i])
        
        h = xs[-1]
        x = self.jks[self.current_phase](torch.stack(xs, dim=-1))

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        y = self.heads[self.current_phase](x)
        return h, y

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        mask = data[f'{stage}_mask']
        xs = [data[f'x{i}'][mask] for i in range(self.current_phase + 1)]
        y = data.y[mask]
        
        preds: Tensor = self(xs)[1]
        acc = preds.argmax(dim=1).eq(y).float().mean() * 100
        metrics = {'acc': acc}

        loss = None
        if stage != 'test':
            loss = F.cross_entropy(input=preds, target=y)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        xs = [data[f'x{i}'] for i in range(self.current_phase + 1)]
        x, y = self(xs)
        return x, torch.softmax(y, dim=-1)
        
    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        for jk in self.jks:
            jk.reset_parameters()
        for head in self.heads:
            head.reset_parameters()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.layerwise:
            yield from self.encoders[self.current_phase].parameters(recurse=recurse)
        else:
            for i in range(self.current_phase + 1):
                yield from self.encoders[i].parameters(recurse=recurse)
        yield from self.jks[self.current_phase].parameters(recurse=recurse)
        yield from self.heads[self.current_phase].parameters(recurse=recurse)
        