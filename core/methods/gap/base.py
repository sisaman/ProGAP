import torch
from torch import Tensor
from typing import Annotated, Optional
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.base import NodeClassification
from core.nn.multi_mlp import MultiMLP
from core.modules.base import Metrics
from core.modules.cm import ClassificationModule
from core.modules.em import EncoderModule


class GAP (NodeClassification):
    """Non-private GAP method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops:            Annotated[int,   ArgInfo(help='number of hops', option='-k')] = 2,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 encoder_layers:  Annotated[int,   ArgInfo(help='number of encoder MLP layers')] = 2,
                 base_layers:     Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 head_layers:     Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 combine:         Annotated[str,   ArgInfo(help='combination type of transformed hops', choices=MultiMLP.supported_combinations)] = 'cat',
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)

        self.hops = hops
        self.encoder_layers = encoder_layers
        activation_fn = self.supported_activations[activation]

        self._encoder = EncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self._classifier = ClassificationModule(
            num_channels=hops+1,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def classifier(self) -> ClassificationModule:
        return self._classifier

    def reset_parameters(self):
        self._encoder.reset_parameters()
        super().reset_parameters()

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        self.data = data.to(self.device, non_blocking=True)
        
        # pre-train encoder
        if self.encoder_layers > 0:
            self.data = self.pretrain_encoder(self.data, prefix=prefix)

        # compute aggregations
        self.data = self.compute_aggregations(self.data)

        # train classifier
        return super().fit(self.data, prefix=prefix)

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is None or data == self.data:
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> Tensor:
        if data is None or data == self.data:
            data = self.data
        else:
            data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().predict(data)

    def _aggregate(self, x: Tensor, adj_t: Tensor) -> Tensor:
        return torch.spmm(adj_t, x)

    def _normalize(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2, dim=-1)

    def pretrain_encoder(self, data: Data, prefix: str) -> Data:
        console.info('pretraining encoder')
        self._encoder.to(self.device)
        
        self.trainer.fit(
            model=self._encoder,
            epochs=self.epochs,
            optimizer=self.configure_encoder_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )

        self.trainer.reset()
        data.x = self._encoder.predict(data)
        return data

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x = self._normalize(x)
                x_list.append(x)

            data.x = torch.stack(x_list, dim=-1)
        return data

    def configure_encoder_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self._encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)