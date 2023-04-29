from typing import Annotated
import torch
from torch import Tensor
import torch.nn.functional as F
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
                 optimizer:       Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        self.num_classes = num_classes
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.base_layers = base_layers
        self.head_layers = head_layers
        self.combine = combine
        self.activation_fn = self.supported_activations[activation]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        super().__init__(num_classes, **kwargs)

        self.encoder_trainer = self.configure_trainer()
        self.encoder = EncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=self.activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    def reset(self):
        self.encoder.reset_parameters()
        self.encoder_trainer = self.configure_trainer()
        super().reset()

    def configure_classifier(self) -> ClassificationModule:
        return ClassificationModule(
            num_channels=self.hops+1,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            base_layers=self.base_layers,
            head_layers=self.head_layers,
            combination=self.combine,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self) -> Metrics:        
        if self.encoder_layers > 0:
            console.info('pretraining encoder')
            self.fit_encoder()
        else:
            console.info('skipping encoder pretraining')

        with console.status('computing aggregations'):
            self.compute_aggregations()

        console.info('training classifier')
        metrics = super().fit()
        return metrics

    def test(self) -> Metrics:
        if not getattr(self.data, 'aggs_ready', False):
            self.compute_aggregations()
        
        return super().test()

    def predict(self) -> Tensor:
        if not getattr(self.data, 'aggs_ready', False):
            self.compute_aggregations()
        
        return super().predict()

    def _aggregate(self, x: Tensor, adj_t: Tensor) -> Tensor:
        return torch.spmm(adj_t, x)

    def _normalize(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2, dim=-1)

    def fit_encoder(self):
        if self.encoder_trainer.logger is not None:
            self.encoder_trainer.logger.set_prefix('encoder')
        self.encoder_trainer.fit(
            model=self.encoder,
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=None,
        )
        if self.encoder_trainer.logger is not None:
            self.encoder_trainer.logger.set_prefix('')

    def compute_aggregations(self):
        # extract node embeddings from encoder
        x = self.encoder_trainer.predict(
            dataloader=self.data_loader('predict'),
        )
        x = self.to_device(x)
        x = F.normalize(x, p=2, dim=-1)
        x_list = [x]

        for _ in range(self.hops):
            x = self._aggregate(x, self.data.adj_t)
            x = self._normalize(x)
            x_list.append(x)

        self.data.x = torch.stack(x_list, dim=-1)
        self.data.aggs_ready = True
