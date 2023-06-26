import torch
from torch import Tensor
from typing import Annotated, Optional
from class_resolver.contrib.torch import activation_resolver
from core import console
from core.args.utils import ArgInfo
from core.modules.lpgnet import LPGNetModule
from core.methods.base import NodeClassification
from core.modules.base import Metrics


class LPGNet (NodeClassification):
    """Non-private LPGNet method"""

    def __init__(self,
                 num_classes,
                 hops:          Annotated[int,   ArgInfo(help='number of additional mlps', option='-k')] = 2,
                 hidden_dim:    Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 hidden_layers: Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 activation:    Annotated[str,   ArgInfo(help='type of activation function', choices=['relu', 'selu', 'tanh'])] = 'selu',
                 dropout:       Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:    Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 optimizer:     Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate: Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:  Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        self.num_classes = num_classes
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.num_stages = hops + 1
        self.classifier: LPGNetModule
        super().__init__(num_classes, **kwargs)

    def configure_classifier(self) -> LPGNetModule:
        return LPGNetModule(
            num_classes=self.num_classes,
            hops=self.hops,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=activation_resolver.make(self.activation),
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    
    def get_degree_matrix(self, y: Tensor, adj_t: Tensor) -> Tensor:
        idxmax = y.argmax(dim=-1, keepdim=True)
        y = y.zero_().scatter_(1, idxmax, 1)
        x = torch.spmm(adj_t, y)
        return x
    
    def pipeline(self, fit: bool=False) -> Optional[Metrics]:
        n = self.num_stages
        
        for i in range(n):
            if i > 0:
                l, y = self.trainer.predict(dataloader=self.data_loader('predict'))
                x = self.get_degree_matrix(y, self.data.adj_t)
                if i == 1:
                    self.data.x = torch.cat([l, x], dim=-1)
                else:
                    self.data.x = torch.cat([self.data.x, l, x], dim=-1)

            self.classifier.set_stage(i)

            if fit:
                console.info(f'Fitting MLP {i+1} of {n}')
                self.trainer = self.configure_trainer()
                metrics = super().fit()

        self.data.ready = True
        return metrics if fit else None
    
    def fit(self) -> Metrics:
        return self.pipeline(fit=True)

    def test(self) -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        if not getattr(self.data, 'ready', False):
            self.pipeline(fit=False)
        
        return super().test()

    def predict(self) -> torch.Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        if not getattr(self.data, 'ready', False):
            self.pipeline(fit=False)
        
        return super().predict()

