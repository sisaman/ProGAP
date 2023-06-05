from typing import Annotated
from class_resolver.contrib.torch import activation_resolver
from core.args.utils import ArgInfo
from core.methods.base import NodeClassification
from core.modules.base import TrainableModule
from core.modules.mlp import MLPNodeClassifier


class SimpleMLP (NodeClassification):
    """Non-private MLP method"""

    def __init__(self,
                 num_classes,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 num_layers:      Annotated[int,   ArgInfo(help='number of MLP layers')] = 2,
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=['relu', 'selu', 'tanh'])] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 optimizer:       Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        super().__init__(num_classes, **kwargs)

    def configure_classifier(self) -> TrainableModule:
        return MLPNodeClassifier(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation_fn=activation_resolver.make(self.activation),
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
