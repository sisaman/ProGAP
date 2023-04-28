import torch
from typing import Annotated, Optional
from torch_geometric.data import Data
from class_resolver.contrib.torch import activation_resolver
from core import console
from core.args.utils import ArgInfo
from core.nn.jk import JumpingKnowledge as JK
from core.methods.base import NodeClassification
from core.nn.nap import NAP
from core.modules.base import Metrics
from core.modules.prog import ProgressiveModule


class ProGAP (NodeClassification):
    """Non-private ProGAP method"""

    def __init__(self,
                 num_classes,
                 depth:         Annotated[int,   ArgInfo(help='model depth', option='-k')] = 2,
                 hidden_dim:    Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 base_layers:   Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 head_layers:   Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 jk:            Annotated[str,   ArgInfo(help='jumping knowledge combination scheme', choices=JK.supported_modes)] = 'cat',
                 activation:    Annotated[str,   ArgInfo(help='type of activation function', choices=['relu', 'selu', 'tanh'])] = 'selu',
                 dropout:       Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:    Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 layerwise:     Annotated[bool,  ArgInfo(help='if true, then model uses layerwise training')] = False,
                 optimizer:       Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        self.num_classes = num_classes
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.base_layers = base_layers
        self.head_layers = head_layers
        self.jk = jk
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layerwise = layerwise
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.num_stages = depth + 1
        self.nap = NAP(noise_std=0, sensitivity=1)
        self.classifier: ProgressiveModule
        super().__init__(num_classes, **kwargs)

    def configure_classifier(self) -> ProgressiveModule:
        return ProgressiveModule(
            num_classes=self.num_classes,
            num_stages=self.num_stages,
            hidden_dim=self.hidden_dim,
            base_layers=self.base_layers,
            head_layers=self.head_layers,
            normalize=True,
            jk_mode=self.jk,
            activation_fn=activation_resolver.make(self.activation),
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            layerwise=self.layerwise,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def test(self, data: Optional[Data] = None) -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        
        if data is None or data == self.data:
            data = self.data
        else:
            self.pipeline(data, train=False)

        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        
        if data is None or data == self.data:
            data = self.data
        else:
            self.pipeline(data, train=False)
        
        return self.trainer.predict(
            dataloader=self.data_loader(data, 'predict'),
        )

    def train_classifier(self, data: Data) -> Metrics:
        return self.pipeline(data, train=True)

    def pipeline(self, data: Data, train: bool=False) -> Optional[Metrics]:
        n = self.num_stages
        data.x0 = data.x
        
        for i in range(n):
            if i > 0:
                x, _ = self.trainer.predict(dataloader=self.data_loader(data, 'predict'))
                x = self.nap(x, data.adj_t)
                data[f'x{i}'] = x
            
            self.classifier.set_stage(i)

            if train:
                console.info(f'Fitting stage {i+1} of {n}')
                metrics = super().train_classifier(data)

        return metrics if train else None

