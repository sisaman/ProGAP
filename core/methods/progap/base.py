import torch
from typing import Annotated, Optional
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from class_resolver.contrib.torch import activation_resolver
from core import console
from core.args.utils import ArgInfo
from core.nn.jk import JumpingKnowledge as JK
from core.methods.base import NodeClassification
from core.nn.nap import NAP
from core.modules.base import Metrics, TrainableModule
from core.modules.prog import ProgressiveModule
from core import globals


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
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.num_stages = depth + 1

        self.model = ProgressiveModule(
            num_classes=num_classes,
            num_stages=self.num_stages,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            normalize=True,
            jk_mode=jk,
            activation_fn=activation_resolver.make(activation),
            dropout=dropout,
            batch_norm=batch_norm,
            layerwise=layerwise,
        )

        self.nap = NAP(noise_std=0, sensitivity=1)

    def reset_parameters(self):
        self.model.reset_parameters()
        self.trainer.reset()
        self.data = None

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        
        if data is None or data == self.data:
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            self.pipeline(data, train=False)

        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
            prefix=prefix,
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        
        if data is None or data == self.data:
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            self.pipeline(data, train=False)
        
        return self.model.predict(data)

    def _train(self, data: Data, prefix: str = '') -> Metrics:
        return self.pipeline(data, train=True, prefix=prefix)

    def pipeline(self, data: Data, train: bool=False, prefix: str = '') -> Optional[Metrics]:
        n = self.num_stages
        data.x0 = data.x
        self.set_stage(0)
        
        for i in range(n):
            if i > 0:
                x, _ = self.model.predict(data)
                x = self.nap(x, data.adj_t)
                data[f'x{i}'] = x
            
            self.set_stage(i)

            if train:
                console.info(f'Fitting stage {i+1} of {n}')
                self.trainer.reset()
                self.model.to(self.device)
                metrics = self.trainer.fit(
                    model=self.model,
                    epochs=self.epochs,
                    optimizer=self.configure_optimizer(self.model),
                    train_dataloader=self.data_loader(data, 'train'), 
                    val_dataloader=self.data_loader(data, 'val'),
                    test_dataloader=self.data_loader(data, 'test') if globals['debug'] else None,
                    checkpoint=True,
                    prefix=prefix,
                )

        return metrics if train else None
    
    def set_stage(self, stage: int) -> None:
        self.model.set_stage(stage)
    
    def configure_optimizer(self, module: TrainableModule) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
