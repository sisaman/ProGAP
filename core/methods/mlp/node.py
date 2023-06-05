import numpy as np
from typing import Annotated, Literal, Union
from opacus.validators import ModuleValidator
from core import console
from core.args.utils import ArgInfo
from core.methods.mlp.edge import SimpleMLP
from core.privacy.algorithms.noisy_sgd import NoisySGD
from core.modules.base import Metrics, Phase
from core.data.loader.node import NodeDataLoader


class PrivateMLP (SimpleMLP):
    """private MLP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[SimpleMLP])]
                 ):

        super().__init__(num_classes, 
            batch_size=batch_size, 
            **kwargs
        )

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None         # will be used to set delta if it is 'auto'

        self.classifier = ModuleValidator.fix(self.classifier)
        ModuleValidator.validate(self.classifier, strict=True)

    def calibrate(self):
        self.noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size, 
            epochs=self.trainer.epochs,
            max_grad_norm=self.max_grad_norm,
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e', delta)
            
            self.noise_scale = self.noisy_sgd.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self.noisy_sgd.prepare_trainable_module(self.classifier)

    def fit(self) -> Metrics:
        num_train_nodes = self.data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        return super().fit()

    def data_loader(self, phase: Phase) -> NodeDataLoader:
        dataloader = super().data_loader(phase)
        if phase == 'train':
            dataloader.poisson_sampling = True
        return dataloader
