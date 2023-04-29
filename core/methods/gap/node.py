import numpy as np
import torch
from torch import Tensor
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from opacus.optimizers import DPOptimizer
from core import console
from core.args.utils import ArgInfo
from core.data.loader.node import NodeDataLoader
from core.methods.gap.base import GAP
from core.privacy.mechanisms.composed import ComposedNoisyMechanism
from core.privacy.algorithms.pma import PMA
from core.privacy.algorithms.noisy_sgd import NoisySGD
from core.data.transforms.bound_degree import BoundOutDegree
from core.modules.base import Metrics, Phase


class NodeLevelGAP (GAP):
    """Node-level private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[GAP], exclude=['batch_norm'])]
                 ):

        super().__init__(num_classes, 
            batch_norm=False, 
            batch_size=batch_size, 
            **kwargs
        )
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm

        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)

        self.encoder_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.encoder_trainer.max_epochs,
            max_grad_norm=self.max_grad_norm,
        )

        self.classifier_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.trainer.max_epochs,
            max_grad_norm=self.max_grad_norm,
        )

        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[
                self.encoder_noisy_sgd, 
                self.pma_mechanism, 
                self.classifier_noisy_sgd
            ]
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self.encoder_noisy_sgd.prepare_lightning_module(self.encoder)
        self.classifier_noisy_sgd.prepare_lightning_module(self.classifier)

    def fit(self) -> Metrics:
        num_train_nodes = self.data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        return super().fit()

    def set_data(self, data: Data) -> Data:
        with console.status('bounding the number of neighbors per node'):
            data = BoundOutDegree(self.max_degree)(data)
        return super().set_data(data)

    def _aggregate(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = torch.spmm(adj_t, x)
        x = self.pma_mechanism(x, sensitivity=np.sqrt(self.max_degree))
        return x

    def data_loader(self, phase: Phase) -> NodeDataLoader:
        dataloader = super().data_loader(phase)
        if phase == 'train':
            dataloader.poisson_sampling = True
        return dataloader

