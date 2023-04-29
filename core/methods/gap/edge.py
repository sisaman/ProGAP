import numpy as np
import torch
from torch import Tensor
from typing import Annotated, Literal, Union
from core import console
from core.args.utils import ArgInfo
from core.methods.gap.base import GAP
from core.privacy.algorithms.pma import PMA
from core.modules.base import Metrics


class EdgeLevelGAP (GAP):
    """Edge-level private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[GAP])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)
        
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

    def fit(self) -> Metrics:
        m = self.data.num_edges
        if self.num_edges != m:
            self.num_edges = m
            self.calibrate()

        return super().fit()

    def _aggregate(self, x: Tensor, adj_t: Tensor) -> torch.Tensor:
        x = torch.spmm(adj_t, x)
        x = self.pma_mechanism(x, sensitivity=1)
        return x
