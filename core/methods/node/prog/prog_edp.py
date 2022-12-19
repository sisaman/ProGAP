import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node.prog.prog_inf import Progressive
from core.privacy.algorithms import PMA
from core.modules.base import Metrics


class EdgePrivProgressive (Progressive):
    """edge-private progressive method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[Progressive])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.hops = len(self.modules) - 1
        self.num_edges = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)
        
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')


    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)             # normalize
        x = matmul(adj_t, x)                        # aggregate
        x = self.pma_mechanism(x, sensitivity=1)    # perturb
        x = F.normalize(x, p=2, dim=-1)             # normalize
        return x
