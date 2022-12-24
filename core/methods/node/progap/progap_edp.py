import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node.progap.progap_inf import ProGAP
from core.privacy.mechanisms import GaussianMechanism, ComposedGaussianMechanism
from core.modules.base import Metrics


class EdgePrivProgGAP (ProGAP):
    """edge-private progressive method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[ProGAP])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.gm = GaussianMechanism(noise_scale=0.0)
        composed_mechanism = ComposedGaussianMechanism(
            noise_scale=1.0,
            mechanism_list=[self.gm],
            coeff_list=[len(self.modules) - 1],
        )
        
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)             # normalize
        x = matmul(adj_t, x)                        # aggregate
        x = self.gm.perturb(x, sensitivity=1)       # perturb
        return x
