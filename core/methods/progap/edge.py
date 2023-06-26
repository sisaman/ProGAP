import numpy as np
from typing import Annotated, Literal, Union

from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.progap.base import ProGAP
from core.nn.nap import NAP
from core.privacy.mechanisms.composed import ComposedGaussianMechanism


class EdgeLevelProGAP (ProGAP):
    """Edge-level private ProGAP method"""

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
        
        # Noise std of NAP is set to 0, and will be calibrated later
        self.nap = NAP(noise_std=0, sensitivity=1)

    def calibrate(self):
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                console.info('num_edges = %d' % self.num_edges)
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            
            if np.isinf(self.epsilon):
                self.noise_scale = 0.0
            else:
                composed_mechanism = ComposedGaussianMechanism(
                    noise_scale=1.0,
                    mechanism_list=[self.nap.gm],
                    coeff_list=[self.num_stages - 1],
                )
                self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

    def setup(self, data: Data) -> None:
        super().setup(data)
        m = self.data.num_edges
        if self.num_edges != m:
            self.num_edges = m
            self.calibrate()
