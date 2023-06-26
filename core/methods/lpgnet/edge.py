from typing import Annotated
from torch import Tensor

from core import console
from core.args.utils import ArgInfo
from core.methods.lpgnet.base import LPGNet
from core.privacy.mechanisms.commons import LaplaceMechanism


class EdgeLevelLPGNet (LPGNet):
    """Edge-level private ProGAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[LPGNet])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.lm = LaplaceMechanism(noise_scale=self.hops / self.epsilon)
        console.info(f'noise scale: {self.lm.params["noise_scale"]:.4f}\n')

    def get_degree_matrix(self, y: Tensor, adj_t: Tensor) -> Tensor:
        x = super().get_degree_matrix(y, adj_t)
        x = self.lm(x, sensitivity=1)
        return x
