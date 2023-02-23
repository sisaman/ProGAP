import numpy as np
from typing import Annotated, Literal, Union
from torch.nn import BatchNorm1d, GroupNorm
from torch_geometric.data import Data
from opacus.optimizers import DPOptimizer
from core import console
from core.args.utils import ArgInfo
from core.data.loader import NodeDataLoader
from core.methods.node.progap.progap_inf import ProGAP
from core.models.nap import NAP
from core.privacy.mechanisms import ComposedNoisyMechanism
from core.privacy.algorithms import NoisySGD
from core.data.transforms import BoundOutDegree
from core.modules.base import Metrics, Stage, TrainableModule
from opacus.validators import ModuleValidator
from opacus.validators.utils import register_module_fixer


@register_module_fixer([BatchNorm1d])
def fix(module: BatchNorm1d) -> GroupNorm:
    return GroupNorm(1, module.num_features, affine=module.affine)


class NodePrivProGAP (ProGAP):
    """node-private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[ProGAP], exclude=['batch_norm'])]
                 ):

        super().__init__(num_classes, 
            batch_norm=True,            # will be replaced with GroupNorm by ModuleValidator
            batch_size=batch_size, 
            **kwargs
        )

        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

        self.modules = [ModuleValidator.fix(module) for module in self.modules]
        for module in self.modules:
            ModuleValidator.validate(module, strict=True)

        # Noise std of NAP is set to 0, and will be calibrated later
        self.nap = NAP(noise_std=0, sensitivity=np.sqrt(max_degree))

    def calibrate(self):
        n = len(self.modules)

        self.noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
        )

        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=1.0,
            mechanism_list=[
                self.nap.gm, 
                self.noisy_sgd
            ],
            coeff_list=[n - 1, n],
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        for module in self.modules:
            self.noisy_sgd.prepare_module(module)

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        with console.status('bounding the number of neighbors per node'):
            data = BoundOutDegree(self.max_degree)(data)

        return super().fit(data, prefix=prefix)

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.poisson_sampling = True
        return dataloader

    def configure_optimizer(self, module: TrainableModule) -> DPOptimizer:
        optimizer = super().configure_optimizer(module)
        optimizer = self.noisy_sgd.prepare_optimizer(optimizer)
        return optimizer
