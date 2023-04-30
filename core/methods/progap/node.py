import numpy as np
from typing import Annotated, Literal, Union
from torch.nn import BatchNorm1d, GroupNorm
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.data.loader.node import NodeDataLoader
from core.methods.progap.base import ProGAP
from core.nn.nap import NAP
from core.privacy.mechanisms.composed import ComposedNoisyMechanism
from core.privacy.algorithms.noisy_sgd import NoisySGD
from core.data.transforms.bound_degree import BoundOutDegree
from core.modules.base import Metrics, Phase
from opacus.validators import ModuleValidator
from opacus.validators.utils import register_module_fixer


@register_module_fixer([BatchNorm1d])
def fix(module: BatchNorm1d) -> GroupNorm:
    return GroupNorm(1, module.num_features, affine=module.affine)


class NodeLevelProGAP (ProGAP):
    """Node-level private ProGAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[ProGAP])]
                 ):

        super().__init__(num_classes, 
            batch_size=batch_size, 
            **kwargs
        )

        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

        # Noise std of NAP is set to 0, and will be calibrated later
        self.nap = NAP(noise_std=0, sensitivity=np.sqrt(max_degree))

        self.classifier = ModuleValidator.fix(self.classifier)
        ModuleValidator.validate(self.classifier, strict=True)

    def calibrate(self):
        n = self.num_stages

        self.noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.trainer.epochs,
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

        self.prepare_classifier()

    def prepare_classifier(self) -> None:
        original_set_stage = self.classifier.set_stage
        self.classifier.set_stage = self.wrap_set_stage(original_set_stage)

    def wrap_set_stage(self, original_set_stage):
        def set_stage(stage: int) -> None:
            original_set_stage(stage)
            self.noisy_sgd.prepare_trainable_module(self.classifier)
        return set_stage

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

    def data_loader(self, phase: Phase) -> NodeDataLoader:
        dataloader = super().data_loader(phase)
        if phase == 'train':
            dataloader.poisson_sampling = True
        return dataloader
