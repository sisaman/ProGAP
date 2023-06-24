from typing import Annotated, Literal, Union
import numpy as np
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.data.loader.node import NodeDataLoader
from core.data.transforms.bound_degree import BoundOutDegree
from core.methods.gnn.base import StandardGNN
from core.modules.gnn import GNNNodeClassifier
from core.privacy.algorithms.gnn_sgd import GNNBasedNoisySGD
from core.privacy.mechanisms.commons import GaussianMechanism
from core.privacy.mechanisms.composed import ComposedNoisyMechanism
from core.typing import Phase


class NodeLevelGNN (StandardGNN):
    """node-level private GNN method"""
    
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[StandardGNN], 
                                                         exclude=['conv', 'mp_layers', 'sage_aggr', 'gcn_improved', 'gcn_add_self_loops', 'gcn_cached',
                                                                  'gcn_normalize', 'gat_heads', 'gat_concat', 'gat_negative_slope', 'gat_add_self_loops'])]
                 ):

        super().__init__(
            num_classes=num_classes, 
            conv='sage', 
            sage_aggr='sum',
            mp_layers=1, 
            batch_size=batch_size, 
            **kwargs
        )
        
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

        self.classifier: GNNNodeClassifier = ModuleValidator.fix(self.classifier)
        ModuleValidator.validate(self.classifier, strict=True)

    def calibrate(self):
        self.noisy_sgd = GNNBasedNoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size, 
            epochs=self.trainer.epochs,
            max_grad_norm=self.max_grad_norm,
            max_degree=self.max_degree,
        )

        self.noisy_aggr_gm = GaussianMechanism(noise_scale=0.0)
        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[self.noisy_sgd], 
            coeff_list=[1]
        )

        # if not hasattr(self, 'normalize_hook'):
        #     def normalize_hook(module, inputs):
        #         # if not module.training:
        #             # refer to SAGEConv.forward
        #         x = inputs[-1]['x'][0]
        #         x = F.normalize(x, p=2, dim=-1)
        #         inputs[-1]['x'] = (x, x)
        #         return inputs
        #     self.normalize_hook = self.classifier.gnn.model.convs[0].register_message_and_aggregate_forward_pre_hook(normalize_hook)

        # if not hasattr(self, 'noisy_aggr_hook'):
        #     self.noisy_aggr_hook = self.classifier.gnn.model.convs[0].register_message_and_aggregate_forward_hook(
        #         lambda module, inputs, output: 
        #             self.noisy_aggr_gm(data=output, sensitivity=np.sqrt(self.max_degree)) if not module.training else output
        #     )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self.noisy_sgd.prepare_trainable_module(self.classifier)

    def setup(self, data: Data) -> None:
        with console.status('bounding the number of neighbors per node'):
            data = BoundOutDegree(self.max_degree)(data)

        super().setup(data)
        num_train_nodes = self.data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

    def data_loader(self, phase: Phase) -> NodeDataLoader:
        dataloader = super().data_loader(phase)
        if phase == 'train':
            dataloader = self.noisy_sgd.prepare_dataloader(dataloader)
            dataloader.hops = 1
        return dataloader
