from typing import Annotated
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from core import console
from core.args.utils import ArgInfo
from core.methods.gnn.base import StandardGNN
from core.privacy.algorithms.arr import AsymmetricRandResponse


class EdgeLevelGNN (StandardGNN):
    """edge-level private GNN method"""
    
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[StandardGNN])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.mechanism = AsymmetricRandResponse(eps=epsilon)

    def perturb_data(self, data: Data) -> Data:
        with console.status('perturbing graph structure'):
            adj_t = SparseTensor.from_torch_sparse_csr_tensor(data.adj_t)
            adj_t = self.mechanism(adj_t, chunk_size=1000)
            data.adj_t = adj_t.to_torch_sparse_csr_tensor()
        return data

    def setup(self, data: Data) -> None:
        super().setup(data)
        self.data = self.perturb_data(self.data)
