from torch_geometric.data import Data
from torch_geometric.utils import is_torch_sparse_tensor

def num_edges(data: Data) -> int:
    if hasattr(data, 'adj_t') and is_torch_sparse_tensor(data.adj_t):
        return data.adj_t._nnz()
    else:
        return data.num_edges
