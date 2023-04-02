from torch import Tensor
from torch_geometric.data import Data

def num_edges(data: Data) -> int:
    if hasattr(data, 'adj_t') and isinstance(data.adj_t, Tensor) and (data.adj_t.is_sparse or data.adj_t.is_sparse_csr):
        return data.adj_t._nnz()
    else:
        return data.num_edges
