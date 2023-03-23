import torch
import torch.utils.cpp_extension
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_torch_csr_tensor
from pyg_lib.sampler import neighbor_sample


class BoundOutDegree(BaseTransform):
    def __init__(self, max_out_degree: int):
        self.num_neighbors = max_out_degree

    def __call__(self, data: Data) -> Data:
        data = self.sample(data)
        return data

    def sample(self, data: Data) -> Data:
        device = data.adj_t.device
        colptr = data.adj_t.crow_indices()
        row = data.adj_t.col_indices()
        seed = torch.arange(0, data.num_nodes-1, dtype=int)
        out = neighbor_sample(
            colptr.cpu(),
            row.cpu(),
            seed=seed,
            num_neighbors=[self.num_neighbors],
            csc=True,
            replace=False,
            directed=True,
            disjoint=False,
            return_edge_id=False,
        )
        row, col, _, _, _, _ = out
        edge_index = torch.stack([row, col], dim=0)
        adj_t = to_torch_csr_tensor(edge_index, size=(data.num_nodes, data.num_nodes))
        data.adj_t = adj_t.to(device)
        return data
