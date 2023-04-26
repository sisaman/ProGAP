from typing import Literal, Optional, Union
import torch
from torch import Tensor
from collections.abc import Iterator
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_edge_index
from torch_geometric.transforms import ToSparseTensor


class NodeDataLoader:
    """ A fast dataloader for node-wise training.

    We have three settings:
    1. batch_size = 'full', hops = None
        The entire graph is used as a single batch.
    2. batch_size = int, hops = None
        The entire graph is returned at every iteration with a new phase mask corresponding to the nodes in the batch.
    3. batch_size = int, hops = int
        The k-hop subgraph is returned at every iteration corresponding to the nodes in the batch.

    Args:
        data (Data): The graph data object.
        subset (LongTensor or BoolTensor, optional): The subset of nodes to use for batching.
            If set to None, all nodes are used. (default: None)
        batch_size (int or 'full', optional): The batch size.
            If set to 'full', the entire graph is used as a single batch.
            (default: 'full')
        hops (int, optional): The number of hops to sample neighbors.
            If set to None, all neighbors are included. (default: None)
        shuffle (bool, optional): If set to True, the nodes are shuffled
            before batching. (default: True)
        drop_last (bool, optional): If set to True, the last batch is
            dropped if it is smaller than the batch size. (default: False)
        poisson_sampling (bool, optional): If set to True, poisson sampling
            is used to sample nodes. (default: False)
    """
    def __init__(self, 
                 data: Data, 
                 subset: Optional[Tensor] = None,
                 batch_size: Union[int, Literal['full']] = 'full', 
                 hops: Optional[int] = None,
                 shuffle: bool = True, 
                 drop_last: bool = False, 
                 poisson_sampling: bool = False):

        self.data = data
        self.batch_size = batch_size
        self.hops = hops
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.poisson_sampling = poisson_sampling
        self.device = data.x.device

        if subset is None:
            self.node_indices = torch.arange(data.num_nodes, device=self.device)
        else:
            if subset.dtype == torch.bool:
                self.node_indices = subset.nonzero().view(-1)
            else:
                self.node_indices = subset
        
        self.num_nodes = self.node_indices.size(0)
        self.is_sparse = not hasattr(data, 'edge_index')

        if not self.is_sparse and self.hops is not None and self.batch_size != 'full':
            self.edge_index = to_edge_index(self.data.adj_t)

    def __iter__(self) -> Iterator[Data]:
        if self.batch_size == 'full':
            data = Data(**self.data.to_dict())
            data.batch_nodes = self.node_indices
            yield data
            return

        if self.shuffle and not self.poisson_sampling:
            perm = torch.randperm(self.num_nodes, device=self.device)
            self.node_indices = self.node_indices[perm]

        for i in range(0, self.num_nodes, self.batch_size):
            if self.drop_last and i + self.batch_size > self.num_nodes:
                break

            if self.poisson_sampling:
                sampling_prob = self.batch_size / self.num_nodes
                sample_mask = torch.rand(self.num_nodes, device=self.device) < sampling_prob
                batch_nodes = self.node_indices[sample_mask]
            else:    
                batch_nodes = self.node_indices[i:i + self.batch_size]

            if self.hops is None:
                data = Data(**self.data.to_dict())
                data.batch_nodes = batch_nodes
            else:
                subset, batch_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx=batch_nodes, 
                    num_hops=self.hops, 
                    edge_index=self.edge_index, 
                    relabel_nodes=True, 
                    num_nodes=self.data.num_nodes
                )

                batch_mask = torch.zeros(subset.size(0), device=self.device, dtype=torch.bool)
                batch_mask[mapping] = True

                data = Data(
                    x=self.data.x[subset],
                    y=self.data.y[subset],
                    edge_index=batch_edge_index,
                )
                data.batch_nodes = mapping
                
                if self.is_sparse:
                    data = ToSparseTensor(layout=torch.sparse_csr)(data)
            
            yield data
            
    def __len__(self) -> int:
        if self.batch_size == 'full':
            return 1
        elif self.drop_last:
            return self.num_nodes // self.batch_size
        else:
            return (self.num_nodes + self.batch_size - 1) // self.batch_size
