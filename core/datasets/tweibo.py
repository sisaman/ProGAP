import torch
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
import pandas as pd
import scipy.sparse as sp


class TWeibo(AttributedGraphDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, name='tweibo', transform=transform, pre_transform=pre_transform)

    def process(self):
        x = sp.load_npz(self.raw_paths[0])
        x = torch.from_numpy(x.todense()).to(torch.float)
        df = pd.read_csv(self.raw_paths[1], header=None, sep=None, engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()
        df = pd.read_csv('labels.txt', delimiter='\t', header=None).rename(columns={0: 'node', 1: 'label'})
        y = torch.from_numpy(df['label'].values).to(torch.long) - 1
        nodes = torch.from_numpy(df['node'].values).to(torch.long)
        edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        x = x[nodes]
        edge_index = to_undirected(edge_index=edge_index, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
