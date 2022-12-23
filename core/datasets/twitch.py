import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Twitch


def load_twitch(root: str, transform=None):
    names = ["DE", "EN", "ES", "FR", "PT", "RU"]
    dataset = [Twitch(name=name, root=root)[0] for name in names]
    num_nodes = [0] + [data.num_nodes for data in dataset][:-1]
    data_all = Data(
        x=torch.cat([data.x for data in dataset], dim=0),
        y=torch.cat([data.y for data in dataset], dim=0),
        edge_index=torch.cat([data.edge_index + n for data, n in zip(dataset, num_nodes)], dim=1),
    )
    if transform:
        data_all = transform(data_all)
    
    return [data_all]