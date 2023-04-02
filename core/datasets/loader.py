import os
from functools import partial
from typing import Annotated
import torch
from core import console
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Amazon as AmazonDataset, FacebookPagePage, Flickr
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit
from core.args.utils import ArgInfo
from core.data.transforms import FilterClassByCount
from core.data.transforms import RemoveSelfLoops
from core.data.transforms import RemoveIsolatedNodes
from core.data.utils import num_edges
from core.datasets import Facebook, Amazon
from core.datasets.facebook import Facebook100
from core.datasets.twitch import load_twitch
from core.datasets.tweibo import TWeibo
from core.utils import dict2table


def load_wenet(root: str, transform=None) -> Data:
    data = torch.load(os.path.join(root, 'data.pt'))
    data = data if transform is None else transform(data)
    return [data]


class DatasetLoader:
    supported_datasets = {
        'reddit': partial(Reddit, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'amazon': partial(Amazon, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'facebook': partial(Facebook, name='UIllinois20', target='year', 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=1000, remove_unlabeled=True)
            ])
        ),
        'computers': partial(AmazonDataset, name="Computers",
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'twitch': partial(load_twitch,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'facebook-pages': partial(FacebookPagePage,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'flickr': partial(Flickr,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'facebook-100': partial(Facebook100, target='year', 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'tweibo': partial(TWeibo,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'wenet': partial(load_wenet,
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                # FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
    }

    def __init__(self,
                 dataset:    Annotated[str, ArgInfo(help='name of the dataset', choices=supported_datasets)] = 'facebook',
                 data_dir:   Annotated[str, ArgInfo(help='directory to store the dataset')] = './datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False) -> Data:
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor(layout=torch.sparse_csr)])(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data: Data):
        nodes_degree: torch.Tensor = data.adj_t.to_sparse_coo().sum(dim=1).to_dense() # in degree
        baseline: float = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
        train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{num_edges(data):,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max() + 1)}',
            'mean degree': f'{nodes_degree.mean():.2f}',
            'median degree': f'{nodes_degree.median()}',
            'train/val/test (%)': f'{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}',
            'baseline acc (%)': f'{baseline:.2f}'
        }

        table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
        console.info(table)
        console.print()
