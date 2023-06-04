import torch
from ogb.nodeproppred import PygNodePropPredDataset


class Amazon(PygNodePropPredDataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root=root, name='ogbn-products', transform=None, pre_transform=pre_transform)
        self._data.__num_nodes__ = self._data.x.size(0)
        self._data.y = self._data.y.view(-1)
        split_idx = self.get_idx_split()
        for split, idx in split_idx.items():
            mask = torch.zeros(self._data.num_nodes, dtype=bool).scatter(0, idx, True)
            split = 'val' if split == 'valid' else split
            self._data[f'{split}_mask'] = mask
        if transform:
            self._data = transform(self._data)