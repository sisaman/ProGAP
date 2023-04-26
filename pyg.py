import time
import os.path as osp

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torchmetrics import Accuracy

from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GraphSAGE
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.strategies import SingleDeviceStrategy
from core.datasets.loader import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import CheckpointIO
from core.data.loader.node import NodeDataLoader


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 256, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers,
                             out_channels, dropout=dropout,
                             norm=BatchNorm1d(hidden_channels))

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.final_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, adj_t):
        return self.gnn(x, adj_t)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.adj_t)[data.batch_nodes]
        y = data.y[data.batch_nodes]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            metric = 'val_acc'
            acc = self.val_acc
        else:
            metric = 'test_acc'
            acc = self.test_acc
            
        y_hat = self(data.x, data.adj_t)[data.batch_nodes]
        y = data.y[data.batch_nodes]
        acc(y_hat.softmax(dim=-1), y)
        self.log(metric, acc, prog_bar=True, on_step=False,
                 on_epoch=True, add_dataloader_idx=False)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.adj_t)[data.batch_nodes]
        y = data.y[data.batch_nodes]
        self.final_acc(y_hat.softmax(dim=-1), y)
        self.log('final_test_acc', self.final_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class InMemoryCheckpoint(CheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_dict = {}

    def load_checkpoint(self, path, map_location = None):
        return self.checkpoint_dict[path]
    
    def save_checkpoint(self, checkpoint, path, storage_options = None):
        self.checkpoint_dict[path] = checkpoint

    def remove_checkpoint(self, path):
        del self.checkpoint_dict[path]


if __name__ == '__main__':
    pl.seed_everything(12345)

    data = DatasetLoader('facebook').load()

    train_dataloader = NodeDataLoader(data=data, subset=data.train_mask)
    val_dataloader = NodeDataLoader(data=data, subset=data.val_mask)
    test_dataloader = NodeDataLoader(data=data, subset=data.test_mask)
    final_dataloader = NodeDataLoader(data=data, subset=data.test_mask)

    num_classes = data.y.max().item() + 1
    model = Model(data.num_features, num_classes)
    checkpoint = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max', dirpath='checkpoints')
    # logger = WandbLogger()

    trainer = pl.Trainer(devices=1, max_epochs=5, callbacks=[checkpoint, RichProgressBar()], deterministic=True, accelerator='cpu',
                        #  logger=WandbLogger(), 
                         plugins=InMemoryCheckpoint(), 
                         log_every_n_steps=1)

    start = time.time()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader, test_dataloader])
    trainer.test(ckpt_path='best', dataloaders=final_dataloader)
    end = time.time()
    duration = end - start
    # logger.log_metrics({'duration': duration})
    print(checkpoint.best_model_path)
