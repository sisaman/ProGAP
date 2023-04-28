import time

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torchmetrics import Accuracy

from torch_geometric.nn import GraphSAGE
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger


from core import console
from core.datasets.loader import DatasetLoader
from core.data.loader.node import NodeDataLoader
from core.trainer.checkpoint import InMemoryCheckpoint


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
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('final_test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


if __name__ == '__main__':
    pl.seed_everything(12345)

    data = DatasetLoader('facebook').load()

    train_dataloader = NodeDataLoader(data=data, subset=data.train_mask)
    val_dataloader = NodeDataLoader(data=data, subset=data.val_mask)
    test_dataloader = NodeDataLoader(data=data, subset=data.test_mask)

    num_classes = data.y.max().item() + 1
    model = Model(data.num_features, num_classes)
    checkpoint = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max', dirpath='checkpoints')
    checkpoint_plugin = InMemoryCheckpoint()
    logger = WandbLogger()

    trainer = pl.Trainer(devices=1, max_epochs=100, callbacks=[checkpoint, RichProgressBar()], deterministic=True, accelerator='gpu',
                         logger=logger, 
                         plugins=InMemoryCheckpoint(), 
                         log_every_n_steps=1)

    start = time.time()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader, test_dataloader])
    trainer.test(ckpt_path='best', dataloaders=test_dataloader)
    
    end = time.time()
    duration = end - start
    logger.log_metrics({'duration': duration})
