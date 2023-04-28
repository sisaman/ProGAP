from abc import ABC, abstractmethod
from typing import Annotated, Literal, Optional, Union
from torch import Tensor
from torch_geometric.data import Data
from core.args.utils import ArgInfo
from core.data.loader.node import NodeDataLoader
from core import globals
from core.modules.base import TrainableModule, Metrics, Phase
from core.trainer.trainer import Trainer


class NodeClassification(ABC):
    def __init__(self, 
                 num_classes:      int, 
                 epochs:           Annotated[int,   ArgInfo(help='number of epochs for training')] = 100,
                 batch_size:       Annotated[Union[Literal['full'], int],   
                                                   ArgInfo(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval:  Annotated[bool,  ArgInfo(help='if true, then model uses full-batch evaluation')] = True,
                 **trainer_args:   Annotated[dict,  ArgInfo(help='extra options passed to the trainer class', bases=[Trainer])]
                 ):

        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.trainer_args = trainer_args

        self.data = None  # data is kept for caching purposes
        self.classifier = self.configure_classifier()
        self.trainer = self.configure_trainer()

    def reset_parameters(self):
        self.classifier.reset_parameters()
        self.trainer.reset()
        self.data = None

    @abstractmethod
    def configure_classifier(self) -> TrainableModule:
        """Configure the classifier module."""

    def fit(self, data: Data) -> Metrics:
        """Fit the model to the given data."""
        self.data = data
        train_metrics = self.train_classifier(self.data)
        test_metrics = self.test(self.data)
        return {**train_metrics, **test_metrics}

    def test(self, data: Optional[Data] = None) -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data

        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data

        return self.trainer.predict(
            dataloader=self.data_loader(data, 'predict'),
        )

    def train_classifier(self, data: Data) -> Metrics:        
        metrics = self.trainer.fit(
            model=self.classifier,
            epochs=self.epochs,
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=self.data_loader(data, 'test') if globals['debug'] else None,
        )

        return metrics

    def configure_trainer(self) -> Trainer:
        trainer = Trainer(
            monitor='val/acc', 
            monitor_mode='max', 
            **self.trainer_args,
        )
        return trainer

    def data_loader(self, data: Data, phase: Phase) -> NodeDataLoader:
        """Return a dataloader for the given phase."""
        
        batch_size = 'full' if (phase != 'train' and self.full_batch_eval) else self.batch_size
        subset = data[f'{phase}_mask'] if phase != 'predict' else None
        shuffle = phase == 'train'

        dataloader = NodeDataLoader(
            data=data, 
            subset=subset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=True,
            poisson_sampling=False,
        )

        return dataloader
