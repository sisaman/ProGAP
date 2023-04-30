from abc import ABC, abstractmethod
from typing import Annotated, Literal, Union
from torch import Tensor
from torch_geometric.data import Data
from core.args.utils import ArgInfo
from core.data.loader.node import NodeDataLoader
from core import globals, console
from core.modules.base import TrainableModule, Metrics, Phase
from core.trainer.trainer import Trainer


class NodeClassification(ABC):
    def __init__(self, 
                 num_classes:      int,
                 batch_size:       Annotated[Union[Literal['full'], int],   
                                                    ArgInfo(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval:  Annotated[bool,  ArgInfo(help='if true, then model uses full-batch evaluation')] = True,
                 **trainer_args:   Annotated[dict,  ArgInfo(help='extra options passed to the trainer class', bases=[Trainer])]
                 ):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.trainer_args = trainer_args

        self.data = None
        self.classifier = self.configure_classifier()
        self.trainer = self.configure_trainer()

    def reset(self):
        self.classifier.reset_parameters()
        self.trainer = self.configure_trainer()
        self.data = None

    @abstractmethod
    def configure_classifier(self) -> TrainableModule:
        """Configure the classifier module."""

    def configure_trainer(self) -> Trainer:
        """Configure the trainer"""
        trainer = Trainer(**self.trainer_args)
        return trainer

    def set_data(self, data: Data) -> Data:
        """Set the data for the method."""
        with console.status('moving data to device'):
            data = self.to_device(data)
        
        self.data = Data(**data.to_dict())
        return self.data

    def run(self, data: Data, fit: bool = True, test: bool = True) -> Metrics:
        """Setup the model for the given data, and run the training and testing procedures."""
        metrics = {}
        self.set_data(data)
        
        if fit:
            train_metrics = self.fit()
            metrics.update(train_metrics)
        if test:
            test_metrics = self.test()
            metrics.update(test_metrics)
        
        return metrics
    
    def fit(self) -> Metrics:    
        """Fit the method to the given data."""    
        self.trainer.current_epoch
        metrics = self.trainer.fit(
            model=self.classifier,
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test') if globals['debug'] else None,
        )

        return metrics

    def test(self) -> Metrics:
        """Test the method on the given data."""
        return self.trainer.test(
            dataloader=self.data_loader('test')
        )

    def predict(self) -> Tensor:
        """Predict output for the given data."""
        return self.trainer.predict(
            dataloader=self.data_loader('predict'),
        )

    def data_loader(self, phase: Phase) -> NodeDataLoader:
        """Return a dataloader for the given phase."""
        batch_size = 'full' if (phase != 'train' and self.full_batch_eval) else self.batch_size
        subset = self.data[f'{phase}_mask'] if phase != 'predict' else None
        shuffle = phase == 'train'

        dataloader = NodeDataLoader(
            data=self.data, 
            subset=subset,
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=True,
            poisson_sampling=False,
        )

        return dataloader
    
    def to_device(self, data: Union[Data, Tensor]):
        """Move the data to the device."""
        return self.trainer.strategy.batch_to_device(data)
