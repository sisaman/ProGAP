from torch import Tensor
from typing import Annotated, Iterable, Literal, Optional
from core.args.utils import ArgInfo
from core.trainer.checkpoint import InMemoryCheckpoint
from core.trainer.progress import TrainerProgress
from core.modules.base import Metrics, TrainableModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint
from core import globals


class Trainer:
    def __init__(self,
                 monitor:       str = 'val/acc',
                 monitor_mode:  Literal['min', 'max'] = 'max',
                 accelerator:   Annotated[str, ArgInfo(help='accelerator to use')] = 'auto',
                 verbose:       Annotated[bool, ArgInfo(help='display progress')] = True,
                 log_trainer:   Annotated[bool, ArgInfo(help='log all training steps')] = False,
                 ):

        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.accelerator = accelerator
        self.verbose = verbose
        self.log_trainer = log_trainer
        self.reset()

    def reset(self):
        self.trainer = None
    
    def fit(self, 
            model: TrainableModule, 
            epochs: int,
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            ) -> Metrics:

        progress = TrainerProgress(
            # num_epochs=epochs, 
            # num_train_steps=len(train_dataloader), 
            # num_val_steps=len(val_dataloader), 
            # num_test_steps=len(test_dataloader),
            # disable=not self.verbose,
        )

        checkpoint = ModelCheckpoint(
            monitor=self.monitor,
            mode=self.monitor_mode,
            save_weights_only=True,
        )

        logger = globals['logger'] if self.log_trainer else False
        
        callbacks = [checkpoint]
        if self.verbose:
            callbacks.append(progress)

        self.trainer = LightningTrainer(
            accelerator=self.accelerator,
            callbacks=callbacks,
            logger=logger,
            max_epochs=epochs,
            check_val_every_n_epoch=1,
            # log_every_n_steps=1,
            enable_checkpointing=True,
            enable_progress_bar=self.verbose,
            enable_model_summary=False,
            plugins=InMemoryCheckpoint(),
            # deterministic=True,
        )

        val_test_dataloaders = []
        
        if val_dataloader:
            val_test_dataloaders.append(val_dataloader)
        if test_dataloader:
            val_test_dataloaders.append(test_dataloader)


        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_test_dataloaders,
        )

        metrics = {
            'epoch': self.trainer.strategy.load_checkpoint(checkpoint.best_model_path)['epoch'],
            self.monitor: checkpoint.best_model_score,
        }

        return metrics

    def test(self, dataloader: Iterable) -> Metrics:
        return self.trainer.test(dataloaders=dataloader, ckpt_path='best', verbose=False)[0]
    
    def predict(self, dataloader: Iterable) -> Tensor:
        return self.trainer.predict(dataloaders=dataloader, ckpt_path='best')[0]
