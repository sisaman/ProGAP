from torch import Tensor
from typing import Annotated, Iterable, Optional
from core.args.utils import ArgInfo
from core.trainer.checkpoint import InMemoryCheckpoint, CheckpointIO
from core.trainer.progress import TrainerProgress
from core.modules.base import Metrics, TrainableModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from core import globals, console


class Trainer(LightningTrainer):
    def __init__(self,
                 epochs:      Annotated[int,  ArgInfo(help='number of epochs for training')] = 100,
                 accelerator: Annotated[str,  ArgInfo(help='accelerator to use')] = 'auto',
                 verbose:     Annotated[bool, ArgInfo(help='display progress')] = True,
                 log_trainer: Annotated[bool, ArgInfo(help='log all training steps')] = False,
                 **kwargs:    Annotated[dict,  ArgInfo(help='extra options passed to the trainer class', bases=[LightningTrainer])]
                 ):
        
        callbacks = kwargs.pop('callbacks', [])
        plugins = kwargs.pop('plugins', [])
        logger = kwargs.pop('logger', globals['logger'] if log_trainer else False)

        if not any(isinstance(callback, ModelCheckpoint) for callback in callbacks):
            checkpoint = ModelCheckpoint(
                monitor='val/acc',
                mode='max',
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
            )
            callbacks.append(checkpoint)

        if verbose and not any(isinstance(callback, ProgressBar) for callback in callbacks):
            progress = TrainerProgress(
                # num_epochs=epochs, 
                # num_train_steps=len(train_dataloader), 
                # num_val_steps=len(val_dataloader), 
                # num_test_steps=len(test_dataloader),
                # disable=not self.verbose,
            )
            callbacks.append(progress)

        if not any(isinstance(plugin, CheckpointIO) for plugin in plugins):
            plugins.append(InMemoryCheckpoint())
        
        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            max_epochs=epochs,
            check_val_every_n_epoch=kwargs.pop('check_val_every_n_epoch', 1),
            enable_progress_bar=verbose,
            enable_model_summary=kwargs.pop('enable_model_summary', False),
            plugins=plugins,
            # deterministic=True,
            **kwargs
        )
    
    def fit(self, 
            model: TrainableModule, 
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            ) -> Metrics:

        val_test_dataloaders = []
        if val_dataloader:
            val_test_dataloaders.append(val_dataloader)
        if test_dataloader:
            val_test_dataloaders.append(test_dataloader)

        super().fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_test_dataloaders,
        )

        checkpoint = self.checkpoint_callback
        metrics = {
            'epoch': self.strategy.load_checkpoint(checkpoint.best_model_path)['epoch'],
            checkpoint.monitor: checkpoint.best_model_score,
        }

        return metrics

    def test(self, dataloader: Iterable) -> Metrics:
        return super().test(dataloaders=dataloader, ckpt_path='best', verbose=False)[0]
    
    def predict(self, dataloader: Iterable) -> Tensor:
        return super().predict(dataloaders=dataloader, ckpt_path='best')[0]
