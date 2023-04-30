from torch import Tensor
from typing import Annotated, Iterable, Optional

import torch
from core.args.utils import ArgInfo
from core.trainer.checkpoint import InMemoryCheckpoint, CheckpointIO
from core.trainer.progress import ProgressCallback
from core.modules.base import Metrics, TrainableModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from core import globals


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
            callbacks.append(ProgressCallback())

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
            num_sanity_val_steps=0,
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

        if self.interrupted:
            raise KeyboardInterrupt

        checkpoint = self.checkpoint_callback
        metrics = {
            'epoch': self.strategy.load_checkpoint(checkpoint.best_model_path)['epoch'],
            checkpoint.monitor: checkpoint.best_model_score,
        }

        return metrics

    def test(self, dataloader: Iterable) -> Metrics:
        return super().test(dataloaders=dataloader, ckpt_path='best', verbose=False)[0]
    
    def predict(self, dataloader: Iterable) -> Tensor:
        # load best model
        ckpt_path = self.checkpoint_callback.best_model_path
        checkpoint = self.strategy.load_checkpoint(ckpt_path)
        self.strategy.load_model_state_dict(checkpoint)
        self.strategy.model_to_device()
        
        preds = []
        model = self.lightning_module
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self.strategy.batch_to_device(batch)
                # out might be a tuple of predictions
                out = model.predict_step(batch, batch_idx)
                preds.append(out)
            
        # concatenate predictions, check if they are tuples
        if isinstance(preds[0], tuple):
            preds = tuple(torch.cat([p[i] for p in preds]) for i in range(len(preds[0])))
        else:
            preds = torch.cat(preds)

        return preds
