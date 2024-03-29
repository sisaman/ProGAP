from opacus.privacy_engine import forbid_accumulation_hook
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from autodp.transformer_zoo import Composition, AmplificationBySampling
from torch.nn import Module
from torch.optim import Optimizer
from core.data.loader.node import NodeDataLoader
from core.modules.base import TrainableModule
from core.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism


class NoisySGD(NoisyMechanism):
    def __init__(self, noise_scale: float, dataset_size: int, batch_size: int, epochs: int, max_grad_norm: float):
        super().__init__(noise_scale)
        self.name = 'NoisySGD'
        self.params = {
            'noise_scale': noise_scale, 
            'dataset_size': dataset_size, 
            'batch_size': batch_size, 
            'epochs': epochs,
            'max_grad_norm': max_grad_norm,
        }

        if epochs == 0:
            mech = ZeroMechanism()
            self.params['noise_scale'] = 0.0
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            subsample = AmplificationBySampling()
            compose = Composition()
            gm = GaussianMechanism(noise_scale=noise_scale)
            subsampled_gm = subsample(gm, prob=batch_size/dataset_size, improved_bound_flag=True)
            mech = compose([subsampled_gm],[epochs * dataset_size // batch_size])
        
        self.set_all_representation(mech)

    def prepare_module(self, module: Module) -> None:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            if hasattr(module, 'autograd_grad_sample_hooks'):
                for hook in module.autograd_grad_sample_hooks:
                    hook.remove()
                del module.autograd_grad_sample_hooks
            GradSampleModule(module).register_full_backward_hook(forbid_accumulation_hook)

    def prepare_dataloader(self, dataloader: NodeDataLoader) -> NodeDataLoader:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            dataloader.poisson_sampling = True
        return dataloader

    def prepare_optimizer(self, optimizer: Optimizer) -> DPOptimizer:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=self.params['noise_scale'],    # noise_multiplier is the same as noise_scale in Opacus
                max_grad_norm=self.params['max_grad_norm'],
                expected_batch_size=self.params['batch_size'],
            )
        return optimizer
    
    def prepare_trainable_module(self, module: TrainableModule) -> None:
        self.prepare_module(module)
        if not hasattr(module, 'original_configure_optimizers'):
            module.original_configure_optimizers = module.configure_optimizers
            module.configure_optimizers = lambda: self.prepare_optimizer(module.original_configure_optimizers())
