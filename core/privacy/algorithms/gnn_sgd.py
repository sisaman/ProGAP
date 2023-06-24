import numpy as np
import scipy
import dp_accounting
from opacus.optimizers import DPOptimizer
from torch.optim import Optimizer
from core.data.loader.node import NodeDataLoader
from core.privacy.algorithms.noisy_sgd import NoisySGD


def multiterm_dpsgd_privacy_accountant(num_training_steps,
                                       noise_multiplier,
                                       target_delta, num_samples,
                                       batch_size,
                                       max_terms_per_node):
    """Computes epsilon after a given number of training steps with DP-SGD/Adam.

    Accounts for the exact distribution of terms in a minibatch,
    assuming sampling of these without replacement.

    Returns np.inf if the noise multiplier is too small.

    Args:
        num_training_steps: Number of training steps.
        noise_multiplier: Noise multiplier that scales the sensitivity.
        target_delta: Privacy parameter delta to choose epsilon for.
        num_samples: Total number of samples in the dataset.
        batch_size: Size of every batch.
        max_terms_per_node: Maximum number of terms affected by the removal of a
            node.

    Returns:
        Privacy parameter epsilon.
    """
    if noise_multiplier < 1e-20:
        return np.inf

    # Compute distribution of terms.
    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    # Compute unamplified RDPs (that is, with sampling probability = 1).
    orders = np.arange(1, 10, 0.1)[1:]

    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
    unamplified_rdps = accountant._rdp  # pylint: disable=protected-access

    # Compute amplified RDPs for each (order, unamplified RDP) pair.
    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node))
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
            order - 1)
        amplified_rdps.append(amplified_rdp)

    # Verify lower bound.
    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(unamplified_rdps *
                  (batch_size / num_samples) ** 2 <= amplified_rdps + 1e-6):
        raise ValueError('The lower bound has been violated. Something is wrong.')

    # Account for multiple training steps.
    amplified_rdps_total = amplified_rdps * num_training_steps

    # Convert to epsilon-delta DP.
    return dp_accounting.rdp.compute_epsilon(orders, amplified_rdps_total,
                                             target_delta)[0]



class GNNBasedNoisySGD(NoisySGD):
    def __init__(self, noise_scale: float, dataset_size: int, batch_size: int, 
                 epochs: int, max_grad_norm: float, max_degree: int):
        super().__init__(
            noise_scale=noise_scale,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

        self.name = 'GNNNoisySGD'
        self.params['max_degree'] = max_degree

        if epochs > 0 and noise_scale > 0.0:
            
            def approxDP(delta):
                return multiterm_dpsgd_privacy_accountant(
                    num_training_steps=epochs * dataset_size // batch_size,
                    noise_multiplier=noise_scale,
                    target_delta=delta,
                    num_samples=dataset_size,
                    batch_size=batch_size,
                    max_terms_per_node=max_degree+1,
                )

            self.propagate_updates(approxDP, type_of_update='approxDP_func')

    def prepare_dataloader(self, dataloader: NodeDataLoader) -> NodeDataLoader:
        # since we don't need poisson sampling, we can use the same dataloader
        return dataloader

    def prepare_optimizer(self, optimizer: Optimizer) -> DPOptimizer:
        noise_scale = self.params['noise_scale']
        epochs = self.params['epochs']
        K = self.params['max_degree']
        C = self.params['max_grad_norm']
        DeltaK = 2 * (K + 1) * C

        if noise_scale > 0.0 and epochs > 0:
            # The noise std in Opacus is equal to noise_multiplier * max_grad_norm
            # But here, we want noise std to be equal to noise_scale * DeltaK
            # So we need to scale the noise_multiplier by DeltaK / C
            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_scale * DeltaK / C,
                max_grad_norm=C,
                expected_batch_size=self.params['batch_size'],
            )
        return optimizer
