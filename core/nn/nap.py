import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from core.privacy.mechanisms.commons import GaussianMechanism


class NAP(Module):
    def __init__(self, noise_std: float, sensitivity: float) -> None:
        super().__init__()
        self.sensitivity = sensitivity
        noise_scale = noise_std / sensitivity
        self.gm = GaussianMechanism(noise_scale)

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = F.normalize(x, p=2, dim=-1)             # normalize
        x = torch.spmm(adj_t, x)                        # aggregate
        x = self.gm.perturb(x, self.sensitivity)    # perturb
        return x
