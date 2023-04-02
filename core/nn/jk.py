import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import Linear, JumpingKnowledge as JK


class JumpingKnowledge(Module):
    supported_modes = ['cat', 'max', 'lstm', 'sum', 'mean', 'attn']
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == 'attn':
            self.hidden_dim = kwargs['hidden_dim']
            self.num_heads = kwargs['num_heads']
            self.fc = Linear(in_channels=-1, out_channels=self.num_heads, bias=False)
        elif mode == 'lstm':
            self.lstm = JK(mode='lstm', **kwargs)

    def forward(self, xs: Tensor) -> Tensor:
        """forward propagation

        Args:
            xs (Tensor): input with shape (batch_size, hidden_dim, num_phases)

        Returns:
            Tensor: aggregated output with shape (batch_size, hidden_dim)
        """
        if self.mode == 'cat':
            xs = xs.unbind(dim=-1)
            return torch.cat(xs, dim=-1)
        elif self.mode == 'sum':
            return xs.sum(dim=-1)
        elif self.mode == 'mean':
            return xs.mean(dim=-1)
        elif self.mode == 'max':
            return xs.max(dim=-1)[0]
        elif self.mode == 'attn':
            H = xs.transpose(1, 2)  # (node, hop, dim)
            W = self.fc(H).softmax(dim=1)  # (node, hop, head)
            out = H.transpose(1, 2).matmul(W).view(-1, self.hidden_dim * self.num_heads)
            return out
        else:
            xs = xs.unbind(dim=-1)
            self.lstm(xs)

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()