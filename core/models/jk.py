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

    def forward(self, xs: list[Tensor]) -> Tensor:
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'sum':
            return torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            return torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'attn':
            H = torch.stack(xs, dim=1)  # (node, hop, dim)
            W = self.fc(H).softmax(dim=1)  # (node, hop, head)
            out = H.transpose(1, 2).matmul(W).view(-1, self.hidden_dim * self.num_heads)
            return out
        else:
            self.lstm(xs)

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()