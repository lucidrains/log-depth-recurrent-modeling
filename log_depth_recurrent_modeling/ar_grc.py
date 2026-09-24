from __future__ import annotations

import torch
from torch import nn, Tensor, cat
from torch.nn import Module, ModuleList, RMSNorm, LayerNorm
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from x_mlps_pytorch import create_mlp

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# classes

class GatedRecursiveCell(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        activation = nn.SiLU(),
        depth = 1,
        use_rmsnorm = True
    ):
        super().__init__()
        dim_out = dim * 4
        dim_hidden = default(dim_hidden, dim_out)

        self.mlp = create_mlp(
            dim_in = 2 * dim,
            dim = dim_hidden,
            dim_out = dim_out,
            activation = activation,
            depth = depth
        )

        self.norm = RMSNorm(dim) if use_rmsnorm else LayerNorm(dim, bias = False)

    def forward(self, x, y):

        x_gate, y_gate, c_gate, c = self.mlp((x, y)).chunk(4, dim = -1)

        return self.norm(x * x_gate.sigmoid() + y * y_gate.sigmoid() + c_gate.sigmoid() * c)

# main class

class AutoregressiveGatedRecursiveCell(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x
    ):
        raise NotImplementedError

# alias

GRC = GatedRecursiveCell
ARGRC = AutoregressiveGatedRecursiveCell