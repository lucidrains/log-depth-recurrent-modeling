from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import Module, Linear, RMSNorm, LayerNorm
import torch.nn.functional as F

from einops import einsum, rearrange, repeat

from x_mlps_pytorch import create_mlp

from torch_einops_utils import pad_at_dim_to_multiple, pack_with_inverse

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
        *,
        num_tokens,
        dim_embed,
        dim,
        window_size,
        separate_grc = False,
        grc_kwargs: dict = dict()
    ):
        super().__init__()

        tree_depth = math.log2(window_size)
        assert tree_depth.is_integer()

        self.dim = dim

        # embed

        self.token_embed = nn.Embedding(num_tokens, dim_embed)

        # project to and from

        self.embed_to_model = Linear(dim_embed, dim)
        self.model_to_embed = Linear(dim, dim_embed)

        # recursive tree, up and down (blelloch scan)

        self.window_size = window_size

        self.tree_depth = int(tree_depth)

        self.root_hidden = nn.Parameter(torch.randn(dim) * 1e-2)

        self.separate_grc = separate_grc
        self.up_grc = GatedRecursiveCell(dim = dim, **grc_kwargs)
        self.down_grc = GatedRecursiveCell(dim = dim, **grc_kwargs) if separate_grc else self.up_grc

    def forward(
        self,
        ids,
        return_loss = False
    ):
        up_grc, down_grc, root_hidden = self.up_grc, self.down_grc, self.root_hidden

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        embeds = self.token_embed(ids)

        x = self.embed_to_model(embeds)

        x, remove_padding = pad_at_dim_to_multiple(x, multiple = self.window_size, dim = -2)

        # divide into window size

        x = rearrange(x, 'b (w n) d -> b w n d', n = self.window_size)
        x, inverse_pack_window = pack_with_inverse(x, '* n d')

        # up sweep

        curr = x
        up_hiddens = [x]

        for _ in range(self.tree_depth - 1):

            left, right = rearrange(curr, 'b (h two) d -> two b h d', two = 2)
            curr = up_grc(left, right)

            up_hiddens.append(curr)

        # down sweep (blelloch)

        curr = repeat(root_hidden, 'd -> b 1 d', b = x.shape[0])

        for _ in range(self.tree_depth):
            left_up, _ = rearrange(up_hiddens.pop(), 'b (h two) d -> two b h d', two = 2)

            right_carry = down_grc(curr, left_up)

            curr = rearrange([curr, right_carry], 'two b h d -> b (h two) d')

        # include each leaf

        x = down_grc(curr, x)

        # restore window

        x = inverse_pack_window(x)
        x = rearrange(x, 'b w n d -> b (w n) d')

        # slice out padding

        x = remove_padding(x)

        embeds = self.model_to_embed(x)

        logits = einsum(embeds, self.token_embed.weight, 'b n d, l d -> b n l')

        if not return_loss:
            return logits

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels, ignore_index = -1)

# alias

GRC = GatedRecursiveCell
ARGRC = AutoregressiveGatedRecursiveCell
