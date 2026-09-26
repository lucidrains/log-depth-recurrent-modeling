from __future__ import annotations
from collections import namedtuple
import math

import torch
from torch import nn, Tensor, stack, cat
from torch.nn import Module, ModuleList, Linear, RMSNorm, LayerNorm
import torch.nn.functional as F

from einops import einsum, rearrange, repeat

from x_mlps_pytorch import GroupedMLP

from torch_einops_utils import pad_at_dim_to_multiple, pack_with_inverse, shift_right, temp_eval

# memory

TreeMemory = namedtuple('TreeMemory', ['step', 'subtrees'])
LayerMemory = namedtuple('LayerMemory', ['tree', 'prev_token'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def cast_tuple(t, length = 1):
    return tuple(t) if isinstance(t, (tuple, list)) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def normalize_memory(memory, num_layers):
    if isinstance(memory, (LayerMemory, TreeMemory)):
        return [memory]

    return default(memory, [None] * num_layers)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# classes

class MultiHeadGatedRecursiveCell(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        activation = nn.SiLU(),
        depth = 1,
        use_rmsnorm = True,
        heads = 8
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * 4)

        # grouped mlp, with the linear projection separated per head (h o i)

        dims = (2 * dim,) + (dim_hidden,) * (depth + 1) + (dim * 4,)
        self.mlp = GroupedMLP(*dims, groups = heads, activation = activation)

        self.norm = RMSNorm(dim) if use_rmsnorm else LayerNorm(dim, bias = False)

    def forward(self, x, y):
        # x, y are (batch, heads, seq, dim)

        gates = rearrange(cat((x, y), dim = -1), 'b h n d -> b n (h d)')
        gates = self.mlp(gates)

        x_gate, y_gate, c_gate, c = rearrange(gates, 'b n h (four d) -> four b h n d', four = 4)

        return self.norm(x * x_gate.sigmoid() + y * y_gate.sigmoid() + c_gate.sigmoid() * c)

# token shift, Bo Peng RWKV

def token_shift(x):
    x, inverse_pack = pack_with_inverse(x, '* n d')

    t, t_shift = x.chunk(2, dim = -1)
    t_shift = shift_right(t_shift, dim = 1)

    return inverse_pack(cat((t, t_shift), dim = -1))

# layer

class MultiHeadARGRCLayer(Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = None,
        max_seq_len: int | None = None,
        prenorm = False,
        shift_tokens = False,
        reverse_seq = False,
        cell: Module | tuple[Module, Module] | None = None,
        separate_grc = False,
        grc_kwargs: dict = dict()
    ):
        super().__init__()
        assert not exists(max_seq_len) or (isinstance(max_seq_len, int) and max_seq_len >= 2 and math.log2(max_seq_len).is_integer())
        assert divisible_by(dim, heads)

        dim_head = default(dim_head, dim // heads)
        assert dim == heads * dim_head

        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.max_seq_len = max_seq_len

        self.prenorm = prenorm
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.shift_tokens = shift_tokens
        self.reverse_seq = reverse_seq

        # project to the head dimension and back, as in attention

        self.to_split = Linear(dim, heads * dim_head)
        self.to_out = Linear(heads * dim_head, dim)

        self.root_hidden = nn.Parameter(torch.randn(heads, dim_head) * 1e-2)

        # cell

        up_grc, down_grc = cast_tuple(cell, 2)

        self.up_grc = default(up_grc, MultiHeadGatedRecursiveCell(dim = dim_head, heads = heads, **grc_kwargs))
        self.down_grc = default(down_grc, MultiHeadGatedRecursiveCell(dim = dim_head, heads = heads, **grc_kwargs) if separate_grc else self.up_grc)

    # forward step for single token

    def forward_step(
        self,
        embed: Tensor,
        memory: LayerMemory | TreeMemory | None = None
    ):
        assert not self.reverse_seq, 'forward_step is not supported when reverse_seq is set to True'

        b, _ = embed.shape
        max_seq_len = self.max_seq_len

        # unpack layer memory

        tree_mem, prev_token = (memory.tree, memory.prev_token) if isinstance(memory, LayerMemory) else (memory, None)

        # token shifting

        next_prev_token = None

        if self.shift_tokens:
            t, t_shift = embed.chunk(2, dim = -1)
            shifted = default(prev_token, torch.zeros_like(t_shift))
            embed = cat((t, shifted), dim = -1)
            next_prev_token = t_shift

        # prenorm

        if self.prenorm:
            embed = self.norm(embed)

        # project and split the heads, a single token is (b, heads, 1, dim_head)

        embed = self.to_split(embed)
        embed = rearrange(embed, 'b (h d) -> b h 1 d', h = self.heads)

        # tree memory carry and sweep

        step = tree_mem.step if exists(tree_mem) else 0
        subtrees = dict(tree_mem.subtrees) if exists(tree_mem) else dict()

        window_step = step % max_seq_len if exists(max_seq_len) else step

        # the merge tree resets at each window boundary

        if window_step == 0:
            subtrees = dict()

        # down-sweep carry: walk from root down to leaf at window_step
        # whenever branching right (bit is 1), absorb left sibling subtree

        carry = repeat(self.root_hidden, 'h d -> b h 1 d', b = b)

        for level in sorted(subtrees.keys(), reverse = True):
            if (window_step >> level) & 1:
                carry = self.down_grc(carry, subtrees[level][:, :, None, :])

        out = self.down_grc(carry, embed)

        # up-sweep: merge completed subtrees

        subtree = embed
        level = 0

        while level in subtrees:
            subtree = self.up_grc(subtrees.pop(level)[:, :, None, :], subtree)
            level += 1

        subtrees[level] = rearrange(subtree, 'b h 1 d -> b h d')

        # combine the heads

        out = rearrange(out, 'b h 1 d -> b (h d)')
        out = self.to_out(out)

        next_step = step + 1

        if exists(max_seq_len) and divisible_by(next_step, max_seq_len):
            subtrees = dict()

        next_memory = LayerMemory(
            tree = TreeMemory(step = next_step, subtrees = subtrees),
            prev_token = next_prev_token
        )

        return out, next_memory

    # forward

    def forward(
        self,
        x,
        memory: LayerMemory | TreeMemory | None = None,
        return_memory = False
    ):
        b, n, d = x.shape

        if self.reverse_seq:
            assert not exists(memory) and not return_memory, 'memory and decoding is not supported with reverse_seq'
            x = x.flip(dims = (1,))

        # sequential with tree memory if memory passed in

        if exists(memory):
            hiddens = []
            curr_memory = memory

            for embed in x.unbind(dim = 1):
                hidden, curr_memory = self.forward_step(embed, curr_memory)
                hiddens.append(hidden)

            # stack outputs

            out = stack(hiddens, dim = 1)
            return (out, curr_memory) if return_memory else out

        # parallel blelloch scan

        up_grc, down_grc, root_hidden = self.up_grc, self.down_grc, self.root_hidden
        heads = self.heads

        # token shifting

        next_prev_token = None

        if self.shift_tokens:
            next_prev_token = x[:, -1, (d // 2):]
            x = token_shift(x)

        # prenorm

        if self.prenorm:
            x = self.norm(x)

        # project and split the heads

        x = self.to_split(x)
        x = rearrange(x, 'b n (h d) -> b h n d', h = heads)

        # window size is `max_seq_len` if given, otherwise the next power of two
        # sequence is padded to a multiple of the window size, then the padding is stripped at the end

        window_size = default(self.max_seq_len, 2 ** max(1, math.ceil(math.log2(n))))
        tree_depth = int(math.log2(window_size))

        x, remove_padding = pad_at_dim_to_multiple(x, multiple = window_size, dim = -2)

        # fold sequence into windows

        x = rearrange(x, 'b h (w n) d -> (b w) h n d', n = window_size)

        # up sweep

        curr = x
        up_hiddens = [x]

        for _ in range(tree_depth - 1):
            left, right = rearrange(curr, 'bw h (l r) d -> r bw h l d', r = 2)
            curr = up_grc(left, right)
            up_hiddens.append(curr)

        # down sweep (blelloch)

        curr = repeat(root_hidden, 'h d -> bw h 1 d', bw = x.shape[0])

        for up_hidden in reversed(up_hiddens):
            left_up, _ = rearrange(up_hidden, 'bw h (l r) d -> r bw h l d', r = 2)
            right_carry = down_grc(curr, left_up)
            curr = rearrange([curr, right_carry], 'two bw h l d -> bw h (l two) d')

        # include each leaf

        x = down_grc(curr, x)

        # unfold windows and strip padding

        x = rearrange(x, '(b w) h n d -> b (w n) (h d)', b = b)
        x = remove_padding(x)

        # combine the heads

        x = self.to_out(x)

        # extract memory from tree intermediates if requested

        next_memory = None

        if return_memory:
            subtrees = dict()

            if not divisible_by(n, window_size):
                window_step = n % window_size
                last_window = (n - 1) // window_size

                up_unpacked = [rearrange(h, '(b w) h t d -> b w h t d', b = b) for h in up_hiddens]

                for level, up_hidden in enumerate(up_unpacked):
                    if not ((window_step >> level) & 1):
                        continue

                    node_idx = (window_step >> level) - 1
                    subtrees[level] = up_hidden[:, last_window, :, node_idx]

            elif not exists(self.max_seq_len):
                left, right = rearrange(up_hiddens[-1], 'b h (l r) d -> r b h l d', r = 2)
                subtrees[tree_depth] = up_grc(left, right)[:, :, 0]

            next_memory = LayerMemory(
                tree = TreeMemory(step = n, subtrees = subtrees),
                prev_token = next_prev_token
            )

        if self.reverse_seq:
            x = x.flip(dims = (1,))

        if not return_memory:
            return x

        return x, next_memory

# main model class

class MultiHeadARGRC(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim = None,
        dim_embed = None,
        depth = 1,
        max_seq_len: int | None = None,
        prenorm = False,
        residual = True,
        shift_tokens = False,
        reverse_seq = False,
        cell: Module | tuple[Module, Module] | None = None,
        separate_grc = False,
        heads = 8,
        dim_head = None,
        grc_kwargs: dict = dict()
    ):
        super().__init__()

        dim, dim_embed = default(dim, dim_embed), default(dim_embed, dim)
        assert exists(dim), 'dim must be specified'

        self.dim = dim
        self.dim_embed = dim_embed
        self.depth = depth
        self.residual = residual
        self.logit_scale = dim_embed ** -0.5

        # embed

        self.token_embed = nn.Embedding(num_tokens, dim_embed)

        # project to and from

        self.embed_to_model = Linear(dim_embed, dim)
        self.model_to_embed = Linear(dim, dim_embed)

        # layers

        reverse_seq = cast_tuple(reverse_seq, depth)

        self.layers = ModuleList([
            MultiHeadARGRCLayer(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                max_seq_len = max_seq_len,
                prenorm = prenorm,
                shift_tokens = shift_tokens,
                reverse_seq = layer_reverse_seq,
                cell = cell,
                separate_grc = separate_grc,
                grc_kwargs = grc_kwargs
            ) for layer_reverse_seq in reverse_seq
        ])

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

    # forward step for decoding with tree memory

    def forward_step(
        self,
        embed: Tensor,
        memory: list | tuple | None = None
    ):
        memory = normalize_memory(memory, len(self.layers))

        next_memories = []
        x = embed

        for layer, layer_memory in zip(self.layers, memory):
            out, next_layer_memory = layer.forward_step(x, layer_memory)
            next_memories.append(next_layer_memory)

            if self.residual:
                out = out + x

            x = out

        return x, next_memories

    # generate

    @torch.no_grad()
    @temp_eval
    def generate(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        prompt, inverse_pack = pack_with_inverse(prompt, '* n')
        prompt_seq_len = prompt.shape[-1]
        sample_num_times = max(0, seq_len - prompt_seq_len)

        logits, memories = self(prompt, return_memory = True)
        logits = logits[:, -1]

        generated = []

        for _ in range(sample_num_times):
            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1, keepdim = True)
            generated.append(sample)

            embeds = self.token_embed(sample[:, 0])
            embed = self.embed_to_model(embeds)

            hidden, memories = self.forward_step(embed, memories)

            hidden = self.norm(hidden)
            embeds = self.model_to_embed(hidden)
            logits = einsum(embeds, self.token_embed.weight, 'b d, l d -> b l') * self.logit_scale

        out = cat(generated, dim = -1) if len(generated) > 0 else prompt[:, :0]
        return inverse_pack(out, '* n')

    # forward

    def forward(
        self,
        ids,
        return_loss = False,
        labels = None,
        memory: list | tuple | None = None,
        return_memory = False
    ):
        ids, inverse_pack = pack_with_inverse(ids, '* n')

        if exists(labels):
            return_loss = True
            labels, _ = pack_with_inverse(labels, '* n')

        if return_loss and not exists(labels):
            ids, labels = ids[:, :-1], ids[:, 1:]

        embeds = self.token_embed(ids)
        x = self.embed_to_model(embeds)

        # handle memories for each layer

        memory = normalize_memory(memory, len(self.layers))

        next_memories = []

        for layer, layer_memory in zip(self.layers, memory):
            if return_memory or exists(layer_memory):
                out, next_layer_memory = layer(x, memory = layer_memory, return_memory = True)
                next_memories.append(next_layer_memory)
            else:
                out = layer(x)

            if self.residual:
                out = out + x

            x = out

        # to logits

        x = self.norm(x)
        embeds = self.model_to_embed(x)
        logits = einsum(embeds, self.token_embed.weight, 'b n d, l d -> b n l') * self.logit_scale

        if return_loss:
            out = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels, ignore_index = -1)
        else:
            out = inverse_pack(logits, '* n l')

        if not return_memory:
            return out

        return out, next_memories
