import pytest
import torch
from torch import cat

def test_grc():
    from log_depth_recurrent_modeling import GRC

    grc = GRC(256)

    x = torch.randn(2, 256)
    y = torch.randn(2, 256)

    assert grc(x, y).shape == x.shape

@pytest.mark.parametrize('max_seq_len', (None, 16))
@pytest.mark.parametrize('seq_len', (32, 33, 100))
def test_ar_grc(seq_len, max_seq_len):
    from log_depth_recurrent_modeling import ARGRC

    model = ARGRC(
        num_tokens = 16,
        dim_embed = 32,
        dim = 32,
        max_seq_len = max_seq_len
    )

    ids = torch.randint(0, 16, (2, seq_len))
    loss = model(ids, return_loss = True)

    assert loss.ndim == 0

    logits = model(ids)
    assert logits.shape == (2, seq_len, 16)

    sampled = model.generate(ids[:, :8], seq_len = 16)
    assert sampled.shape == (2, 8)

@pytest.mark.parametrize('prenorm', (False, True))
@pytest.mark.parametrize('shift_tokens', (False, True))
@pytest.mark.parametrize('max_seq_len', (16, None))
def test_argrc_layer_sequential_vs_parallel(prenorm, shift_tokens, max_seq_len):
    from log_depth_recurrent_modeling import ARGRCLayer

    layer = ARGRCLayer(
        dim = 32,
        max_seq_len = max_seq_len,
        prenorm = prenorm,
        shift_tokens = shift_tokens
    )

    seq_len = 23
    x = torch.randn(2, seq_len, 32)

    # parallel

    parallel_out = layer(x)

    # prompt parallel with memory, then sequential continuation

    prompt_len = 10
    prompt_out, memory = layer(x[:, :prompt_len], return_memory = True)

    seq_out, _ = layer(x[:, prompt_len:], memory = memory, return_memory = True)

    combo_out = cat((prompt_out, seq_out), dim = 1)
    assert torch.allclose(parallel_out, combo_out, atol = 1e-4)

    # token by token sequential with layer memory

    token_outs = []
    curr_memory = None

    for t in range(seq_len):
        step_out, curr_memory = layer(x[:, t:t+1], memory = curr_memory, return_memory = True)
        token_outs.append(step_out)

    token_by_token_out = cat(token_outs, dim = 1)
    assert torch.allclose(parallel_out, token_by_token_out, atol = 1e-4)

@pytest.mark.parametrize('depth', (1, 2))
@pytest.mark.parametrize('prenorm', (False, True))
@pytest.mark.parametrize('shift_tokens', (False, True))
@pytest.mark.parametrize('residual', (False, True))
def test_argrc_depth_sequential_vs_parallel(depth, prenorm, shift_tokens, residual):
    from log_depth_recurrent_modeling import ARGRC

    model = ARGRC(
        num_tokens = 16,
        dim_embed = 32,
        dim = 32,
        depth = depth,
        max_seq_len = 16,
        prenorm = prenorm,
        residual = residual,
        shift_tokens = shift_tokens
    )

    seq_len = 23
    ids = torch.randint(0, 16, (2, seq_len))

    # parallel

    parallel_logits = model(ids)

    # prompt parallel with memory, then sequential continuation

    prompt_len = 11
    prompt_logits, memory = model(ids[:, :prompt_len], return_memory = True)

    seq_logits, _ = model(ids[:, prompt_len:], memory = memory, return_memory = True)

    combo_logits = cat((prompt_logits, seq_logits), dim = 1)
    assert torch.allclose(parallel_logits, combo_logits, atol = 1e-4)

    # token by token sequential with tree memory

    token_logits = []
    curr_memory = None

    for t in range(seq_len):
        step_logits, curr_memory = model(ids[:, t:t+1], memory = curr_memory, return_memory = True)
        token_logits.append(step_logits)

    token_by_token_logits = cat(token_logits, dim = 1)
    assert torch.allclose(parallel_logits, token_by_token_logits, atol = 1e-4)

    # generate

    sampled = model.generate(ids[:, :8], seq_len = 16)
    assert sampled.shape == (2, 8)

def test_argrc_reverse_seq():
    from log_depth_recurrent_modeling import ARGRCLayer, ARGRC

    layer = ARGRCLayer(
        dim = 32,
        reverse_seq = True,
        prenorm = True,
        shift_tokens = True
    )

    x = torch.randn(2, 17, 32)
    out = layer(x)
    assert out.shape == x.shape

    model = ARGRC(
        num_tokens = 16,
        dim_embed = 32,
        dim = 32,
        depth = 2,
        reverse_seq = (False, True),
        prenorm = True,
        shift_tokens = True
    )

    ids = torch.randint(0, 16, (2, 17))
    logits = model(ids)
    assert logits.shape == (2, 17, 16)
    loss = model(ids, return_loss = True)
    assert loss.ndim == 0

def test_custom_cell_alternative():
    from torch.nn import Module, Linear
    from log_depth_recurrent_modeling import ARGRCLayer

    class SimpleLinearCell(Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = Linear(dim * 2, dim)

        def forward(self, x, y):
            return self.proj(cat((x, y), dim = -1))

    dim = 32
    cell = SimpleLinearCell(dim)

    layer = ARGRCLayer(
        dim = dim,
        cell = cell,
        prenorm = True,
        shift_tokens = True
    )

    x = torch.randn(2, 17, dim)

    # parallel

    parallel_out = layer(x)

    # sequential with memory

    prompt_out, memory = layer(x[:, :9], return_memory = True)
    seq_out, _ = layer(x[:, 9:], memory = memory, return_memory = True)
    combo_out = cat((prompt_out, seq_out), dim = 1)

    assert torch.allclose(parallel_out, combo_out, atol = 1e-4)
