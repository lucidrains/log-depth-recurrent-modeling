import pytest
import torch
from torch import cat

def test_grouped_grc():
    from log_depth_recurrent_modeling.multi_head_ar_grc import MultiHeadGatedRecursiveCell

    heads, dim = 4, 8
    grc = MultiHeadGatedRecursiveCell(dim = dim, heads = heads)

    x = torch.randn(2, heads, 5, dim)
    y = torch.randn(2, heads, 5, dim)

    assert grc(x, y).shape == x.shape

@pytest.mark.parametrize('max_seq_len', (None, 16))
@pytest.mark.parametrize('seq_len', (32, 33, 100))
def test_multi_head_ar_grc(seq_len, max_seq_len):
    from log_depth_recurrent_modeling.multi_head_ar_grc import MultiHeadARGRC

    model = MultiHeadARGRC(
        num_tokens = 16,
        dim_embed = 32,
        dim = 64,
        max_seq_len = max_seq_len,
        heads = 4,
        dim_head = 16
    )

    ids = torch.randint(0, 16, (2, seq_len))
    loss = model(ids, return_loss = True)
    loss.backward()

    assert loss.ndim == 0

    logits = model(ids)
    assert logits.shape == (2, seq_len, 16)

    sampled = model.generate(ids[:, :8], seq_len = 16)
    assert sampled.shape == (2, 8)

@pytest.mark.parametrize('max_seq_len', (None, 16))
@pytest.mark.parametrize('prompt_len', (8, 10))
def test_multi_head_ar_grc_sequential_vs_parallel(max_seq_len, prompt_len):
    from log_depth_recurrent_modeling.multi_head_ar_grc import MultiHeadARGRCLayer

    layer = MultiHeadARGRCLayer(
        dim = 64,
        heads = 4,
        dim_head = 16,
        max_seq_len = max_seq_len,
        prenorm = True,
        shift_tokens = True
    )

    seq_len = 23
    x = torch.randn(2, seq_len, 64)

    # parallel

    parallel_out = layer(x)

    # prompt parallel with memory, then sequential continuation

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
