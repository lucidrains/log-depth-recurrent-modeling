import pytest
import torch

def test_grc():
    from log_depth_recurrent_modeling.ar_grc import GRC

    grc = GRC(256)

    x = torch.randn(2, 256)
    y = torch.randn(2, 256)

    assert grc(x, y).shape == x.shape

@pytest.mark.parametrize('max_seq_len', (None, 16))
@pytest.mark.parametrize('seq_len', (32, 33, 100))
def test_ar_grc(seq_len, max_seq_len):
    from log_depth_recurrent_modeling.ar_grc import ARGRC

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
