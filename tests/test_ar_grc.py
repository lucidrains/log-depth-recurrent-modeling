import pytest
import torch

def test_grc():
    from log_depth_recurrent_modeling.ar_grc import GRC

    grc = GRC(256)

    x = torch.randn(2, 256)
    y = torch.randn(2, 256)

    assert grc(x, y).shape == x.shape

@pytest.mark.parametrize('seq_len', (32, 33))
def test_ar_grc(seq_len):
    from log_depth_recurrent_modeling.ar_grc import ARGRC

    model = ARGRC(
        num_tokens = 16,
        dim_embed = 32,
        dim = 32,
        window_size = 16
    )

    ids = torch.randint(0, 16, (2, seq_len))
    loss = model(ids, return_loss = True)

    assert loss.ndim == 0
