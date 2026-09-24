import pytest
import torch

def test_grc():
    from log_depth_recurrent_modeling.ar_grc import GRC

    grc = GRC(256)

    x = torch.randn(2, 256)
    y = torch.randn(2, 256)

    assert grc(x, y).shape == x.shape
