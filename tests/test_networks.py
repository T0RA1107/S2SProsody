import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import pytest

from libs.models.networks import get_norm_layer, GANLoss, Identity


@pytest.mark.parametrize("norm_type,dim", [
    ("batch", 1), ("batch", 2), ("batch", 3),
    ("instance", 1), ("instance", 2), ("instance", 3),
])
def test_get_norm_layer(norm_type, dim):
    norm_layer = get_norm_layer(norm_type=norm_type, dim=dim)
    assert norm_layer is not None


def test_get_norm_layer_none():
    norm_layer = get_norm_layer(norm_type="none")
    x = torch.rand(2, 4)
    out = norm_layer(x)
    assert isinstance(out, Identity)


def test_get_norm_layer_invalid():
    with pytest.raises(NotImplementedError):
        get_norm_layer(norm_type="unknown")


@pytest.mark.parametrize("target_is_real", [True, False])
def test_gan_loss(target_is_real):
    criterion = GANLoss(gan_mode="lsgan")
    pred = torch.rand(2, 1, 4, 4)
    loss = criterion(pred, target_is_real)
    assert loss.item() >= 0
