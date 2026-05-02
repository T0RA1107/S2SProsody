"""
Tests for libs/models/networks.py.

Imports networks.py directly (bypassing libs/models/__init__.py) to avoid
pulling in the full model stack (fastspeech2, unidecode, etc.) in CI.
"""
import sys
import os
import importlib.util

# Load networks.py directly without triggering libs/models/__init__.py
_spec = importlib.util.spec_from_file_location(
    "networks",
    os.path.join(os.path.dirname(__file__), "../src/libs/models/networks.py"),
)
networks = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(networks)  # type: ignore[union-attr]

import torch
import pytest


@pytest.mark.parametrize("norm_type,dim", [
    ("batch", 1), ("batch", 2), ("batch", 3),
    ("instance", 1), ("instance", 2), ("instance", 3),
])
def test_get_norm_layer(norm_type, dim):
    norm_layer = networks.get_norm_layer(norm_type=norm_type, dim=dim)
    assert norm_layer is not None


def test_get_norm_layer_none():
    norm_layer = networks.get_norm_layer(norm_type="none")
    x = torch.rand(2, 4)
    out = norm_layer(x)
    assert isinstance(out, networks.Identity)


def test_get_norm_layer_invalid():
    with pytest.raises(NotImplementedError):
        networks.get_norm_layer(norm_type="unknown")


@pytest.mark.parametrize("target_is_real", [True, False])
def test_gan_loss(target_is_real):
    criterion = networks.GANLoss(gan_mode="lsgan")
    pred = torch.rand(2, 1, 4, 4)
    loss = criterion(pred, target_is_real)
    assert loss.item() >= 0
