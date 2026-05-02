import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import pytest

from libs.util.tool import get_mask_from_lengths, init_random_seeds


def test_get_mask_from_lengths_basic():
    lengths = torch.tensor([3, 5, 2])
    mask = get_mask_from_lengths(lengths)
    assert mask.shape == (3, 5)
    # positions beyond each length should be True (masked)
    assert mask[0, 3].item() is True
    assert mask[0, 2].item() is False
    assert mask[2, 2].item() is True


def test_get_mask_from_lengths_explicit_max():
    lengths = torch.tensor([2, 4])
    mask = get_mask_from_lengths(lengths, max_len=6)
    assert mask.shape == (2, 6)
    assert mask[0, 2].item() is True
    assert mask[1, 3].item() is False


def test_init_random_seeds_runs():
    init_random_seeds(random_seed=42, rank=0)
    v1 = torch.rand(1).item()
    init_random_seeds(random_seed=42, rank=0)
    v2 = torch.rand(1).item()
    assert v1 == pytest.approx(v2)
