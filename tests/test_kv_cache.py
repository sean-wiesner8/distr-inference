"""Tests for KVBlockConfig in kv_cache.py."""

import torch

from distr_inference.kv_cache import KVBlockConfig, BlockState


def test_bytes_per_block():
    config = KVBlockConfig(
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    # 2 (k+v) * 4 layers * 4 slots * 8 heads * 64 dim * 4 bytes = 65536
    assert config.bytes_per_block == 2 * 4 * 4 * 8 * 64 * 4


def test_block_state_values():
    assert BlockState.FREE.name == "FREE"
    assert BlockState.ALLOCATED.name == "ALLOCATED"
