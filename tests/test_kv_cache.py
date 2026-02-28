"""Tests for KVBlock in kv_cache.py."""

import pytest
import torch

from distr_inference.kv_cache import KVBlock, KVBlockConfig, BlockState


CONFIG = KVBlockConfig(
    num_layers=4,
    num_kv_heads=8,
    head_dim=64,
    block_size=4,
    dtype=torch.float32,
    device="cpu",
)


def make_block(block_id: int = 0) -> KVBlock:
    return KVBlock(block_id=block_id, config=CONFIG)


def random_kv():
    """Return a random (keys, values) pair for one token across all layers."""
    shape = (CONFIG.num_layers, CONFIG.num_kv_heads, CONFIG.head_dim)
    return torch.randn(shape), torch.randn(shape)


# ---------------------------------------------------------------------------
# append / get_filled_kv
# ---------------------------------------------------------------------------

def test_append_increments_num_filled():
    block = make_block()
    assert block.num_filled == 0

    for i in range(1, CONFIG.block_size + 1):
        k, v = random_kv()
        slot = block.append(k, v)
        assert slot == i - 1
        assert block.num_filled == i


def test_get_filled_kv_returns_written_tensors():
    block = make_block()
    k0, v0 = random_kv()
    k1, v1 = random_kv()
    block.append(k0, v0)
    block.append(k1, v1)

    filled_k, filled_v = block.get_filled_kv()

    assert filled_k.shape == (CONFIG.num_layers, 2, CONFIG.num_kv_heads, CONFIG.head_dim)
    assert filled_v.shape == (CONFIG.num_layers, 2, CONFIG.num_kv_heads, CONFIG.head_dim)

    # Each layer's slot 0 and 1 should match what was written
    assert torch.allclose(filled_k[:, 0], k0)
    assert torch.allclose(filled_v[:, 0], v0)
    assert torch.allclose(filled_k[:, 1], k1)
    assert torch.allclose(filled_v[:, 1], v1)


def test_append_raises_when_full():
    block = make_block()
    for _ in range(CONFIG.block_size):
        block.append(*random_kv())

    assert block.is_full
    with pytest.raises(RuntimeError):
        block.append(*random_kv())


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_filled_count_and_state():
    block = make_block()
    for _ in range(CONFIG.block_size):
        block.append(*random_kv())

    block.reset()

    assert block.num_filled == 0
    assert block.is_empty
    assert block.state == BlockState.FREE
    k, v = block.get_filled_kv()
    assert k.shape[1] == 0  # no filled slots
