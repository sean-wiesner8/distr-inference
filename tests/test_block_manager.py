"""Tests for BlockManager in block_manager.py."""

import pytest
import torch

from distr_inference.block_manager import BlockManager
from distr_inference.kv_cache import KVBlockConfig, BlockState


CONFIG = KVBlockConfig(
    num_layers=4,
    num_kv_heads=8,
    head_dim=64,
    block_size=4,
    dtype=torch.float32,
    device="cpu",
)


def make_manager(num_blocks: int = 8) -> BlockManager:
    return BlockManager(num_blocks=num_blocks, config=CONFIG)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_all_blocks_free_on_init():
    manager = make_manager(num_blocks=8)
    assert manager.num_free_blocks == 8
    assert manager.num_used_blocks == 0


def test_contiguous_cache_shape():
    manager = make_manager(num_blocks=8)
    expected = (CONFIG.num_layers, 8, CONFIG.block_size, CONFIG.num_kv_heads, CONFIG.head_dim)
    assert manager.k_cache.shape == expected
    assert manager.v_cache.shape == expected


# ---------------------------------------------------------------------------
# register_sequence / free_sequence
# ---------------------------------------------------------------------------

def test_register_sequence():
    manager = make_manager()
    manager.register_sequence(seq_id=0)
    assert manager.num_blocks_for_sequence(seq_id=0) == 0


def test_register_duplicate_sequence_raises():
    manager = make_manager()
    manager.register_sequence(seq_id=0)
    with pytest.raises(ValueError):
        manager.register_sequence(seq_id=0)


def test_free_sequence_returns_blocks_to_pool():
    manager = make_manager(num_blocks=8)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)
    manager.allocate_block(seq_id=0)
    assert manager.num_free_blocks == 6

    manager.free_sequence(seq_id=0)
    assert manager.num_free_blocks == 8


def test_free_unregistered_sequence_raises():
    manager = make_manager()
    with pytest.raises(KeyError):
        manager.free_sequence(seq_id=99)


# ---------------------------------------------------------------------------
# allocate_block
# ---------------------------------------------------------------------------

def test_allocate_block_decrements_free_pool():
    manager = make_manager(num_blocks=8)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)
    assert manager.num_free_blocks == 7
    assert manager.num_used_blocks == 1


def test_allocate_block_returns_sequential_logical_indices():
    manager = make_manager()
    manager.register_sequence(seq_id=0)
    for expected_idx in range(4):
        logical_idx = manager.allocate_block(seq_id=0)
        assert logical_idx == expected_idx


def test_allocate_block_to_unregistered_sequence_raises():
    manager = make_manager()
    with pytest.raises(KeyError):
        manager.allocate_block(seq_id=99)


def test_allocate_block_raises_when_pool_exhausted():
    manager = make_manager(num_blocks=2)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)
    manager.allocate_block(seq_id=0)
    with pytest.raises(RuntimeError):
        manager.allocate_block(seq_id=0)


# ---------------------------------------------------------------------------
# get_block_table
# ---------------------------------------------------------------------------

def test_get_block_table_reflects_allocations():
    manager = make_manager()
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)
    manager.allocate_block(seq_id=0)
    table = manager.get_block_table(seq_id=0)
    assert len(table) == 2
    assert set(table.keys()) == {0, 1}


def test_get_block_table_is_a_copy():
    manager = make_manager()
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)
    table = manager.get_block_table(seq_id=0)
    table[99] = 99  # mutate the copy
    assert 99 not in manager.get_block_table(seq_id=0)


# ---------------------------------------------------------------------------
# can_allocate
# ---------------------------------------------------------------------------

def test_can_allocate():
    manager = make_manager(num_blocks=2)
    assert manager.can_allocate(1) is True
    assert manager.can_allocate(2) is True
    assert manager.can_allocate(3) is False


# ---------------------------------------------------------------------------
# write_slot / get_kv_cache
# ---------------------------------------------------------------------------

def test_write_slot_and_read_back():
    manager = make_manager(num_blocks=4)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)

    physical_id = manager.get_block_table(seq_id=0)[0]

    k = torch.randn(CONFIG.num_kv_heads, CONFIG.head_dim)
    v = torch.randn(CONFIG.num_kv_heads, CONFIG.head_dim)

    manager.write_slot(physical_id, slot_idx=0, layer_idx=0, k=k, v=v)

    k_layer, v_layer = manager.get_kv_cache(layer_idx=0)
    assert k_layer.shape == (4, CONFIG.block_size, CONFIG.num_kv_heads, CONFIG.head_dim)
    assert torch.allclose(k_layer[physical_id, 0], k)
    assert torch.allclose(v_layer[physical_id, 0], v)


def test_num_filled_tracking():
    manager = make_manager(num_blocks=4)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)

    physical_id = manager.get_block_table(seq_id=0)[0]
    assert manager.get_num_filled(physical_id) == 0

    manager.set_num_filled(physical_id, 3)
    assert manager.get_num_filled(physical_id) == 3


def test_free_sequence_resets_num_filled():
    manager = make_manager(num_blocks=4)
    manager.register_sequence(seq_id=0)
    manager.allocate_block(seq_id=0)

    physical_id = manager.get_block_table(seq_id=0)[0]
    manager.set_num_filled(physical_id, 2)

    manager.free_sequence(seq_id=0)
    assert manager.get_num_filled(physical_id) == 0
