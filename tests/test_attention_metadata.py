"""Tests for build_step_inputs / AttentionMetadata (CPU only)."""

import pytest
import torch

from distr_inference.attention_metadata import BLOCK_TABLE_PAD, build_step_inputs
from distr_inference.block_manager import BlockManager
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.sequence import SamplingParams, Sequence


BLOCK_SIZE = 4

CONFIG = KVBlockConfig(
    num_layers=2,
    num_kv_heads=2,
    head_dim=8,
    block_size=BLOCK_SIZE,
    dtype=torch.float32,
    device="cpu",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(num_blocks: int = 16) -> BlockManager:
    return BlockManager(num_blocks=num_blocks, config=CONFIG)


def make_seq(seq_id: int, prompt_len: int, num_output: int = 0, num_cached: int = 0) -> Sequence:
    """
    A sequence with ``prompt_len`` prompt tokens, ``num_output`` generated
    tokens, and ``num_cached`` of them already in the cache. Token values encode
    the sequence and position (``seq_id * 1000 + pos``) so packing is checkable.
    """
    seq = Sequence(
        seq_id,
        [seq_id * 1000 + p for p in range(prompt_len)],
        SamplingParams(max_tokens=1_000),
    )
    for p in range(prompt_len, prompt_len + num_output):
        seq.append_token(seq_id * 1000 + p)
    seq.num_cached_tokens = num_cached
    return seq


def allocate(bm: BlockManager, *seqs: Sequence) -> None:
    """Register and allocate blocks the way the scheduler does."""
    for seq in seqs:
        bm.register_sequence(seq.seq_id)
        bm.ensure_capacity(seq.seq_id, seq.seq_len)


def slots(block_ids, positions):
    """Expected flat slot for each absolute position, from physical block ids."""
    return [block_ids[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in positions]


# ---------------------------------------------------------------------------
# Slot mapping
# ---------------------------------------------------------------------------

def test_prefill_within_one_block():
    bm = make_manager()
    seq = make_seq(0, prompt_len=3)
    allocate(bm, seq)
    b0, = bm.get_block_ids(0)

    _, _, md = build_step_inputs([seq], bm, "cpu")
    assert md.slot_mapping.tolist() == [b0 * 4 + 0, b0 * 4 + 1, b0 * 4 + 2]


def test_prefill_spanning_blocks():
    bm = make_manager()
    seq = make_seq(0, prompt_len=9)
    allocate(bm, seq)
    b0, b1, b2 = bm.get_block_ids(0)

    _, _, md = build_step_inputs([seq], bm, "cpu")
    assert md.slot_mapping.tolist() == (
        [b0 * 4 + i for i in range(4)] + [b1 * 4 + i for i in range(4)] + [b2 * 4 + 0]
    )


def test_decode_mid_block():
    bm = make_manager()
    seq = make_seq(0, prompt_len=5, num_output=1, num_cached=5)
    allocate(bm, seq)
    _, b1 = bm.get_block_ids(0)

    _, _, md = build_step_inputs([seq], bm, "cpu")
    assert md.slot_mapping.tolist() == [b1 * 4 + 1]


def test_decode_crossing_block_boundary():
    bm = make_manager()
    seq = make_seq(0, prompt_len=4, num_output=1, num_cached=4)
    allocate(bm, seq)
    _, b1 = bm.get_block_ids(0)

    _, _, md = build_step_inputs([seq], bm, "cpu")
    assert md.slot_mapping.tolist() == [b1 * 4 + 0]


def test_mixed_batch_concatenates_in_batch_order():
    bm = make_manager()
    decode = make_seq(0, prompt_len=5, num_output=1, num_cached=5)
    prefill = make_seq(1, prompt_len=3)
    allocate(bm, decode, prefill)

    _, _, md = build_step_inputs([decode, prefill], bm, "cpu")

    expected = slots(bm.get_block_ids(0), [5]) + slots(bm.get_block_ids(1), [0, 1, 2])
    assert md.slot_mapping.tolist() == expected
    assert md.cu_seqlens_cpu == [0, 1, 4]
    # Each sequence's slice of slot_mapping is its own tokens.
    cu = md.cu_seqlens_cpu
    assert md.slot_mapping[cu[0]:cu[1]].tolist() == slots(bm.get_block_ids(0), [5])
    assert md.slot_mapping[cu[1]:cu[2]].tolist() == slots(bm.get_block_ids(1), [0, 1, 2])


def test_non_sequential_physical_blocks():
    """Interleaved allocation: logical != physical. Catches logical-index bugs."""
    bm = make_manager()
    bm.register_sequence(0)
    bm.register_sequence(1)
    for _ in range(2):
        bm.allocate_block(0)
        bm.allocate_block(1)
    assert bm.get_block_ids(0) == [0, 2]
    assert bm.get_block_ids(1) == [1, 3]

    s0 = make_seq(0, prompt_len=6)
    s1 = make_seq(1, prompt_len=7)
    _, _, md = build_step_inputs([s0, s1], bm, "cpu")

    assert md.slot_mapping.tolist() == [
        0, 1, 2, 3, 8, 9,         # seq 0: block 0, then block 2
        4, 5, 6, 7, 12, 13, 14,   # seq 1: block 1, then block 3
    ]
    assert md.block_table.tolist() == [[0, 2], [1, 3]]


# ---------------------------------------------------------------------------
# Block table
# ---------------------------------------------------------------------------

def test_block_table_shape_rows_and_padding():
    bm = make_manager()
    long_seq = make_seq(0, prompt_len=9)    # 3 blocks
    short_seq = make_seq(1, prompt_len=2)   # 1 block
    allocate(bm, long_seq, short_seq)

    _, _, md = build_step_inputs([long_seq, short_seq], bm, "cpu")

    assert md.block_table.shape == (2, 3)   # ceil(max_seqlen_k=9 / 4)
    assert md.block_table[0].tolist() == bm.get_block_ids(0)
    assert md.block_table[1].tolist() == bm.get_block_ids(1) + [BLOCK_TABLE_PAD] * 2


def test_block_table_width_follows_kv_length_not_new_tokens():
    """A decode's row covers its whole cached history, not just the new token."""
    bm = make_manager()
    seq = make_seq(0, prompt_len=8, num_output=1, num_cached=8)
    allocate(bm, seq)

    _, _, md = build_step_inputs([seq], bm, "cpu")
    assert md.block_table.tolist() == [bm.get_block_ids(0)]
    assert md.block_table.shape == (1, 3)


# ---------------------------------------------------------------------------
# Scalars and host copies
# ---------------------------------------------------------------------------

def test_lengths_and_scalars():
    bm = make_manager()
    decode = make_seq(3, prompt_len=5, num_output=2, num_cached=6)   # 1 new, kv 7
    prefill = make_seq(7, prompt_len=4)                              # 4 new, kv 4
    allocate(bm, decode, prefill)

    _, _, md = build_step_inputs([decode, prefill], bm, "cpu")

    assert md.seqused_k.tolist() == [7, 4]
    assert md.cu_seqlens_q.tolist() == [0, 1, 5]
    assert md.max_seqlen_q == 4
    assert md.max_seqlen_k == 7
    assert type(md.max_seqlen_q) is int and type(md.max_seqlen_k) is int

    assert md.seq_ids == [3, 7]
    assert md.num_cached == [6, 0]
    assert md.cu_seqlens_cpu == [0, 1, 5]


def test_dtypes_and_devices():
    bm = make_manager()
    seq = make_seq(0, prompt_len=5)
    allocate(bm, seq)

    input_ids, position_ids, md = build_step_inputs([seq], bm, "cpu")

    assert input_ids.dtype == torch.int64
    assert position_ids.dtype == torch.int32
    assert md.slot_mapping.dtype == torch.int64
    assert md.block_table.dtype == torch.int32
    assert md.cu_seqlens_q.dtype == torch.int32
    assert md.seqused_k.dtype == torch.int32

    cpu = torch.device("cpu")
    for t in (input_ids, position_ids, md.slot_mapping, md.block_table,
              md.cu_seqlens_q, md.seqused_k):
        assert t.device == cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensors_land_on_cuda():
    bm = make_manager()
    seq = make_seq(0, prompt_len=5)
    allocate(bm, seq)

    input_ids, position_ids, md = build_step_inputs([seq], bm, "cuda")

    for t in (input_ids, position_ids, md.slot_mapping, md.block_table,
              md.cu_seqlens_q, md.seqused_k):
        assert t.is_cuda
    assert md.slot_mapping.tolist() == slots(bm.get_block_ids(0), range(5))


# ---------------------------------------------------------------------------
# Packed token buffers
# ---------------------------------------------------------------------------

def test_input_and_position_ids():
    bm = make_manager()
    decode = make_seq(0, prompt_len=5, num_output=2, num_cached=6)
    prefill = make_seq(1, prompt_len=3)
    allocate(bm, decode, prefill)

    input_ids, position_ids, _ = build_step_inputs([decode, prefill], bm, "cpu")

    assert position_ids.tolist() == [6, 0, 1, 2]
    assert input_ids.tolist() == [6, 1000, 1001, 1002]


def test_uncached_suffix_spanning_prompt_and_output():
    """Watermark inside the prompt with outputs present: both lists contribute."""
    bm = make_manager()
    seq = make_seq(0, prompt_len=4, num_output=2, num_cached=2)
    allocate(bm, seq)

    input_ids, position_ids, md = build_step_inputs([seq], bm, "cpu")

    assert input_ids.tolist() == [2, 3, 4, 5]
    assert position_ids.tolist() == [2, 3, 4, 5]
    assert md.slot_mapping.tolist() == slots(bm.get_block_ids(0), [2, 3, 4, 5])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_missing_allocation_raises():
    """Allocation is the scheduler's job; the builder must not paper over it."""
    bm = make_manager()
    seq = make_seq(0, prompt_len=1)
    bm.register_sequence(0)

    with pytest.raises(RuntimeError):
        build_step_inputs([seq], bm, "cpu")
    assert bm.num_blocks_for_sequence(0) == 0


def test_partial_allocation_raises():
    bm = make_manager()
    seq = make_seq(0, prompt_len=5)   # needs 2 blocks
    bm.register_sequence(0)
    bm.allocate_block(0)

    with pytest.raises(RuntimeError):
        build_step_inputs([seq], bm, "cpu")


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        build_step_inputs([], make_manager(), "cpu")


def test_fully_cached_sequence_raises():
    bm = make_manager()
    seq = make_seq(0, prompt_len=3, num_cached=3)
    allocate(bm, seq)

    with pytest.raises(ValueError):
        build_step_inputs([seq], bm, "cpu")
