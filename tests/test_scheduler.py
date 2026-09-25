"""Tests for the continuous-batching Scheduler (block accounting, admission, eviction)."""

import pytest
import torch

from distr_inference.block_manager import BlockManager
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.scheduler import Scheduler, SchedulerConfig
from distr_inference.sequence import SamplingParams, Sequence, SequenceStatus


CONFIG = KVBlockConfig(
    num_layers=2,
    num_kv_heads=4,
    head_dim=32,
    block_size=4,
    dtype=torch.float32,
    device="cpu",
)


def make_scheduler(num_blocks=8, max_num_seqs=8, max_num_batched_tokens=1024):
    bm = BlockManager(num_blocks=num_blocks, config=CONFIG)
    cfg = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return Scheduler(bm, cfg), bm


def seq(seq_id, prompt_len, **sp):
    return Sequence(seq_id, list(range(prompt_len)), SamplingParams(**sp))


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

def test_admits_waiting_and_registers_blocks():
    sched, bm = make_scheduler()
    sched.add_request(seq(0, prompt_len=3))

    batch = sched.schedule()

    assert [s.seq_id for s in batch] == [0]
    assert batch[0].status == SequenceStatus.RUNNING
    assert sched.running == batch
    assert not sched.waiting
    # Sequence must be registered so the forward pass can allocate its blocks.
    assert bm.num_blocks_for_sequence(0) == 0  # lazily grown during forward, not here


def test_admission_is_fifo():
    sched, _ = make_scheduler()
    for i in range(3):
        sched.add_request(seq(i, prompt_len=2))
    batch = sched.schedule()
    assert [s.seq_id for s in batch] == [0, 1, 2]


def test_max_num_seqs_cap():
    sched, _ = make_scheduler(max_num_seqs=2)
    for i in range(5):
        sched.add_request(seq(i, prompt_len=2))
    batch = sched.schedule()
    assert [s.seq_id for s in batch] == [0, 1]
    assert len(sched.waiting) == 3


def test_token_budget_limits_admission():
    # block_size=4 → each 4-token prompt needs 1 block. Budget of 8 admits 2.
    sched, _ = make_scheduler(max_num_batched_tokens=8)
    for i in range(4):
        sched.add_request(seq(i, prompt_len=4))
    batch = sched.schedule()
    assert [s.seq_id for s in batch] == [0, 1]
    assert len(sched.waiting) == 2


def test_kv_block_budget_limits_admission():
    # 2 blocks total, block_size=4, prompt of 4 needs 1 block each → only 2 fit.
    sched, _ = make_scheduler(num_blocks=2)
    for i in range(4):
        sched.add_request(seq(i, prompt_len=4))
    batch = sched.schedule()
    assert [s.seq_id for s in batch] == [0, 1]
    assert len(sched.waiting) == 2


def test_oversized_prompt_admitted_alone_over_budget():
    # A prompt larger than the token budget must still be admitted (no chunked
    # prefill), but only when it's first in line / batch is empty.
    sched, _ = make_scheduler(max_num_batched_tokens=4)
    sched.add_request(seq(0, prompt_len=10))
    batch = sched.schedule()
    assert [s.seq_id for s in batch] == [0]


def test_prompt_too_large_for_cache_raises():
    # 1 block of size 4 = 4 slots; a 5-token prompt can never fit.
    sched, _ = make_scheduler(num_blocks=1)
    sched.add_request(seq(0, prompt_len=5))
    with pytest.raises(RuntimeError, match="too long for the KV cache"):
        sched.schedule()


# ---------------------------------------------------------------------------
# Mixed prefill + decode
# ---------------------------------------------------------------------------

def test_running_and_waiting_batched_together():
    sched, bm = make_scheduler()
    # Admit seq 0, simulate a completed prefill step so it's mid-decode.
    sched.add_request(seq(0, prompt_len=4))
    sched.schedule()
    s0 = sched.running[0]
    s0.advance_cache(4)          # prefill wrote 4 tokens
    s0.append_token(99)          # decoded one token, now uncached
    bm.allocate_block(0)         # its prefill block (mirrors forward-time alloc)

    # New request arrives; next schedule must batch decode(seq0) + prefill(seq1).
    sched.add_request(seq(1, prompt_len=3))
    batch = sched.schedule()

    assert [s.seq_id for s in batch] == [0, 1]  # running first, then admitted
    assert batch[0].num_uncached_tokens == 1    # decode
    assert batch[1].num_uncached_tokens == 3    # prefill


def test_running_decode_out_of_blocks_raises():
    # No preemption: a running sequence that can't grow its cache must raise.
    sched, bm = make_scheduler(num_blocks=1)
    sched.add_request(seq(0, prompt_len=4))
    sched.schedule()
    s0 = sched.running[0]
    s0.advance_cache(4)
    bm.allocate_block(0)          # consumes the only block (now full at 4 slots)
    s0.append_token(50)           # crosses into a 2nd block on next step
    assert bm.num_free_blocks == 0
    with pytest.raises(RuntimeError, match="preemption is not implemented"):
        sched.schedule()


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_free_finished_evicts_and_releases_blocks():
    sched, bm = make_scheduler()
    sched.add_request(seq(0, prompt_len=4, max_tokens=1))
    sched.add_request(seq(1, prompt_len=4, max_tokens=10))
    sched.schedule()
    for s in sched.running:
        s.advance_cache(4)
        bm.allocate_block(s.seq_id)

    used_before = bm.num_used_blocks
    sched.running[0].append_token(7)   # seq 0 hits max_tokens=1 → finished

    finished = sched.free_finished()

    assert [s.seq_id for s in finished] == [0]
    assert [s.seq_id for s in sched.running] == [1]
    assert bm.num_used_blocks == used_before - 1   # seq 0's block released


def test_has_unfinished():
    sched, _ = make_scheduler()
    assert not sched.has_unfinished()
    sched.add_request(seq(0, prompt_len=2))
    assert sched.has_unfinished()
    sched.schedule()
    assert sched.has_unfinished()          # now running
    sched.running[0].mark_finished()
    sched.free_finished()
    assert not sched.has_unfinished()
