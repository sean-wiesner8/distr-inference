"""Tests for the Sequence/Request abstraction in sequence.py."""

import pytest

from distr_inference.sequence import (
    SamplingParams,
    Sequence,
    SequenceStatus,
    SequenceIdAllocator,
    RequestQueue,
)


def make_seq(seq_id: int = 0, prompt=(1, 2, 3), **sp_kwargs) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(**sp_kwargs),
    )


# ---------------------------------------------------------------------------
# Construction & derived views
# ---------------------------------------------------------------------------

def test_initial_state():
    seq = make_seq(prompt=(1, 2, 3))
    assert seq.status == SequenceStatus.WAITING
    assert not seq.done
    assert seq.num_prompt_tokens == 3
    assert seq.num_output_tokens == 0
    assert seq.seq_len == 3
    assert seq.num_cached_tokens == 0
    assert seq.num_uncached_tokens == 3


def test_prompt_is_copied_not_aliased():
    prompt = [1, 2, 3]
    seq = Sequence(0, prompt, SamplingParams())
    prompt.append(4)
    assert seq.prompt_token_ids == [1, 2, 3]


def test_last_token_falls_back_to_prompt():
    seq = make_seq(prompt=(1, 2, 9))
    assert seq.last_token_id == 9


def test_all_token_ids_and_last_token_after_append():
    seq = make_seq(prompt=(1, 2, 3), max_tokens=10)
    seq.append_token(7)
    seq.append_token(8)
    assert seq.all_token_ids == [1, 2, 3, 7, 8]
    assert seq.last_token_id == 8
    assert seq.seq_len == 5


# ---------------------------------------------------------------------------
# Finishing conditions
# ---------------------------------------------------------------------------

def test_finishes_at_max_tokens():
    seq = make_seq(max_tokens=2)
    seq.append_token(100)
    assert not seq.done
    seq.append_token(101)
    assert seq.done
    assert seq.status == SequenceStatus.FINISHED


def test_finishes_on_stop_token():
    seq = make_seq(max_tokens=100, stop_token_ids=(42,))
    seq.append_token(10)
    assert not seq.done
    seq.append_token(42)
    assert seq.done


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

def test_mark_running_and_finished():
    seq = make_seq()
    seq.mark_running()
    assert seq.status == SequenceStatus.RUNNING
    seq.mark_finished()
    assert seq.done


# ---------------------------------------------------------------------------
# KV-cache watermark
# ---------------------------------------------------------------------------

def test_advance_cache_tracks_watermark():
    seq = make_seq(prompt=(1, 2, 3), max_tokens=10)
    seq.advance_cache(3)  # prefill wrote the whole prompt
    assert seq.num_cached_tokens == 3
    assert seq.num_uncached_tokens == 0

    seq.append_token(7)   # decoded a token, not yet cached
    assert seq.num_uncached_tokens == 1
    seq.advance_cache(1)
    assert seq.num_cached_tokens == 4
    assert seq.num_uncached_tokens == 0


def test_advance_cache_past_seq_len_raises():
    seq = make_seq(prompt=(1, 2, 3))
    with pytest.raises(ValueError):
        seq.advance_cache(4)


# ---------------------------------------------------------------------------
# SequenceIdAllocator
# ---------------------------------------------------------------------------

def test_id_allocator_is_monotonic():
    alloc = SequenceIdAllocator()
    assert [alloc.next_id() for _ in range(3)] == [0, 1, 2]

    alloc2 = SequenceIdAllocator(start=100)
    assert alloc2.next_id() == 100
    assert alloc2.next_id() == 101


# ---------------------------------------------------------------------------
# RequestQueue
# ---------------------------------------------------------------------------

def test_request_queue_fifo_order():
    q = RequestQueue()
    assert not q
    assert len(q) == 0
    assert q.peek() is None

    a, b, c = make_seq(0), make_seq(1), make_seq(2)
    q.add(a)
    q.add(b)
    q.add(c)

    assert len(q) == 3
    assert bool(q)
    assert q.peek() is a          # peek does not remove
    assert len(q) == 3

    assert q.pop() is a
    assert q.pop() is b
    assert q.pop() is c
    assert len(q) == 0


def test_request_queue_pop_empty_raises():
    q = RequestQueue()
    with pytest.raises(IndexError):
        q.pop()


def test_request_queue_is_iterable_front_to_back():
    q = RequestQueue()
    seqs = [make_seq(i) for i in range(3)]
    for s in seqs:
        q.add(s)
    assert list(q) == seqs
