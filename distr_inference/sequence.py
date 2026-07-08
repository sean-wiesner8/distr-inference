"""
Request/Sequence abstraction for the inference engine.

A :class:`Sequence` is the engine's unit of work: it owns the per-request
state that must survive *across* forward passes — the prompt, the tokens
generated so far, the sampling parameters, and how much of the sequence is
already resident in the paged KV cache. The model and attention layers operate
on anonymous packed buffers (``input_ids``, ``cu_seqlens``, ``seq_ids``,
``seq_lens``); ``Sequence`` is where that per-request state lives between the
scheduler's iterations.

Lifecycle
---------
A sequence moves through three states:

    WAITING  → sitting in a :class:`RequestQueue`, prompt not yet prefilled.
    RUNNING  → prefilled and in the scheduler's running list, decoding.
    FINISHED → hit ``max_tokens`` or a stop token; blocks can be freed.

The scheduler pulls from the :class:`RequestQueue` to prefill, moves sequences
into its running list to decode, and drops them once ``done`` is set.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Deque, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingParams:
    """
    Per-request sampling / stopping configuration.

    Parameters
    ----------
    temperature   : Softmax temperature. ``0.0`` means greedy (argmax).
    top_k         : Restrict sampling to the ``k`` highest-probability tokens.
                    ``-1`` disables top-k filtering.
    max_tokens    : Maximum number of tokens to generate before finishing.
    stop_token_ids: Token IDs that finish the sequence when produced
                    (e.g. the EOS token).
    """
    temperature: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop_token_ids: Tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Sequence status
# ---------------------------------------------------------------------------

class SequenceStatus(Enum):
    WAITING  = auto()   # queued, prompt not yet prefilled
    RUNNING  = auto()   # prefilled, decoding
    FINISHED = auto()   # hit max_tokens or a stop token


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

class Sequence:
    """
    A single generation request and its evolving decode state.

    Parameters
    ----------
    seq_id           : Unique integer id. Keys this sequence's block table in
                       the :class:`~distr_inference.block_manager.BlockManager`
                       and identifies it in the model's packed ``seq_ids``.
    prompt_token_ids : The prompt, tokenized. Immutable after construction.
    sampling_params  : Sampling / stopping configuration.
    status           : Initial lifecycle state (defaults to ``WAITING``).
    """

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        status: SequenceStatus = SequenceStatus.WAITING,
    ) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.sampling_params = sampling_params
        self.status = status

        # How many of this sequence's tokens are already written to the paged
        # KV cache. Feeds the model's ``seq_lens`` ("tokens cached before this
        # step"). 0 until prefill runs; advanced by the scheduler after each
        # forward pass.
        self.num_cached_tokens: int = 0

    # ------------------------------------------------------------------
    # Derived length / token views
    # ------------------------------------------------------------------

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def seq_len(self) -> int:
        """Current total length: prompt tokens + generated tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> List[int]:
        """Prompt followed by all generated tokens."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        """
        The most recent token — the model's input for the next decode step.

        Falls back to the final prompt token before any tokens are generated.
        """
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def num_uncached_tokens(self) -> int:
        """Tokens not yet written to the KV cache (what the next step must process)."""
        return self.seq_len - self.num_cached_tokens

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """True once the sequence has finished generating."""
        return self.status == SequenceStatus.FINISHED

    def mark_running(self) -> None:
        self.status = SequenceStatus.RUNNING

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        """
        Append a generated token and update the finished state.

        The sequence is marked ``FINISHED`` if the new token is a stop token
        or if it brings the output up to ``max_tokens``.
        """
        self.output_token_ids.append(token_id)

        sp = self.sampling_params
        if token_id in sp.stop_token_ids:
            self.mark_finished()
        elif self.num_output_tokens >= sp.max_tokens:
            self.mark_finished()

    def advance_cache(self, num_tokens: int) -> None:
        """
        Advance the KV-cache watermark by ``num_tokens`` after a forward pass.

        Raises
        ------
        ValueError if the watermark would exceed the current sequence length.
        """
        new_cached = self.num_cached_tokens + num_tokens
        if new_cached > self.seq_len:
            raise ValueError(
                f"Cache watermark ({new_cached}) exceeds sequence length "
                f"({self.seq_len}) for sequence {self.seq_id}."
            )
        self.num_cached_tokens = new_cached

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, status={self.status.name}, "
            f"prompt={self.num_prompt_tokens}, output={self.num_output_tokens}, "
            f"cached={self.num_cached_tokens})"
        )


# ---------------------------------------------------------------------------
# Sequence id allocation
# ---------------------------------------------------------------------------

class SequenceIdAllocator:
    """
    Monotonic source of unique sequence ids.

    Kept separate from ``Sequence.__init__`` so tests can build sequences with
    fixed ids while the engine draws fresh ones at request-submission time.
    """

    def __init__(self, start: int = 0) -> None:
        self._counter = count(start)

    def next_id(self) -> int:
        return next(self._counter)


# ---------------------------------------------------------------------------
# Request queue
# ---------------------------------------------------------------------------

class RequestQueue:
    """
    FIFO queue of newly-submitted sequences waiting to be prefilled.

    The scheduler drains this each iteration to admit new work; sequences it
    admits move to the scheduler's (plain list) running set for decode.
    """

    def __init__(self) -> None:
        self._queue: Deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """Enqueue a waiting sequence at the back."""
        self._queue.append(seq)

    def pop(self) -> Sequence:
        """
        Remove and return the sequence at the front (oldest).

        Raises
        ------
        IndexError if the queue is empty.
        """
        return self._queue.popleft()

    def peek(self) -> Optional[Sequence]:
        """Return the front sequence without removing it, or None if empty."""
        return self._queue[0] if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self._queue)

    def __repr__(self) -> str:
        return f"RequestQueue(waiting={len(self._queue)})"
