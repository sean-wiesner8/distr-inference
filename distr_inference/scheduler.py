"""
Continuous-batching scheduler (iteration-level scheduling).

Each engine step, :meth:`Scheduler.schedule` builds one mixed prefill/decode
batch: every running sequence decodes, and waiting sequences are admitted
FIFO while they fit a per-step token budget, a max-sequences cap, and the KV
block budget. After the forward pass and sampling, :meth:`free_finished`
evicts finished sequences and releases their KV blocks back to the pool.

Block allocation itself is *not* reimplemented here — the paged-attention
forward lazily calls :meth:`BlockManager.allocate_block` as tokens cross block
boundaries. The scheduler only *reserves against* the free-block count so it
never admits work the cache can't hold.

Out of scope (clean seams left, not implemented)
------------------------------------------------
* Chunked prefill  — an oversized prompt is admitted whole (see the
  "admit at least one" rule below) rather than split across steps.
* Preemption / KV eviction under pressure — if a *running* sequence can't get
  a block on decode, we raise. There is nothing to evict yet.
* Prefix / KV-block sharing, CPU<->GPU swapping, speculative decoding,
  priority scheduling — admission is strict FIFO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .block_manager import BlockManager
from .sequence import RequestQueue, Sequence


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Per-step scheduling limits.

    Parameters
    ----------
    max_num_seqs           : Cap on sequences in a single batch (running +
                             newly admitted).
    max_num_batched_tokens : Soft token budget per step. A prompt larger than
                             this is still admitted alone to guarantee forward
                             progress (chunked prefill is out of scope).
    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192


class Scheduler:
    """
    FIFO continuous-batching scheduler over a shared KV block pool.

    Parameters
    ----------
    block_manager : The shared paged KV cache allocator.
    config        : Per-step scheduling limits.
    """

    def __init__(self, block_manager: BlockManager, config: SchedulerConfig) -> None:
        self.block_manager = block_manager
        self.config = config

        # Waiting = not yet prefilled (FIFO). Running = decoding (plain list).
        self.waiting = RequestQueue()
        self.running: List[Sequence] = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def add_request(self, seq: Sequence) -> None:
        """Enqueue a freshly-submitted sequence for admission."""
        self.waiting.add(seq)

    def has_unfinished(self) -> bool:
        """True while any sequence is waiting or running."""
        return bool(self.waiting) or bool(self.running)

    # ------------------------------------------------------------------
    # Block accounting
    # ------------------------------------------------------------------

    def _blocks_to_add(self, seq: Sequence, already_allocated: int) -> int:
        """
        Physical blocks this step must allocate to hold ``seq``'s tokens.

        A step processes ``seq`` up to its current ``seq_len``; the cache must
        cover ``ceil(seq_len / block_size)`` blocks in total. The delta over
        what's already allocated is what this step will lazily allocate during
        the forward pass.
        """
        block_size = self.block_manager.config.block_size
        target_blocks = (seq.seq_len + block_size - 1) // block_size
        return max(0, target_blocks - already_allocated)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> List[Sequence]:
        """
        Select the sequences to run this step (mixed prefill + decode).

        Returns them in batch order — the order the engine must use to pack
        ``input_ids`` / ``cu_seqlens`` / ``seq_ids`` / ``seq_lens``.
        """
        batch: List[Sequence] = []
        token_budget = self.config.max_num_batched_tokens
        free_blocks = self.block_manager.num_free_blocks

        # --- Phase 1: running sequences decode. Decode has priority for blocks. ---
        for seq in self.running:
            need = self._blocks_to_add(
                seq, self.block_manager.num_blocks_for_sequence(seq.seq_id)
            )
            if need > free_blocks:
                # Preemption/eviction is out of scope: there is nothing we are
                # allowed to reclaim. Fail loudly rather than corrupt the cache.
                raise RuntimeError(
                    f"Out of KV blocks for running sequence {seq.seq_id} on "
                    f"decode (need {need}, free {free_blocks}); preemption is "
                    f"not implemented."
                )
            free_blocks -= need
            token_budget -= seq.num_uncached_tokens
            batch.append(seq)

        # --- Phase 2: admit waiting sequences (prefill), strict FIFO. ---
        while self.waiting and len(batch) < self.config.max_num_seqs:
            seq = self.waiting.peek()
            n_tokens = seq.num_uncached_tokens          # full prompt at prefill
            need = self._blocks_to_add(seq, already_allocated=0)

            # Token budget is a soft target, but always admit at least one
            # sequence so the engine can't deadlock on a prompt bigger than the
            # budget (splitting it would need chunked prefill — out of scope).
            if batch and n_tokens > token_budget:
                break

            if need > free_blocks:
                if not batch:
                    # Nothing running to free blocks later and it still won't
                    # fit: the prompt is too big for the whole cache.
                    raise RuntimeError(
                        f"Sequence {seq.seq_id} needs {need} KV blocks but only "
                        f"{free_blocks} are free with nothing running to evict; "
                        f"prompt too long for the KV cache."
                    )
                break

            self.waiting.pop()
            self.block_manager.register_sequence(seq.seq_id)
            seq.mark_running()
            self.running.append(seq)

            free_blocks -= need
            token_budget -= n_tokens
            batch.append(seq)

        return batch

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def free_finished(self) -> List[Sequence]:
        """
        Remove finished sequences from the running list and free their blocks.

        Returns the sequences evicted this call, in running-list order.
        """
        finished = [seq for seq in self.running if seq.done]
        if finished:
            self.running = [seq for seq in self.running if not seq.done]
            for seq in finished:
                self.block_manager.free_sequence(seq.seq_id)
        return finished

    def __repr__(self) -> str:
        return (
            f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)})"
        )
