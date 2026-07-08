"""
Continuous-batching engine loop.

Drives sequences from submission to completion. Each :meth:`LLMEngine.step`:

    1. asks the :class:`Scheduler` for one mixed prefill/decode batch,
    2. packs it into the model's 1-D buffers,
    3. runs a single forward pass,
    4. samples one token per sequence, advancing the KV-cache watermark and
       appending to each sequence, and
    5. evicts any sequence that finished, freeing its KV blocks.

The model is held behind a callable with the signature of
:meth:`LlamaModel.forward`, so the loop can be exercised on CPU with a stub
model (no CUDA / flash-attn) as well as with the real paged-attention model.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch

from .block_manager import BlockManager
from .sampler import sample_token
from .scheduler import Scheduler, SchedulerConfig
from .sequence import (
    SamplingParams,
    Sequence,
    SequenceIdAllocator,
)


# The model is any callable matching LlamaModel.forward:
#   (input_ids, position_ids, cu_seqlens, block_manager, seq_ids, seq_lens)
#   -> logits [total_tokens, vocab_size]
ModelFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, BlockManager, List[int], List[int]],
    torch.Tensor,
]


class LLMEngine:
    """
    Owns the model, KV block pool, scheduler, and request id allocation.

    Parameters
    ----------
    model         : Callable with :meth:`LlamaModel.forward`'s signature.
    block_manager : Shared paged KV cache allocator.
    scheduler     : Continuous-batching scheduler over ``block_manager``.
    device        : Device for the packed input tensors (must match the model).
    """

    def __init__(
        self,
        model: ModelFn,
        block_manager: BlockManager,
        scheduler: Scheduler,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.block_manager = block_manager
        self.scheduler = scheduler
        self.device = torch.device(device)
        self._ids = SequenceIdAllocator()

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: ModelFn,
        block_manager: BlockManager,
        scheduler_config: SchedulerConfig,
        device: torch.device | str = "cuda",
    ) -> "LLMEngine":
        """Convenience constructor that wires up the scheduler."""
        scheduler = Scheduler(block_manager, scheduler_config)
        return cls(model, block_manager, scheduler, device)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> int:
        """
        Submit a prompt for generation. Returns the assigned sequence id.
        """
        seq_id = self._ids.next_id()
        seq = Sequence(seq_id, prompt_token_ids, sampling_params)
        self.scheduler.add_request(seq)
        return seq_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------------
    # Input packing
    # ------------------------------------------------------------------

    def _build_inputs(
        self, batch: List[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Pack a scheduled batch into the model's 1-D buffers.

        Each sequence contributes its *uncached* tokens
        (``all_token_ids[num_cached_tokens:]``) — the whole prompt on prefill,
        a single token on decode — at absolute positions
        ``[num_cached_tokens, seq_len)``.

        Returns
        -------
        input_ids   : [total_tokens]   int64
        position_ids: [total_tokens]   int32
        cu_seqlens  : [num_seqs + 1]   int32
        seq_ids     : per-sequence ids, batch order
        seq_lens    : tokens cached *before* this step, per sequence
        """
        flat_tokens: List[int] = []
        flat_positions: List[int] = []
        cu: List[int] = [0]
        seq_ids: List[int] = []
        seq_lens: List[int] = []

        for seq in batch:
            start = seq.num_cached_tokens
            new_tokens = seq.all_token_ids[start:]
            flat_tokens.extend(new_tokens)
            flat_positions.extend(range(start, seq.seq_len))
            cu.append(cu[-1] + len(new_tokens))
            seq_ids.append(seq.seq_id)
            seq_lens.append(start)

        input_ids = torch.tensor(flat_tokens, dtype=torch.int64, device=self.device)
        position_ids = torch.tensor(flat_positions, dtype=torch.int32, device=self.device)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=self.device)
        return input_ids, position_ids, cu_seqlens, seq_ids, seq_lens

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> List[Sequence]:
        """
        Run one scheduling + forward + sample + evict iteration.

        Returns the sequences that finished this step (in eviction order);
        empty if nothing was scheduled.
        """
        batch = self.scheduler.schedule()
        if not batch:
            return []

        input_ids, position_ids, cu_seqlens, seq_ids, seq_lens = self._build_inputs(batch)

        logits = self.model(
            input_ids, position_ids, cu_seqlens, self.block_manager, seq_ids, seq_lens
        )

        # Logits for the last position of each packed segment.
        last_idx = cu_seqlens[1:].long() - 1
        last_logits = logits[last_idx]  # [num_seqs, vocab_size]

        for i, seq in enumerate(batch):
            # The forward pass wrote every uncached token into the cache; the
            # watermark advances before the newly sampled token is appended.
            seq.advance_cache(seq.num_uncached_tokens)
            token_id = sample_token(last_logits[i], seq.sampling_params)
            seq.append_token(token_id)

        return self.scheduler.free_finished()

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_to_completion(self) -> List[Sequence]:
        """
        Step until every submitted request has finished.

        Returns finished sequences in completion order. Intended for offline
        batch generation; a streaming server would consume :meth:`step`
        directly.
        """
        finished: List[Sequence] = []
        while self.has_unfinished():
            finished.extend(self.step())
        return finished
