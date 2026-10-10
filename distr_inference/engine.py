"""
Continuous-batching engine loop.

Drives sequences from submission to completion. Each :meth:`LLMEngine.step`:

    1. asks the :class:`Scheduler` for one mixed prefill/decode batch,
    2. packs it into the model's 1-D buffers and attention metadata,
    3. runs a single forward pass,
    4. samples one token per sequence, advancing the KV-cache watermark and
       appending to each sequence, and
    5. evicts any sequence that finished, freeing its KV blocks.

The model is held behind a callable with the signature of
:meth:`LlamaModel.forward`, so the loop can be exercised on CPU with a stub
model (no CUDA / flash-attn) as well as with the real paged-attention model.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch

from .attention_metadata import AttentionMetadata, build_step_inputs
from .block_manager import BlockManager
from .sampler import sample_token
from .scheduler import Scheduler, SchedulerConfig
from .sequence import (
    SamplingParams,
    Sequence,
    SequenceIdAllocator,
)


# The model is any callable matching LlamaModel.forward:
#   (input_ids, position_ids, attn_metadata, block_manager)
#   -> logits [total_tokens, vocab_size]
ModelFn = Callable[
    [torch.Tensor, torch.Tensor, AttentionMetadata, BlockManager],
    torch.Tensor,
]

# Picks the next token for a sequence from its last-position logits row.
#   (seq, logits [vocab_size]) -> token_id
SamplerFn = Callable[[Sequence, torch.Tensor], int]


def default_sampler(seq: Sequence, logits: torch.Tensor) -> int:
    """Sample under the sequence's own :class:`SamplingParams`."""
    return sample_token(logits, seq.sampling_params)


class LLMEngine:
    """
    Owns the model, KV block pool, scheduler, and request id allocation.

    Parameters
    ----------
    model         : Callable with :meth:`LlamaModel.forward`'s signature.
    block_manager : Shared paged KV cache allocator.
    scheduler     : Continuous-batching scheduler over ``block_manager``.
    device        : Device for the packed input tensors (must match the model).
    sampler       : Token picker; defaults to :func:`default_sampler`. Tests
                    override it to record logits or force a token path.
    eos_token_ids : The model's EOS ids (see
                    :func:`~distr_inference.config.resolve_eos_token_ids`).
                    Every request stops on these unless it sets
                    ``SamplingParams.ignore_eos``.
    """

    def __init__(
        self,
        model: ModelFn,
        block_manager: BlockManager,
        scheduler: Scheduler,
        device: torch.device | str = "cuda",
        sampler: SamplerFn = default_sampler,
        eos_token_ids: Iterable[int] = (),
    ) -> None:
        self.model = model
        self.block_manager = block_manager
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.sampler = sampler
        self.eos_token_ids: Tuple[int, ...] = tuple(eos_token_ids)
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
        sampler: SamplerFn = default_sampler,
        eos_token_ids: Iterable[int] = (),
    ) -> "LLMEngine":
        """Convenience constructor that wires up the scheduler."""
        scheduler = Scheduler(block_manager, scheduler_config)
        return cls(model, block_manager, scheduler, device, sampler, eos_token_ids)

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
        seq = Sequence(
            seq_id, prompt_token_ids, sampling_params,
            eos_token_ids=self.eos_token_ids,
        )
        self.scheduler.add_request(seq)
        return seq_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

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

        # Each sequence contributes its uncached tokens (the whole prompt on
        # prefill, one token on decode). Everything attention needs is built
        # here, once, and shared by every layer.
        input_ids, position_ids, attn_metadata = build_step_inputs(
            batch, self.block_manager, self.device
        )

        logits = self.model(input_ids, position_ids, attn_metadata, self.block_manager)

        # Logits for the last position of each packed segment.
        last_idx = attn_metadata.cu_seqlens_q[1:].long() - 1
        last_logits = logits[last_idx]  # [num_seqs, vocab_size]

        for i, seq in enumerate(batch):
            # The forward pass wrote every uncached token into the cache; the
            # watermark advances before the newly sampled token is appended.
            seq.advance_cache(seq.num_uncached_tokens)
            token_id = self.sampler(seq, last_logits[i])
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
