"""
Per-step attention metadata for the paged KV cache.

Everything attention needs to locate KV in the paged cache is computed once
per engine step, on the CPU, from Python state the engine already holds (the
scheduled :class:`Sequence` objects and the :class:`BlockManager`'s block
tables). It is copied to the device in a handful of batched transfers and then
handed unchanged to every layer, so the forward pass itself never syncs with
the host or rebuilds index tensors.

Kept free of model / engine imports so the builder can be tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .block_manager import BlockManager
from .sequence import Sequence


# Fills block_table entries past a sequence's last allocated block. Attention
# only reads the first ceil(seqused_k[i] / block_size) entries of row i, so
# padding is never read in correct operation; if a bug ever reads it, -1 gives
# an out-of-range access rather than silently reading a real (wrong) block.
BLOCK_TABLE_PAD = -1


@dataclass(frozen=True)
class AttentionMetadata:
    """
    Paged-attention inputs for one engine step, shared by all layers.

    GPU tensors (on the model's device)
    -----------------------------------
    slot_mapping : [total_tokens]           int64. Flat cache slot of each new
                   token: ``physical_block * block_size + pos % block_size``,
                   an index into the cache viewed as
                   ``[num_blocks * block_size, H, D]``. int64 because
                   ``index_copy_`` / ``index_put_`` require a long index.
    block_table  : [num_seqs, max_blocks]   int32. Physical block ids in
                   logical order, right-padded with :data:`BLOCK_TABLE_PAD`.
    cu_seqlens_q : [num_seqs + 1]           int32. Cumulative new-token counts;
                   sequence *i* occupies ``[cu[i], cu[i+1])`` of the packed
                   buffers.
    seqused_k    : [num_seqs]               int32. Cached + new tokens per
                   sequence (the KV length attention reads).

    ``block_table``, ``cu_seqlens_q`` and ``seqused_k`` are int32 because
    vLLM's ``flash_attn_varlen_func`` requires it.

    Host scalars
    ------------
    max_seqlen_q : Longest new-token run in the batch.
    max_seqlen_k : Longest KV length in the batch.

    Computed from Python lists; deriving them from a device tensor via
    ``.max().item()`` would force a sync.

    Host copies (for the engine, test stubs, debugging)
    ---------------------------------------------------
    seq_ids        : Sequence ids, batch order.
    num_cached     : Tokens in the cache *before* this step, per sequence.
    cu_seqlens_cpu : ``cu_seqlens_q`` as a list.
    """

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor

    max_seqlen_q: int
    max_seqlen_k: int

    seq_ids: List[int]
    num_cached: List[int]
    cu_seqlens_cpu: List[int]


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Copy a CPU tensor to ``device``. CUDA copies go through pinned memory so
    ``non_blocking`` is truly asynchronous (from pageable memory it degrades to
    a synchronous copy). The caching host allocator keeps the pinned buffer
    alive until the copy completes. Pinning needs CUDA, so CPU is a plain move.
    """
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def build_step_inputs(
    batch: List[Sequence],
    block_manager: BlockManager,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
    """
    Pack a scheduled batch into the model's 1-D buffers and attention metadata.

    Each sequence contributes its *uncached* tokens — the whole prompt on
    prefill, a single token on decode — at absolute positions
    ``[num_cached_tokens, seq_len)``. Every sequence must already have blocks
    covering ``seq_len`` tokens (the scheduler allocates them before the
    forward pass); this function never allocates.

    Returns
    -------
    input_ids    : [total_tokens]   int64
    position_ids : [total_tokens]   int32
    attn_metadata: :class:`AttentionMetadata`

    Raises
    ------
    ValueError   if the batch is empty or a sequence has no uncached tokens.
    RuntimeError if a sequence has fewer blocks than ``seq_len`` needs.
    """
    if not batch:
        raise ValueError("Cannot build step inputs for an empty batch.")

    device = torch.device(device)
    block_size = block_manager.config.block_size

    # --- Per-sequence Python pass ---
    # Positions and slots are built here as plain lists, not with CPU tensor
    # ops: CPU repeat_interleave wakes the intra-op thread pool for any batch
    # over one sequence, and the spinning workers starved the main thread,
    # doubling batched decode ITL.
    flat_tokens: List[int] = []
    positions: List[int] = []
    slot_mapping: List[int] = []
    cu: List[int] = [0]
    seq_ids: List[int] = []
    num_cached: List[int] = []
    num_new: List[int] = []
    seqused: List[int] = []
    block_rows: List[List[int]] = []

    for seq in batch:
        start = seq.num_cached_tokens
        end = seq.seq_len
        n_new = end - start
        if n_new <= 0:
            raise ValueError(
                f"Sequence {seq.seq_id} has no uncached tokens to process "
                f"(cached {start}, seq_len {end})."
            )

        flat_tokens.extend(seq.uncached_token_ids)
        cu.append(cu[-1] + n_new)
        seq_ids.append(seq.seq_id)
        num_cached.append(start)
        num_new.append(n_new)
        seqused.append(end)

        block_ids = block_manager.get_block_ids(seq.seq_id)
        blocks_needed = -(-end // block_size)
        if len(block_ids) < blocks_needed:
            raise RuntimeError(
                f"Sequence {seq.seq_id} has {len(block_ids)} KV blocks but "
                f"needs {blocks_needed} for {end} tokens; blocks must be "
                f"allocated before the forward pass."
            )
        block_rows.append(block_ids[:blocks_needed])

        # The allocation check above guarantees block_ids covers every
        # position below end.
        positions.extend(range(start, end))
        slot_mapping.extend(
            block_ids[p // block_size] * block_size + p % block_size
            for p in range(start, end)
        )

    max_seqlen_q = max(num_new)
    max_seqlen_k = max(seqused)

    # --- Block table (CPU) ---
    max_blocks = -(-max_seqlen_k // block_size)
    padded = [row + [BLOCK_TABLE_PAD] * (max_blocks - len(row)) for row in block_rows]
    block_table_cpu = torch.tensor(padded, dtype=torch.int32)

    # --- Host -> device ---
    input_ids = _to_device(torch.tensor(flat_tokens, dtype=torch.int64), device)
    position_ids = _to_device(torch.tensor(positions, dtype=torch.int32), device)
    attn_metadata = AttentionMetadata(
        slot_mapping=_to_device(torch.tensor(slot_mapping, dtype=torch.int64), device),
        block_table=_to_device(block_table_cpu, device),
        cu_seqlens_q=_to_device(torch.tensor(cu, dtype=torch.int32), device),
        seqused_k=_to_device(torch.tensor(seqused, dtype=torch.int32), device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seq_ids=seq_ids,
        num_cached=num_cached,
        cu_seqlens_cpu=cu,
    )
    return input_ids, position_ids, attn_metadata
