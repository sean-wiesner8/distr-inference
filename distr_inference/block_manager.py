"""
Block manager for paged KV cache.

Manages a fixed pool of pre-allocated blocks backed by contiguous K/V buffers
and maps logical block indices to physical block IDs on a per-sequence basis.

The KV cache is stored as two contiguous tensors (one for K, one for V) with
shape [num_layers, num_blocks, block_size, num_kv_heads, head_dim].  This
layout lets flash-attention index directly into the cache using a block table
without gathering into temporary buffers.

Concepts
--------
Physical block : A slice along the ``num_blocks`` dimension of the cache
                 tensors, identified by an integer ID.  All physical blocks
                 are pre-allocated at startup and live in the free pool until
                 assigned to a sequence.

Logical block  : A sequence-local index (0, 1, 2, ...) that grows as a
                 sequence produces more tokens. The block table translates
                 logical → physical.

Block table    : Per-sequence dict[int, int]  (logical_idx → physical_block_id)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import torch

from .kv_cache import KVBlockConfig, BlockState


class BlockManager:
    """
    Manages a contiguous KV cache pool and per-sequence block tables.

    Parameters
    ----------
    num_blocks : int
        Total number of physical blocks to pre-allocate.
    config : KVBlockConfig
        Shape / dtype / device configuration for the cache buffers.
    """

    def __init__(self, num_blocks: int, config: KVBlockConfig) -> None:
        self.config = config
        self.num_blocks = num_blocks

        # Contiguous KV cache buffers
        shape = (
            config.num_layers,
            num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        )
        self.k_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.v_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)

        # Per-block metadata
        self._block_states: list[BlockState] = [BlockState.FREE] * num_blocks

        # Free pool — deque for O(1) pop/append
        self._free_pool: deque[int] = deque(range(num_blocks))

        # Per-sequence block tables: seq_id → {logical_idx: physical_block_id}
        self._block_tables: Dict[int, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Pool queries
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_pool)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free_pool)

    def can_allocate(self, num_blocks: int = 1) -> bool:
        """Return True if at least `num_blocks` physical blocks are free."""
        return len(self._free_pool) >= num_blocks

    # ------------------------------------------------------------------
    # Sequence lifetime
    # ------------------------------------------------------------------

    def register_sequence(self, seq_id: int) -> None:
        """
        Register a new sequence. Must be called before allocating blocks for it.

        Raises
        ------
        ValueError if seq_id is already registered.
        """
        if seq_id in self._block_tables:
            raise ValueError(f"Sequence {seq_id} is already registered.")
        self._block_tables[seq_id] = {}

    def free_sequence(self, seq_id: int) -> None:
        """
        Release all blocks owned by a sequence and remove its block table.

        Raises
        ------
        KeyError if seq_id is not registered.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")

        for physical_id in self._block_tables[seq_id].values():
            self._block_states[physical_id] = BlockState.FREE
            self._free_pool.append(physical_id)

        del self._block_tables[seq_id]

    # ------------------------------------------------------------------
    # Block allocation
    # ------------------------------------------------------------------

    def allocate_block(self, seq_id: int) -> int:
        """
        Append one new physical block to a sequence's block table.

        The logical index assigned is len(current_block_table), i.e. blocks
        are always appended in order.

        Returns
        -------
        logical_idx : int
            The logical block index just assigned.

        Raises
        ------
        KeyError    if seq_id is not registered.
        RuntimeError if the free pool is empty.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        if not self._free_pool:
            raise RuntimeError("Out of physical KV cache blocks.")

        physical_id = self._free_pool.popleft()
        self._block_states[physical_id] = BlockState.ALLOCATED

        logical_idx = len(self._block_tables[seq_id])
        self._block_tables[seq_id][logical_idx] = physical_id
        return logical_idx

    def ensure_capacity(self, seq_id: int, num_tokens: int) -> int:
        """
        Grow a sequence's block table until it can hold ``num_tokens`` tokens.

        All-or-nothing: if the pool can't cover the shortfall, nothing is
        allocated.

        Returns
        -------
        The number of blocks newly allocated (0 if already large enough).

        Raises
        ------
        KeyError    if seq_id is not registered.
        RuntimeError if the free pool can't cover the shortfall.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")

        block_size = self.config.block_size
        target = (num_tokens + block_size - 1) // block_size
        need = max(0, target - len(self._block_tables[seq_id]))
        if need > len(self._free_pool):
            raise RuntimeError(
                f"Out of physical KV cache blocks (sequence {seq_id} needs "
                f"{need}, {len(self._free_pool)} free)."
            )
        for _ in range(need):
            self.allocate_block(seq_id)
        return need

    # ------------------------------------------------------------------
    # Block table access
    # ------------------------------------------------------------------

    def get_block_table(self, seq_id: int) -> Dict[int, int]:
        """
        Return a copy of the block table for a sequence.

        Returns
        -------
        dict mapping logical_idx → physical_block_id.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        return dict(self._block_tables[seq_id])

    def get_block_ids(self, seq_id: int) -> List[int]:
        """
        Return a sequence's physical block IDs in logical order.

        Blocks are only ever appended, so list position == logical index.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        return list(self._block_tables[seq_id].values())

    def num_blocks_for_sequence(self, seq_id: int) -> int:
        """Return how many physical blocks are currently allocated to seq_id."""
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        return len(self._block_tables[seq_id])

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return views of the K and V cache for a single layer.

        The views share storage with the full cache, so writes through them
        (``PagedAttention._write_kv_to_paged_cache``) update it in place.

        Returns
        -------
        (k, v) each of shape [num_blocks, block_size, num_kv_heads, head_dim]
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BlockManager("
            f"total={self.num_blocks}, "
            f"free={self.num_free_blocks}, "
            f"used={self.num_used_blocks}, "
            f"sequences={list(self._block_tables.keys())})"
        )
