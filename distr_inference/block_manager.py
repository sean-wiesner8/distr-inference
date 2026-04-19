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
from typing import Dict, Tuple

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
        self._num_filled: list[int] = [0] * num_blocks

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
            self._num_filled[physical_id] = 0
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

    def num_blocks_for_sequence(self, seq_id: int) -> int:
        """Return how many physical blocks are currently allocated to seq_id."""
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        return len(self._block_tables[seq_id])

    # ------------------------------------------------------------------
    # Cache read / write
    # ------------------------------------------------------------------

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return views of the K and V cache for a single layer.

        Returns
        -------
        (k, v) each of shape [num_blocks, block_size, num_kv_heads, head_dim]
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def write_slot(
        self,
        physical_block_id: int,
        slot_idx: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write K/V for a single (block, slot, layer) triple.

        Parameters
        ----------
        physical_block_id : Index into the cache's block dimension.
        slot_idx          : Position within the block [0, block_size).
        layer_idx         : Attention layer index.
        k                 : [num_kv_heads, head_dim]
        v                 : [num_kv_heads, head_dim]
        """
        self.k_cache[layer_idx, physical_block_id, slot_idx] = k
        self.v_cache[layer_idx, physical_block_id, slot_idx] = v

    def get_num_filled(self, physical_block_id: int) -> int:
        """Return how many slots have been written in a block."""
        return self._num_filled[physical_block_id]

    def set_num_filled(self, physical_block_id: int, count: int) -> None:
        """Update the filled-slot count for a block."""
        self._num_filled[physical_block_id] = count

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
