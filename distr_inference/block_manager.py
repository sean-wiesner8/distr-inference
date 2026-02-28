"""
Block manager for paged KV cache.

Manages a fixed pool of pre-allocated KVBlocks and maps logical block indices
to physical block IDs on a per-sequence basis.

Concepts
--------
Physical block : A KVBlock instance with a unique integer ID. All physical
                 blocks are allocated once at startup and live in the free pool
                 until assigned to a sequence.

Logical block  : A sequence-local index (0, 1, 2, ...) that grows as a
                 sequence produces more tokens. The block table translates
                 logical → physical.

Block table    : Per-sequence dict[int, int]  (logical_idx → physical_block_id)
"""

from __future__ import annotations

from collections import deque
from typing import Dict

from .kv_cache import KVBlock, KVBlockConfig


class BlockManager:
    """
    Manages a fixed pool of KVBlocks and per-sequence block tables.

    Parameters
    ----------
    num_blocks : int
        Total number of physical blocks to pre-allocate.
    config : KVBlockConfig
        Shape / dtype / device configuration forwarded to every KVBlock.
    """

    def __init__(self, num_blocks: int, config: KVBlockConfig) -> None:
        self.config = config
        self.num_blocks = num_blocks

        # Pre-allocate all physical blocks
        self._blocks: Dict[int, KVBlock] = {
            i: KVBlock(block_id=i, config=config) for i in range(num_blocks)
        }

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
            block = self._blocks[physical_id]
            block.reset()
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
        block = self._blocks[physical_id]
        block.state  # block.reset() was called on free, so state is FREE

        logical_idx = len(self._block_tables[seq_id])
        self._block_tables[seq_id][logical_idx] = physical_id
        return logical_idx

    # ------------------------------------------------------------------
    # Block access
    # ------------------------------------------------------------------

    def get_block(self, seq_id: int, logical_idx: int) -> KVBlock:
        """
        Retrieve the physical KVBlock for a given (sequence, logical index) pair.

        Raises
        ------
        KeyError if seq_id or logical_idx is not found.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} is not registered.")
        table = self._block_tables[seq_id]
        if logical_idx not in table:
            raise KeyError(
                f"Logical block {logical_idx} not allocated for sequence {seq_id}."
            )
        return self._blocks[table[logical_idx]]

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
