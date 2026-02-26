"""
KV cache block (page) for paged attention.

A KVBlock pre-allocates a fixed-size contiguous region of GPU memory that
holds key and value tensors for `block_size` token positions across every
attention layer of a transformer model. Blocks are the unit of allocation
used by the block manager (not implemented here).

Tensor layout: [num_layers, block_size, num_kv_heads, head_dim]

  - num_layers  : one entry per transformer attention layer
  - block_size  : fixed capacity in tokens (e.g. 16)
  - num_kv_heads: number of key/value heads (may differ from query heads in GQA)
  - head_dim    : per-head feature dimension
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVBlockConfig:
    """Immutable description of a block's shape and placement."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    @property
    def bytes_per_block(self) -> int:
        """Total GPU memory consumed by one block (both K and V)."""
        elems = 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim
        return elems * torch.finfo(self.dtype).bits // 8


# ---------------------------------------------------------------------------
# Block state
# ---------------------------------------------------------------------------

class BlockState(Enum):
    FREE      = auto()   # available for allocation
    ALLOCATED = auto()   # owned by one or more sequences


# ---------------------------------------------------------------------------
# KVBlock
# ---------------------------------------------------------------------------

class KVBlock:
    """
    A single fixed-size page of KV cache.

    Memory is allocated once at construction and reused across requests via
    reset(). The block tracks how many of its slots have been written so that
    callers can read back only the live portion.

    Typical usage
    -------------
    Prefill (write a prompt's worth of tokens):
        for token_pos in range(prompt_len):
            block.append(keys[token_pos], values[token_pos])

    Decode (append one token per step):
        block.append(k_new, v_new)

    Retrieve for attention:
        k, v = block.get_filled_kv()   # shape: [L, filled, H, D]
    """

    def __init__(self, block_id: int, config: KVBlockConfig) -> None:
        self.block_id = block_id
        self.config   = config
        self.state    = BlockState.FREE
        self._num_filled = 0

        shape = (config.num_layers, config.block_size, config.num_kv_heads, config.head_dim)
        self.k_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.v_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_filled(self) -> int:
        """Number of token slots that have been written."""
        return self._num_filled

    @property
    def num_free(self) -> int:
        """Number of remaining writable slots."""
        return self.config.block_size - self._num_filled

    @property
    def is_full(self) -> bool:
        return self._num_filled >= self.config.block_size

    @property
    def is_empty(self) -> bool:
        return self._num_filled == 0

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> int:
        """
        Write one token's KV tensors (all layers) into the next free slot.

        Args:
            keys:   [num_layers, num_kv_heads, head_dim]
            values: [num_layers, num_kv_heads, head_dim]

        Returns:
            The slot index that was written.

        Raises:
            RuntimeError if the block is already full.
        """
        if self.is_full:
            raise RuntimeError(
                f"KVBlock {self.block_id} is full "
                f"({self.config.block_size}/{self.config.block_size} slots used)"
            )
        slot = self._num_filled
        self.k_cache[:, slot] = keys
        self.v_cache[:, slot] = values
        self._num_filled += 1
        return slot

    def write_slot(
        self,
        slot_idx: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write K/V for a single (slot, layer) pair. Useful for random-access
        writes (e.g. prefix caching, copy-on-write).

        Args:
            slot_idx:  Position within the block [0, block_size)
            layer_idx: Attention layer index
            k:         [num_kv_heads, head_dim]
            v:         [num_kv_heads, head_dim]
        """
        if not (0 <= slot_idx < self.config.block_size):
            raise IndexError(f"slot_idx {slot_idx} out of range [0, {self.config.block_size})")
        if not (0 <= layer_idx < self.config.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.config.num_layers})")
        self.k_cache[layer_idx, slot_idx] = k
        self.v_cache[layer_idx, slot_idx] = v

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def read_slot(self, slot_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read K/V tensors for a single slot across all layers.

        Returns:
            (k, v) each of shape [num_layers, num_kv_heads, head_dim]
        """
        if not (0 <= slot_idx < self.config.block_size):
            raise IndexError(f"slot_idx {slot_idx} out of range [0, {self.config.block_size})")
        return self.k_cache[:, slot_idx], self.v_cache[:, slot_idx]

    def get_filled_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return only the live (filled) portion of the cache.

        Returns:
            (k, v) each of shape [num_layers, num_filled, num_kv_heads, head_dim]
        """
        return (
            self.k_cache[:, : self._num_filled],
            self.v_cache[:, : self._num_filled],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return the block to a free, empty state (memory is retained)."""
        self._num_filled = 0
        self.state = BlockState.FREE

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_filled

    def __repr__(self) -> str:
        return (
            f"KVBlock(id={self.block_id}, "
            f"filled={self._num_filled}/{self.config.block_size}, "
            f"state={self.state.name}, "
            f"device={self.config.device})"
        )
