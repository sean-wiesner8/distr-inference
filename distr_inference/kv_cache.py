"""
KV cache configuration for paged attention.

The actual cache storage is a pair of contiguous buffers managed by
BlockManager.  This module provides the configuration dataclass and
block-state enum.

Buffer layout (per K and V):
    [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
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
