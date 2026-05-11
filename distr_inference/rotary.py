"""
Thin wrapper around HF's ``LlamaRotaryEmbedding``.

HF's module returns cos/sin shaped ``[batch, seq, head_dim]``. Our
``PagedAttention`` multiplies cos/sin against q/k tensors shaped
``[batch, seq, num_heads, head_dim]`` (see ``attention.py``), so we need a
length-1 ``num_heads`` dim inserted at position 2 for broadcasting to work.
This wrapper adds that ``unsqueeze(2)``; everything else (including Llama-3
NTK frequency scaling) is delegated to HF.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding as HFLlamaRotaryEmbedding,
)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.inner = HFLlamaRotaryEmbedding(config=config)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.inner(x, position_ids)
        return cos.unsqueeze(2), sin.unsqueeze(2)
