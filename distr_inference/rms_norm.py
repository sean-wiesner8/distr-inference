"""
Thin wrapper around HF's ``LlamaRMSNorm``.

The HF implementation operates on the last dim only, so it works on packed
``[total_tokens, hidden]`` buffers without modification. This wrapper exists
purely to give the rest of the codebase a stable construction signature
(``LlamaRMSNorm(hidden_size, eps)``) that won't break if HF reorganises its
internals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm as HFLlamaRMSNorm,
)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.inner = HFLlamaRMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)
