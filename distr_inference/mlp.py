"""
Thin wrapper around HF's ``LlamaMLP``.

``LlamaMLP`` is the SwiGLU block: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
All three projections act on the last dim, so the module works on packed
``[total_tokens, hidden]`` buffers as-is. The wrapper isolates HF's internal
constructor signature from the rest of the codebase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP as HFLlamaMLP


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.inner = HFLlamaMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)
