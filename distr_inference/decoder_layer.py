"""
One transformer block in the Llama decoder stack.

Pre-norm layout: ``input_layernorm`` → ``self_attn`` → residual →
``post_attention_layernorm`` → ``mlp`` → residual.

Operates on packed ``[total_tokens, hidden]`` buffers and threads the
paged-attention metadata (``cu_seqlens``, ``seq_ids``, ``seq_lens``,
``block_manager``) straight through to :class:`PagedAttention`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import LlamaConfig

from .attention import PagedAttention
from .block_manager import BlockManager
from .mlp import LlamaMLP
from .rms_norm import LlamaRMSNorm


class LlamaDecoderLayer(nn.Module):
    """
    A single Llama decoder block.

    Parameters
    ----------
    hf_config  : HuggingFace ``LlamaConfig`` driving submodule shapes and the
                 RMSNorm epsilon. Also handed to ``LlamaMLP``.
    layer_idx  : Index of this layer within the transformer stack; used by
                 ``PagedAttention`` to select its slice of the shared KV cache.
    rotary_emb : Shared rotary embedding module. Built once at the model level
                 and reused across all layers — do not construct per layer.
    """

    def __init__(
        self,
        hf_config: LlamaConfig,
        layer_idx: int,
        rotary_emb: nn.Module,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", None) or hidden_size // num_heads
        eps = hf_config.rms_norm_eps

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=eps)

        q_proj  = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        kv_proj = nn.Linear(hidden_size, 2 * num_kv_heads * head_dim, bias=False)
        o_proj  = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.self_attn = PagedAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=q_proj,
            kv_proj=kv_proj,
            o_proj=o_proj,
            rotary_emb=rotary_emb,
        )

        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=eps)
        self.mlp = LlamaMLP(hf_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Forward pass on a packed 1-D token buffer.

        Parameters
        ----------
        hidden_states : [total_tokens, hidden]

        Returns
        -------
        out : [total_tokens, hidden]

        See ``PagedAttention.forward`` for the meaning of ``position_ids``,
        ``cu_seqlens``, ``seq_ids``, and ``seq_lens``.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
            cu_seqlens,
            block_manager,
            seq_ids,
            seq_lens,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
