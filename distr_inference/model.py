"""
Top-level Llama decoder model.

Wires ``embed_tokens`` → stack of :class:`LlamaDecoderLayer` → final
``norm`` → ``lm_head`` on packed ``[total_tokens, hidden]`` buffers. Owns the
single shared :class:`LlamaRotaryEmbedding` instance reused by every layer,
and ties ``lm_head.weight`` to ``embed_tokens.weight`` when the checkpoint
config requests it (true for Llama-3.2-1B).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import LlamaConfig

from .block_manager import BlockManager
from .decoder_layer import LlamaDecoderLayer
from .rms_norm import LlamaRMSNorm
from .rotary import LlamaRotaryEmbedding


class LlamaModel(nn.Module):
    """
    Full Llama decoder stack.

    Takes a packed batch (mixed prefill/decode) and returns logits for every
    token in that buffer. The scheduler is responsible for slicing the last
    position of each segment using ``cu_seqlens`` when sampling; doing it
    here would force a branch between prefill and decode that the packed
    layout exists to avoid.

    All KV cache state lives in the ``BlockManager`` passed into each
    ``forward`` — the model itself is stateless across calls.
    """

    def __init__(self, hf_config: LlamaConfig) -> None:
        super().__init__()
        self.hf_config = hf_config

        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)

        self.rotary_emb = LlamaRotaryEmbedding(hf_config)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hf_config, layer_idx=i, rotary_emb=self.rotary_emb)
            for i in range(hf_config.num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)

        if hf_config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
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
        input_ids : [total_tokens]

        Returns
        -------
        logits : [total_tokens, vocab_size]
            Logits for every packed token. The scheduler slices the last
            position of each segment (``cu_seqlens[i+1] - 1``) when sampling.

        See ``PagedAttention.forward`` for the meaning of ``position_ids``,
        ``cu_seqlens``, ``seq_ids``, and ``seq_lens``.
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
                cu_seqlens,
                block_manager,
                seq_ids,
                seq_lens,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
