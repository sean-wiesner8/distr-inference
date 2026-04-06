"""
Paged attention module for dropping into HuggingFace transformer models.

Replace any HF attention layer with PagedAttention to get a paged KV cache
backed by BlockManager / KVBlock.  Inference state (block manager, sequence
IDs, sequence lengths, phase flag) is set globally on the class before each
forward pass via set_state(), so individual forward calls keep the standard
HF attention signature.

Typical usage
-------------
    model = LlamaForCausalLM.from_pretrained(...)
    block_manager = BlockManager(num_blocks=512, config=kv_cfg)

    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PagedAttention.from_hf_attention(
            layer.self_attn, model.config, i
        )

    # Before each forward pass:
    PagedAttention.set_state(
        model, block_manager, seq_ids, seq_lens, is_prefill=True
    )
    outputs = model(input_ids)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .block_manager import BlockManager


# ---------------------------------------------------------------------------
# Rotary helpers (self-contained; avoids coupling to HF model internals)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ---------------------------------------------------------------------------
# PagedAttention
# ---------------------------------------------------------------------------

class PagedAttention(nn.Module):
    """
    Drop-in replacement for HF self-attention layers with a paged KV cache.

    Parameters
    ----------
    layer_idx   : Index of this layer within the transformer stack.
    num_heads   : Number of query attention heads.
    num_kv_heads: Number of key/value heads (< num_heads for GQA/MQA).
    head_dim    : Per-head feature dimension.
    q_proj      : Query projection (nn.Linear, weights copied from the HF layer).
    kv_proj     : Fused key/value projection with output dim 2 * num_kv_heads * head_dim.
    o_proj      : Output projection.
    rotary_emb  : Optional rotary embedding module from the original HF layer.
    """

    # ------------------------------------------------------------------
    # Class-level inference state — set once per forward pass via set_state()
    # ------------------------------------------------------------------
    _block_manager: Optional[BlockManager] = None
    _block_table: Optional[List[int]] = None   # seq_id per batch element
    _seq_lens: Optional[List[int]] = None      # current KV length per batch element
    _is_prefill: bool = False

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        o_proj: nn.Linear,
        rotary_emb: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = q_proj
        self.kv_proj = kv_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_attention(
        cls,
        attn_layer: nn.Module,
        model_config,
        layer_idx: int,
    ) -> "PagedAttention":
        """
        Build a PagedAttention from an existing HF attention layer, copying weights.

        Supports Llama-style layers that expose q_proj / k_proj / v_proj / o_proj
        as nn.Linear and optionally a rotary_emb module.
        """
        num_heads    = model_config.num_attention_heads
        num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
        head_dim     = model_config.hidden_size // num_heads

        # Fuse separate K and V projections into a single linear layer.
        k_proj = attn_layer.k_proj
        v_proj = attn_layer.v_proj
        has_bias = k_proj.bias is not None

        kv_proj = nn.Linear(
            k_proj.in_features,
            2 * num_kv_heads * head_dim,
            bias=has_bias,
            device=k_proj.weight.device,
            dtype=k_proj.weight.dtype,
        )
        kv_proj.weight.data.copy_(torch.cat([k_proj.weight, v_proj.weight], dim=0))
        if has_bias:
            kv_proj.bias.data.copy_(torch.cat([k_proj.bias, v_proj.bias], dim=0))

        return cls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=attn_layer.q_proj,
            kv_proj=kv_proj,
            o_proj=attn_layer.o_proj,
            rotary_emb=getattr(attn_layer, "rotary_emb", None),
        )

    # ------------------------------------------------------------------
    # Global inference state
    # ------------------------------------------------------------------

    @staticmethod
    def set_state(
        model: nn.Module,
        block_manager: BlockManager,
        block_table: List[int],
        seq_lens: List[int],
        is_prefill: bool,
    ) -> None:
        """
        Set inference state on all PagedAttention layers before a forward pass.

        Parameters
        ----------
        model         : HF model that contains PagedAttention layers.
        block_manager : Shared physical KV cache block pool.
        block_table   : Sequence ID for each element in the current batch.
        seq_lens      : Current KV sequence length for each batch element.
        is_prefill    : True for the prompt phase; False for single-token decode.
        """
        PagedAttention._block_manager = block_manager
        PagedAttention._block_table   = block_table
        PagedAttention._seq_lens      = seq_lens
        PagedAttention._is_prefill    = is_prefill

    # ------------------------------------------------------------------
    # Shared sub-operations
    # ------------------------------------------------------------------

    def _project(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Q and fused KV projections.

        Returns
        -------
        q  : [batch, seq_len, num_heads, head_dim]
        kv : [batch, seq_len, 2, num_kv_heads, head_dim]
        """
        bsz, seq_len, _ = hidden_states.shape
        q  = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        kv = self.kv_proj(hidden_states).view(bsz, seq_len, 2, self.num_kv_heads, self.head_dim)
        return q, kv

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to Q and K.

        Expects rotary_emb to follow the HF convention:
            cos, sin = rotary_emb(q, position_ids)
        where cos/sin have shape [batch, 1, seq_len, head_dim].

        No-op if rotary_emb is absent or position_ids is None.
        """
        if self.rotary_emb is None or position_ids is None:
            return q, k

        cos, sin = self.rotary_emb(q, position_ids)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k

    # ------------------------------------------------------------------
    # Placeholder kernels (not yet implemented)
    # ------------------------------------------------------------------

    def _prefill_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention computation over the full prompt. Not yet implemented."""
        pass

    def _prefill_paged_kv_write(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write prefill K/V tensors into paged KV blocks. Not yet implemented."""
        pass

    def _decode_paged_kv_write(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append a single decode-step K/V token into the paged cache. Not yet implemented."""
        pass

    def _decode_paged_flash_attention(
        self,
        seq_id: int,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Paged flash attention over cached K/V blocks during decode. Not yet implemented."""
        pass

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        q, kv   = self._project(hidden_states)
        k, v    = kv.unbind(dim=2)
        q, k    = self._apply_rotary(q, k, position_ids)

        for i, seq_id in enumerate(self._block_table):
            self._prefill_paged_kv_write(seq_id, k[i], v[i])

        return self._prefill_flash_attention(q, k, v)

    def _decode(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        q, kv   = self._project(hidden_states)
        k, v    = kv.unbind(dim=2)
        q, k    = self._apply_rotary(q, k, position_ids)

        attn_outs = []
        for i, seq_id in enumerate(self._block_table):
            self._decode_paged_kv_write(seq_id, k[i], v[i])
            attn_outs.append(self._decode_paged_flash_attention(seq_id, q[i]))

        # attn_outs is a list of per-sequence outputs; stacking is deferred until
        # _decode_paged_flash_attention is implemented and returns real tensors.
        return None

    # ------------------------------------------------------------------
    # Forward — HF-compatible signature
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], None, None]:
        if self._is_prefill:
            attn_out = self._prefill(hidden_states, position_ids)
        else:
            attn_out = self._decode(hidden_states, position_ids)

        # Apply output projection once kernels are implemented and attn_out is real.
        if attn_out is not None:
            bsz, _, seq_len, _ = attn_out.shape
            attn_out = attn_out.transpose(1, 2).contiguous().view(
                bsz, seq_len, self.num_heads * self.head_dim
            )
            attn_out = self.o_proj(attn_out)

        return attn_out, None, None
