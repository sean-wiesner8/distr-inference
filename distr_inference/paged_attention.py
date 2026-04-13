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

import math
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

    The forward pass accepts HF-style batched inputs [batch, seq_len, hidden]
    and internally converts them to a packed 1-D token buffer
    [total_tokens, hidden] for processing by paged-attention kernels.

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
    _seq_ids: Optional[List[int]] = None       # seq_id per batch element
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
        seq_ids: List[int],
        seq_lens: List[int],
        is_prefill: bool,
    ) -> None:
        """
        Set inference state on all PagedAttention layers before a forward pass.

        Parameters
        ----------
        model         : HF model that contains PagedAttention layers.
        block_manager : Shared physical KV cache block pool.
        seq_ids       : Sequence ID for each element in the current batch.
        seq_lens      : Current KV sequence length for each batch element
                        (number of tokens already cached, *before* this step).
        is_prefill    : True for the prompt phase; False for single-token decode.
        """
        PagedAttention._block_manager = block_manager
        PagedAttention._seq_ids       = seq_ids
        PagedAttention._seq_lens      = seq_lens
        PagedAttention._is_prefill    = is_prefill

    # ------------------------------------------------------------------
    # Batch ↔ 1-D buffer conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_batch(
        hidden_states: torch.Tensor,
        seq_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack a padded [batch, max_seq_len, hidden] tensor into a contiguous
        1-D token buffer [total_tokens, hidden], stripping per-sequence padding.

        Uses a boolean mask + ``torch.masked_select`` to avoid Python loops.

        Parameters
        ----------
        hidden_states : [batch, max_seq_len, hidden]  (may be right-padded).
        seq_lens      : Actual (unpadded) length of each sequence.

        Returns
        -------
        flat       : [total_tokens, hidden]
        cu_seqlens : int32 tensor of length num_seqs + 1 with cumulative
                     token counts.  Sequence *i* occupies
                     flat[cu_seqlens[i] : cu_seqlens[i+1]].
        """
        bsz, max_seq_len, hidden = hidden_states.shape
        device = hidden_states.device

        lens = torch.tensor(seq_lens, device=device, dtype=torch.int32)
        cu_seqlens = torch.zeros(bsz + 1, device=device, dtype=torch.int32)
        torch.cumsum(lens, dim=0, out=cu_seqlens[1:])

        # Build [batch, max_seq_len] boolean mask of valid (non-padding) positions.
        arange = torch.arange(max_seq_len, device=device).unsqueeze(0)   # [1, S], [[0, 1, 2, ...]]
        mask = arange < lens.unsqueeze(1)                                # [B, S], [[True, True, ..., False], [True, True, True, ..., False], ...]

        flat = hidden_states[mask]           # [total_tokens, hidden]
        return flat, cu_seqlens

    @staticmethod
    def _unflatten_batch(
        flat: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Scatter a 1-D token buffer back into a padded [batch, max_seq_len, dim]
        tensor (zero-padded on the right).

        Uses ``torch.arange`` masking to avoid Python loops.

        Parameters
        ----------
        flat        : [total_tokens, dim]
        cu_seqlens  : int32 tensor, length num_seqs + 1.
        max_seq_len : Target second dimension of the output.

        Returns
        -------
        out : [batch, max_seq_len, dim]
        """
        num_seqs = cu_seqlens.shape[0] - 1
        dim = flat.shape[-1]
        device = flat.device

        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1])          # [num_seqs]
        arange = torch.arange(max_seq_len, device=device).unsqueeze(0)
        mask = arange < seq_lens.unsqueeze(1)                   # [num_seqs, S]

        out = flat.new_zeros(num_seqs, max_seq_len, dim)
        out[mask] = flat
        return out

    # ------------------------------------------------------------------
    # Block-table helpers
    # ------------------------------------------------------------------

    def _build_block_table_tensor(
        self,
        seq_ids: List[int],
        seq_lens: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a [num_seqs, max_blocks_per_seq] int32 tensor of physical block
        IDs from the block manager's per-sequence block tables.

        Sequences with fewer blocks than the maximum are right-padded with -1.
        """
        block_size = self._block_manager.config.block_size
        max_blocks = max(
            math.ceil(slen / block_size) for slen in seq_lens
        ) if seq_lens else 0

        table = torch.full(
            (len(seq_ids), max_blocks), -1, dtype=torch.int32, device=device,
        )
        for i, sid in enumerate(seq_ids):
            bt = self._block_manager.get_block_table(sid)
            for logical_idx, physical_id in bt.items():
                table[i, logical_idx] = physical_id
        return table

    # ------------------------------------------------------------------
    # Shared sub-operations
    # ------------------------------------------------------------------

    def _project(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Q and fused KV projections on a 1-D token buffer.

        Parameters
        ----------
        hidden_states : [total_tokens, hidden]

        Returns
        -------
        q  : [total_tokens, num_heads, head_dim]
        kv : [total_tokens, 2, num_kv_heads, head_dim]
        """
        total_tokens = hidden_states.shape[0]
        q  = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        kv = self.kv_proj(hidden_states).view(total_tokens, 2, self.num_kv_heads, self.head_dim)
        return q, kv

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to Q and K on a 1-D token buffer.

        Parameters
        ----------
        q            : [total_tokens, num_heads, head_dim]
        k            : [total_tokens, num_kv_heads, head_dim]
        position_ids : [1, total_tokens]

        No-op if rotary_emb is absent.
        """
        if self.rotary_emb is None:
            return q, k

        # rotary_emb expects [batch, seq, heads, dim] — wrap the flat buffer
        # in a batch-of-1 so cos/sin shapes are broadcast-compatible.
        q_4d = q.unsqueeze(0)   # [1, total_tokens, num_heads, head_dim]
        k_4d = k.unsqueeze(0)   # [1, total_tokens, num_kv_heads, head_dim]

        cos, sin = self.rotary_emb(q_4d, position_ids)
        q_4d = q_4d * cos + _rotate_half(q_4d) * sin
        k_4d = k_4d * cos + _rotate_half(k_4d) * sin

        return q_4d.squeeze(0), k_4d.squeeze(0)

    # ------------------------------------------------------------------
    # KV cache write helpers
    # ------------------------------------------------------------------

    def _write_kv_to_paged_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_ids: List[int],
        seq_lens_before: List[int],
    ) -> None:
        """
        Scatter K/V tokens from the 1-D buffer into the paged KV cache.

        For prefill, each sequence contributes ``seq_lens[i]`` new tokens.
        For decode, each sequence contributes exactly 1 new token.

        Parameters
        ----------
        k              : [total_tokens, num_kv_heads, head_dim]
        v              : [total_tokens, num_kv_heads, head_dim]
        cu_seqlens     : int32 tensor of cumulative token counts (len num_seqs+1).
        seq_ids        : Sequence ID for each batch element.
        seq_lens_before: Number of tokens already in the cache *before* this
                         step, per sequence. Used to compute the correct
                         block/slot offsets for writes.
        """
        bm = self._block_manager
        block_size = bm.config.block_size

        # TODO: Convert to a single fused kernel to reduce kernel launch overhead. Maybe use cache_ops.reshape_and_cache
        for i, sid in enumerate(seq_ids):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            num_new = end - start
            base_pos = seq_lens_before[i]       # token position in the full sequence

            for t in range(num_new):
                abs_pos = base_pos + t
                logical_idx = abs_pos // block_size
                slot_idx = abs_pos % block_size

                # Allocate a new block when we step into an unallocated logical index.
                if slot_idx == 0 and bm.num_blocks_for_sequence(sid) <= logical_idx:
                    bm.allocate_block(sid)

                block = bm.get_block(sid, logical_idx)
                block.write_slot(
                    slot_idx=slot_idx,
                    layer_idx=self.layer_idx,
                    k=k[start + t],
                    v=v[start + t],
                )
                # Update filled count so get_filled_kv reflects written slots.
                if block._num_filled <= slot_idx:
                    block._num_filled = slot_idx + 1

    # ------------------------------------------------------------------
    # Placeholder kernels (not yet implemented)
    # ------------------------------------------------------------------

    def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: List[int],
        max_seqlen: int,
    ) -> torch.Tensor:
        """
        Flash-attention over the full prompt tokens (1-D packed layout).

        Parameters
        ----------
        q           : [total_tokens, num_heads, head_dim]
        k           : [total_tokens, num_kv_heads, head_dim]
        v           : [total_tokens, num_kv_heads, head_dim]
        cu_seqlens  : Cumulative sequence lengths (length num_seqs + 1).
        max_seqlen  : Maximum sequence length in the batch.

        Returns
        -------
        out : [total_tokens, num_heads, head_dim]

        Not yet implemented — returns zeros with the correct shape.
        """
        return torch.zeros_like(q)

    def _decode_attention(
        self,
        q: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Paged attention over cached KV blocks during decode (1-D packed layout).

        Parameters
        ----------
        q           : [num_seqs, num_heads, head_dim]  (one query token per seq)
        block_table : [num_seqs, max_blocks_per_seq]  physical block IDs.
        seq_lens    : Total KV length per sequence (including the new token).

        Returns
        -------
        out : [num_seqs, num_heads, head_dim]

        Not yet implemented — returns zeros with the correct shape.
        """
        return torch.zeros_like(q)

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Prefill phase: process a batch of variable-length prompts.

        1. Flatten the padded [batch, max_seq_len, hidden] input to a packed
           [total_tokens, hidden] buffer.
        2. Project Q / K / V and apply rotary embeddings.
        3. Write K / V into the paged cache.
        4. Run (placeholder) flash attention on the packed buffer.
        5. Unflatten back to [batch, max_seq_len, hidden].
        """
        bsz, max_seq_len, _ = hidden_states.shape
        seq_ids = self._seq_ids
        seq_lens_before = self._seq_lens  # tokens already cached (usually 0 for prefill)

        # For prefill the new token count per sequence is the input length.
        # seq_lens_before tells us the offset; the actual prompt lengths come
        # from the input tensor (capped by max_seq_len, but may be shorter if
        # the batch is padded).  We use position_ids to infer actual lengths
        # when available, otherwise fall back to max_seq_len for all sequences.
        if position_ids is not None:
            # position_ids: [batch, seq_len]. The actual length of sequence i
            # is the number of non-padding positions. A simple heuristic: the
            # last non-zero-stride position + 1, but the most robust approach
            # is max(position_ids[i]) + 1 since positions are 0-based.
            input_lens = [(int(position_ids[i].max().item()) + 1) for i in range(bsz)]
        else:
            input_lens = [max_seq_len] * bsz

        # --- 1. Flatten to 1-D buffer ---
        flat, cu_seqlens = self._flatten_batch(hidden_states, input_lens)

        # --- 2. Build flat position_ids for the 1-D buffer ---
        flat_positions = []
        for i in range(bsz):
            base = seq_lens_before[i]
            flat_positions.append(
                torch.arange(base, base + input_lens[i], device=hidden_states.device)
            )
        flat_position_ids = torch.cat(flat_positions).unsqueeze(0)  # [1, total_tokens]

        # --- 3. Project + rotary ---
        q, kv = self._project(flat)
        k, v = kv[:, 0], kv[:, 1]     # [total_tokens, num_kv_heads, head_dim]
        q, k = self._apply_rotary(q, k, flat_position_ids)

        # --- 4. Write K/V into paged cache ---
        self._write_kv_to_paged_cache(k, v, cu_seqlens, seq_ids, seq_lens_before)

        # --- 5. Attention (placeholder) ---
        attn_out = self._prefill_attention(q, k, v, cu_seqlens, max(input_lens))

        # --- 6. Unflatten back to batched layout ---
        # attn_out: [total_tokens, num_heads, head_dim] → [batch, max_seq_len, num_heads * head_dim]
        attn_flat = attn_out.reshape(attn_out.shape[0], self.num_heads * self.head_dim)
        return self._unflatten_batch(attn_flat, cu_seqlens, max_seq_len)

    def _decode(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode phase: process one new token per sequence.

        1. Flatten the [batch, 1, hidden] input to [num_seqs, hidden].
        2. Project Q / K / V and apply rotary embeddings.
        3. Write the new K / V token into the paged cache.
        4. Run (placeholder) paged attention using the block table.
        5. Reshape output to [batch, 1, hidden].
        """
        bsz = hidden_states.shape[0]
        seq_ids = self._seq_ids
        seq_lens_before = self._seq_lens  # tokens already cached

        # --- 1. Squeeze out the seq_len=1 dimension ---
        flat = hidden_states.squeeze(1)                          # [bsz, hidden]
        cu_seqlens = torch.arange(
            bsz + 1, device=hidden_states.device, dtype=torch.int32,
        )                                                        # [0, 1, 2, ..., bsz]

        # --- 2. Position IDs for the new tokens ---
        if position_ids is not None:
            flat_position_ids = position_ids.reshape(1, -1)  # [1, num_seqs]
        else:
            flat_position_ids = torch.tensor(
                seq_lens_before, device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0)  # [1, num_seqs]

        # --- 3. Project + rotary ---
        q, kv = self._project(flat)
        k, v = kv[:, 0], kv[:, 1]
        q, k = self._apply_rotary(q, k, flat_position_ids)

        # --- 4. Write K/V into paged cache ---
        self._write_kv_to_paged_cache(k, v, cu_seqlens, seq_ids, seq_lens_before)

        # --- 5. Build block table and compute total KV lengths (including new token) ---
        total_seq_lens = [s + 1 for s in seq_lens_before]
        block_table = self._build_block_table_tensor(seq_ids, total_seq_lens, hidden_states.device)

        # --- 6. Attention (placeholder) ---
        attn_out = self._decode_attention(q, block_table, total_seq_lens)

        # --- 7. Reshape to [batch, 1, num_heads * head_dim] ---
        return attn_out.reshape(bsz, 1, self.num_heads * self.head_dim)

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
    ) -> Tuple[torch.Tensor, None, None]:
        if self._is_prefill:
            attn_out = self._prefill(hidden_states, position_ids)
        else:
            attn_out = self._decode(hidden_states, position_ids)

        # attn_out is already [batch, seq_len, num_heads * head_dim]
        attn_out = self.o_proj(attn_out)

        return attn_out, None, None
