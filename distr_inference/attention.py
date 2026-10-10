"""
Paged attention module for transformer inference with packed token buffers.

Supports mixed prefill/decode batches using packed 1-D tensors where tokens
from multiple sequences (both prompt and generation) are concatenated into
a single contiguous buffer.

Typical usage
-------------
    attn = PagedAttention(
        layer_idx=0,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_proj=q_linear,
        kv_proj=kv_linear,
        o_proj=o_linear,
        rotary_emb=rope_module,
    )

    input_ids, position_ids, attn_metadata = build_step_inputs(batch, block_manager, device)
    out = attn(
        hidden_states,   # [total_tokens, hidden]
        position_ids,    # [total_tokens]
        attn_metadata,   # AttentionMetadata, built once per step
        block_manager,
    )
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention_metadata import AttentionMetadata
from .block_manager import BlockManager


def load_flash_attn_varlen_func():
    """
    Resolve vLLM's paged flash-attn kernel.

    Normally this is the standalone build produced by
    ``scripts/build_vllm_flash_attn.sh`` -- the PyPI distribution of the same
    name is abandoned (2.6.2, pinned to torch 2.4), so a standalone module being
    importable means someone compiled it against this environment's torch.

    Falls back to the copy vendored inside ``vllm``, which exposes the same
    function, so the library still works in an environment that installed full
    vLLM instead of running the build script.

    Imported lazily rather than at module scope so CPU-only tests can import
    this module and monkeypatch the attention call without the package present.
    """
    try:
        from vllm_flash_attn import flash_attn_varlen_func
    except ImportError:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func


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
    Attention layer with a paged KV cache, operating on packed 1-D token
    buffers to support mixed prefill/decode batches.

    Parameters
    ----------
    layer_idx   : Index of this layer within the transformer stack.
    num_heads   : Number of query attention heads.
    num_kv_heads: Number of key/value heads (< num_heads for GQA/MQA).
    head_dim    : Per-head feature dimension.
    q_proj      : Query projection (nn.Linear).
    kv_proj     : Fused key/value projection with output dim 2 * num_kv_heads * head_dim.
    o_proj      : Output projection.
    rotary_emb  : Optional rotary embedding module.
    """

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
    # Block-table helpers
    # ------------------------------------------------------------------

    def _build_block_table_tensor(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        seq_lens: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a [num_seqs, max_blocks_per_seq] int32 tensor of physical block
        IDs from the block manager's per-sequence block tables.

        Sequences with fewer blocks than the maximum are right-padded with -1.
        """
        block_size = block_manager.config.block_size
        max_blocks = max(
            math.ceil(slen / block_size) for slen in seq_lens
        ) if seq_lens else 0

        table = torch.full(
            (len(seq_ids), max_blocks), -1, dtype=torch.int32, device=device,
        )
        for i, sid in enumerate(seq_ids):
            bt = block_manager.get_block_table(sid)
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
        cu_seqlens: List[int],
        block_manager: BlockManager,
        seq_ids: List[int],
        num_cached: List[int],
    ) -> None:
        """
        Scatter K/V tokens from the 1-D buffer into the paged KV cache.

        Each sequence contributes ``cu_seqlens[i+1] - cu_seqlens[i]`` new
        tokens (multiple for prefill, one for decode).

        Parameters
        ----------
        k             : [total_tokens, num_kv_heads, head_dim]
        v             : [total_tokens, num_kv_heads, head_dim]
        cu_seqlens    : Host-side cumulative token counts (len num_seqs+1).
        block_manager : Shared physical KV cache block pool.
        seq_ids       : Sequence ID for each batch element.
        num_cached    : Number of tokens already in the cache *before* this
                        step, per sequence. Used to compute the correct
                        block/slot offsets for writes.

        Every sequence's blocks must already be allocated (the scheduler does
        this before the forward pass); this method never allocates.
        """
        bm = block_manager
        block_size = bm.config.block_size

        # TODO: Convert to a single fused kernel to reduce kernel launch overhead. Maybe use cache_ops.reshape_and_cache
        for i, sid in enumerate(seq_ids):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            num_new = end - start
            base_pos = num_cached[i]            # token position in the full sequence
            block_ids = bm.get_block_ids(sid)

            for t in range(num_new):
                abs_pos = base_pos + t
                logical_idx = abs_pos // block_size
                slot_idx = abs_pos % block_size

                physical_id = block_ids[logical_idx]
                bm.write_slot(
                    physical_block_id=physical_id,
                    slot_idx=slot_idx,
                    layer_idx=self.layer_idx,
                    k=k[start + t],
                    v=v[start + t],
                )

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined attention for mixed prefill/decode batches.

        Uses vllm_flash_attn's flash_attn_varlen_func with a block table to read K/V directly
        from the contiguous paged cache.

        Parameters
        ----------
        q            : [total_tokens, num_heads, head_dim]
        k_cache      : [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache      : [num_blocks, block_size, num_kv_heads, head_dim]
        cu_seqlens_q : int32 tensor of cumulative query token counts (len num_seqs+1).
        seqused_k    : int32 tensor of per-sequence key token counts (len num_seqs).
                       The fork requires this instead of cu_seqlens_k when a
                       block_table is given.
        max_seqlen_q : Maximum query sequence length in the batch.
        max_seqlen_k : Maximum key sequence length in the batch.
        block_table  : [num_seqs, max_blocks_per_seq] physical block IDs.

        Returns
        -------
        out : [total_tokens, num_heads, head_dim]
        """
        # vLLM's fork rather than upstream flash-attn: upstream requires the
        # paged block size to be divisible by 256, the fork only requires 16.
        flash_attn_varlen_func = load_flash_attn_varlen_func()
        return flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_table,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        block_manager: BlockManager,
    ) -> torch.Tensor:
        """
        Forward pass on a packed 1-D token buffer.

        Parameters
        ----------
        hidden_states : [total_tokens, hidden]
            Packed token representations from all sequences in the batch.
        position_ids  : [total_tokens]
            Absolute position of each token within its sequence.
        attn_metadata : Per-step paged-attention inputs (block table, KV
                        lengths, cumulative query lengths), built once by
                        :func:`~distr_inference.attention_metadata.build_step_inputs`
                        and shared unchanged by every layer.
        block_manager : Shared physical KV cache block pool.

        Returns
        -------
        out : [total_tokens, hidden]
        """
        md = attn_metadata

        # --- 1. Project + rotary ---
        q, kv = self._project(hidden_states)
        k, v = kv[:, 0], kv[:, 1]     # [total_tokens, num_kv_heads, head_dim]
        q, k = self._apply_rotary(q, k, position_ids.unsqueeze(0))

        # --- 2. Write K/V into paged cache ---
        self._write_kv_to_paged_cache(
            k, v, md.cu_seqlens_cpu, block_manager, md.seq_ids, md.num_cached,
        )

        # --- 3. Attention ---
        # K/V are read directly from the contiguous paged cache by the kernel.
        k_cache, v_cache = block_manager.get_kv_cache(self.layer_idx)

        attn_out = self._attention(
            q, k_cache, v_cache,
            cu_seqlens_q=md.cu_seqlens_q,
            seqused_k=md.seqused_k,
            max_seqlen_q=md.max_seqlen_q,
            max_seqlen_k=md.max_seqlen_k,
            block_table=md.block_table,
        )

        # --- 4. Output projection ---
        # attn_out: [total_tokens, num_heads, head_dim] → [total_tokens, hidden]
        attn_flat = attn_out.reshape(attn_out.shape[0], self.num_heads * self.head_dim)
        return self.o_proj(attn_flat)
