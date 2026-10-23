"""Tests for PagedAttention in paged_attention.py."""

import pytest
import torch
import torch.nn as nn

from distr_inference.attention import PagedAttention, _rotate_half
from distr_inference.attention_metadata import build_step_inputs
from distr_inference.block_manager import BlockManager
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.sequence import SamplingParams, Sequence


# ---------------------------------------------------------------------------
# Test dimensions (kept small for speed)
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 32
BLOCK_SIZE = 4

CONFIG = KVBlockConfig(
    num_layers=NUM_LAYERS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    block_size=BLOCK_SIZE,
    dtype=torch.float32,
    device="cpu",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(num_blocks: int = 16) -> BlockManager:
    return BlockManager(num_blocks=num_blocks, config=CONFIG)


def make_paged_attention(layer_idx: int = 0, rotary_emb=None) -> PagedAttention:
    q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
    kv_proj = nn.Linear(HIDDEN_SIZE, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False)
    o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
    return PagedAttention(
        layer_idx=layer_idx,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_proj=q_proj,
        kv_proj=kv_proj,
        o_proj=o_proj,
        rotary_emb=rotary_emb,
    )


class FakeRotaryEmb(nn.Module):
    """Minimal rotary embedding that returns deterministic cos/sin."""

    def forward(self, x, position_ids):
        # x: [1, total_tokens, heads, head_dim]
        # Return cos/sin broadcastable to x's shape.
        seq_len = x.shape[1]
        head_dim = x.shape[-1]
        # Use position-dependent values so rotation is non-trivial.
        positions = position_ids.squeeze(0).float()  # [total_tokens]
        freqs = torch.arange(1, head_dim + 1, device=x.device, dtype=x.dtype)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0) * 0.01  # [T, D]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
        return cos, sin


def _register_and_prefill(bm: BlockManager, seq_id: int, num_tokens: int, layer_idx: int = 0):
    """Register a sequence and pre-populate its blocks in the cache."""
    bm.register_sequence(seq_id)
    block_size = bm.config.block_size
    for t in range(num_tokens):
        logical_idx = t // block_size
        slot_idx = t % block_size
        if slot_idx == 0 and bm.num_blocks_for_sequence(seq_id) <= logical_idx:
            bm.allocate_block(seq_id)
        physical_id = bm.get_block_table(seq_id)[logical_idx]
        k = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        bm.write_slot(physical_id, slot_idx, layer_idx, k, v)


def make_metadata(bm: BlockManager, specs, device="cpu"):
    """
    Build ``(position_ids, attn_metadata)`` for a batch via the real builder.

    ``specs`` is one ``(seq_id, num_cached, num_new)`` per sequence, in batch
    order. Each sequence's blocks must already cover ``num_cached + num_new``.
    """
    batch = []
    for seq_id, num_cached, num_new in specs:
        seq = Sequence(seq_id, [0] * (num_cached + num_new), SamplingParams())
        seq.num_cached_tokens = num_cached
        batch.append(seq)
    _, position_ids, md = build_step_inputs(batch, bm, device)
    return position_ids, md


# ---------------------------------------------------------------------------
# Group 1: _rotate_half
# ---------------------------------------------------------------------------

def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = _rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Group 2: _project
# ---------------------------------------------------------------------------

def test_project_output_shapes():
    attn = make_paged_attention()
    hidden = torch.randn(5, HIDDEN_SIZE)
    q, kv = attn._project(hidden)
    assert q.shape == (5, NUM_HEADS, HEAD_DIM)
    assert kv.shape == (5, 2, NUM_KV_HEADS, HEAD_DIM)


def test_project_single_token():
    attn = make_paged_attention()
    hidden = torch.randn(1, HIDDEN_SIZE)
    q, kv = attn._project(hidden)
    assert q.shape == (1, NUM_HEADS, HEAD_DIM)
    assert kv.shape == (1, 2, NUM_KV_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# Group 3: _apply_rotary
# ---------------------------------------------------------------------------

def test_apply_rotary_noop_when_no_rotary_emb():
    attn = make_paged_attention(rotary_emb=None)
    q = torch.randn(5, NUM_HEADS, HEAD_DIM)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)
    pos = torch.arange(5).unsqueeze(0)
    q_out, k_out = attn._apply_rotary(q, k, pos)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_apply_rotary_modifies_q_and_k():
    rotary = FakeRotaryEmb()
    attn = make_paged_attention(rotary_emb=rotary)
    q = torch.randn(5, NUM_HEADS, HEAD_DIM)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)
    # Use non-zero positions so rotation is non-trivial.
    pos = torch.arange(1, 6).unsqueeze(0)
    q_out, k_out = attn._apply_rotary(q, k, pos)
    assert not torch.equal(q_out, q)
    assert not torch.equal(k_out, k)


def test_apply_rotary_output_shapes():
    rotary = FakeRotaryEmb()
    attn = make_paged_attention(rotary_emb=rotary)
    q = torch.randn(5, NUM_HEADS, HEAD_DIM)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)
    pos = torch.arange(5).unsqueeze(0)
    q_out, k_out = attn._apply_rotary(q, k, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# ---------------------------------------------------------------------------
# Group 4: _write_kv_to_paged_cache
# ---------------------------------------------------------------------------

def _write(attn, bm, k, v, specs):
    """Write ``k``/``v`` for the batch described by ``specs`` (see make_metadata)."""
    _, md = make_metadata(bm, specs)
    k_cache, v_cache = bm.get_kv_cache(attn.layer_idx)
    attn._write_kv_to_paged_cache(k, v, k_cache, v_cache, md.slot_mapping)


def test_write_kv_prefill_single_seq():
    """6 tokens into empty cache with block_size=4 → 2 blocks."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 6)

    num_tokens = 6
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k, v, [(0, 0, num_tokens)])

    # Verify values in cache: first block full (4), second partial (2).
    table = bm.get_block_table(0)
    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    for t in range(num_tokens):
        logical_idx = t // BLOCK_SIZE
        slot_idx = t % BLOCK_SIZE
        pid = table[logical_idx]
        assert torch.allclose(k_cache[pid, slot_idx], k[t])
        assert torch.allclose(v_cache[pid, slot_idx], v[t])


def test_write_kv_single_token_decode():
    """Decode: 1 new token with 3 already cached."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    _register_and_prefill(bm, seq_id=0, num_tokens=3, layer_idx=0)

    k_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    v_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k_new, v_new, [(0, 3, 1)])

    # Still 1 block; the new token lands in its last slot.
    table = bm.get_block_table(0)
    assert bm.num_blocks_for_sequence(0) == 1

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    pid = table[0]
    assert torch.allclose(k_cache[pid, 3], k_new[0])
    assert torch.allclose(v_cache[pid, 3], v_new[0])


def test_write_kv_decode_crosses_block_boundary():
    """Decode with full first block → token lands in slot 0 of the second block."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    _register_and_prefill(bm, seq_id=0, num_tokens=4, layer_idx=0)
    bm.ensure_capacity(0, 5)

    k_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    v_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k_new, v_new, [(0, 4, 1)])

    assert bm.num_blocks_for_sequence(0) == 2
    table = bm.get_block_table(0)

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    pid = table[1]
    assert torch.allclose(k_cache[pid, 0], k_new[0])
    assert torch.allclose(v_cache[pid, 0], v_new[0])


def test_write_kv_mixed_batch():
    """Seq 0 prefills 3 tokens, seq 1 decodes 1 token (5 already cached)."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()

    # Seq 0: fresh prefill.
    bm.register_sequence(0)
    bm.ensure_capacity(0, 3)
    # Seq 1: already has 5 tokens (2 blocks: one full, one with 1 slot).
    _register_and_prefill(bm, seq_id=1, num_tokens=5, layer_idx=0)

    k = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)  # 3 + 1 tokens
    v = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k, v, [(0, 0, 3), (1, 5, 1)])

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)

    # Seq 0: tokens 0-2 in slots 0-2 of its only block.
    table0 = bm.get_block_table(0)
    assert bm.num_blocks_for_sequence(0) == 1
    for t in range(3):
        assert torch.allclose(k_cache[table0[0], t], k[t])
        assert torch.allclose(v_cache[table0[0], t], v[t])

    # Seq 1: still 2 blocks; its new token (position 5) is slot 1 of block 1.
    table1 = bm.get_block_table(1)
    assert bm.num_blocks_for_sequence(1) == 2
    assert torch.allclose(k_cache[table1[1], 1], k[3])
    assert torch.allclose(v_cache[table1[1], 1], v[3])


def test_write_kv_spans_multiple_blocks():
    """9 tokens → 3 blocks (4 + 4 + 1)."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 9)

    num_tokens = 9
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k, v, [(0, 0, num_tokens)])

    assert bm.num_blocks_for_sequence(0) == 3
    table = bm.get_block_table(0)
    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    for t in range(num_tokens):
        pid = table[t // BLOCK_SIZE]
        assert torch.allclose(k_cache[pid, t % BLOCK_SIZE], k[t])
        assert torch.allclose(v_cache[pid, t % BLOCK_SIZE], v[t])


def test_write_kv_only_touches_own_layer():
    """Writing layer 1 leaves layer 0's cache untouched."""
    attn = make_paged_attention(layer_idx=1)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 5)

    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)

    _write(attn, bm, k, v, [(0, 0, 5)])

    k0, v0 = bm.get_kv_cache(layer_idx=0)
    assert torch.all(k0 == 0)
    assert torch.all(v0 == 0)
    k1, _ = bm.get_kv_cache(layer_idx=1)
    assert torch.allclose(k1[bm.get_block_table(0)[1], 0], k[4])


def test_write_kv_accepts_non_contiguous_kv():
    """k/v sliced from the fused [T, 2, H, D] projection are non-contiguous views."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 6)

    kv = torch.randn(6, 2, NUM_KV_HEADS, HEAD_DIM)
    k, v = kv[:, 0], kv[:, 1]
    assert not k.is_contiguous()

    _write(attn, bm, k, v, [(0, 0, 6)])

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    pid = bm.get_block_table(0)[1]
    assert torch.equal(k_cache[pid, 1], k[5])
    assert torch.equal(v_cache[pid, 1], v[5])


def test_write_kv_casts_to_cache_dtype():
    """K/V in a different dtype than the cache are cast on write."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 2)

    k = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float64)
    v = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float64)

    _write(attn, bm, k, v, [(0, 0, 2)])

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    pid = bm.get_block_table(0)[0]
    assert k_cache.dtype == torch.float32
    assert torch.equal(k_cache[pid, 1], k[1].float())
    assert torch.equal(v_cache[pid, 1], v[1].float())


def _reference_write(bm, layer_idx, k, v, cu_seqlens, seq_ids, num_cached):
    """
    The original per-token cache write, kept as the reference the vectorized
    ``index_copy_`` path must match exactly.
    """
    block_size = bm.config.block_size
    for i, sid in enumerate(seq_ids):
        start = cu_seqlens[i]
        num_new = cu_seqlens[i + 1] - start
        block_ids = bm.get_block_ids(sid)
        for t in range(num_new):
            abs_pos = num_cached[i] + t
            physical_id = block_ids[abs_pos // block_size]
            slot_idx = abs_pos % block_size
            bm.k_cache[layer_idx, physical_id, slot_idx] = k[start + t]
            bm.v_cache[layer_idx, physical_id, slot_idx] = v[start + t]


def test_write_kv_matches_reference_loop():
    """Vectorized write is bit-identical to the per-token loop on a random mixed batch."""
    gen = torch.Generator().manual_seed(0)
    layer_idx = 1
    attn = make_paged_attention(layer_idx=layer_idx)

    # (seq_id, num_cached, num_new): prefills, decodes, and chunks that start
    # mid-block and cross block boundaries.
    specs = [(3, 0, 7), (8, 5, 1), (1, 3, 6), (5, 8, 1), (2, 0, 1), (6, 11, 4)]
    target = {sid: cached + new for sid, cached, new in specs}

    bm_ref = make_manager(num_blocks=32)
    bm_new = make_manager(num_blocks=32)
    for bm in (bm_ref, bm_new):
        for sid in target:
            bm.register_sequence(sid)
        # Grow round-robin, one block at a time, so each sequence's physical
        # blocks are interleaved with the other sequences' blocks.
        while any(bm.num_blocks_for_sequence(s) * BLOCK_SIZE < n for s, n in target.items()):
            for sid, n in target.items():
                if bm.num_blocks_for_sequence(sid) * BLOCK_SIZE < n:
                    bm.allocate_block(sid)
    for sid in target:
        assert bm_ref.get_block_ids(sid) == bm_new.get_block_ids(sid)

    # Same random contents in both caches beforehand, so the comparison also
    # checks that slots outside the batch are left alone.
    bm_new.k_cache.copy_(torch.randn(bm_new.k_cache.shape, generator=gen))
    bm_new.v_cache.copy_(torch.randn(bm_new.v_cache.shape, generator=gen))
    bm_ref.k_cache.copy_(bm_new.k_cache)
    bm_ref.v_cache.copy_(bm_new.v_cache)

    total = sum(new for _, _, new in specs)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=gen)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=gen)

    _, md = make_metadata(bm_new, specs)
    k_cache, v_cache = bm_new.get_kv_cache(layer_idx)
    attn._write_kv_to_paged_cache(k, v, k_cache, v_cache, md.slot_mapping)

    _reference_write(
        bm_ref, layer_idx, k, v, md.cu_seqlens_cpu, md.seq_ids, md.num_cached,
    )

    assert torch.equal(bm_new.k_cache, bm_ref.k_cache)
    assert torch.equal(bm_new.v_cache, bm_ref.v_cache)


# ---------------------------------------------------------------------------
# Group 5: _build_block_table_tensor
# ---------------------------------------------------------------------------

def test_build_block_table_single_seq():
    attn = make_paged_attention()
    bm = make_manager()
    bm.register_sequence(0)
    bm.allocate_block(0)
    bm.allocate_block(0)

    table = attn._build_block_table_tensor(bm, seq_ids=[0], seq_lens=[8], device=torch.device("cpu"))

    assert table.shape == (1, 2)
    assert table.dtype == torch.int32

    bt = bm.get_block_table(0)
    assert table[0, 0].item() == bt[0]
    assert table[0, 1].item() == bt[1]


def test_build_block_table_padding():
    attn = make_paged_attention()
    bm = make_manager()

    bm.register_sequence(0)
    bm.allocate_block(0)
    bm.allocate_block(0)
    bm.allocate_block(0)

    bm.register_sequence(1)
    bm.allocate_block(1)

    table = attn._build_block_table_tensor(
        bm, seq_ids=[0, 1], seq_lens=[12, 3], device=torch.device("cpu"),
    )

    assert table.shape == (2, 3)
    # Seq 1 has only 1 block; columns 1 and 2 should be padded with -1.
    assert table[1, 1].item() == -1
    assert table[1, 2].item() == -1


def test_build_block_table_dtype_and_device():
    attn = make_paged_attention()
    bm = make_manager()
    bm.register_sequence(0)
    bm.allocate_block(0)

    table = attn._build_block_table_tensor(bm, seq_ids=[0], seq_lens=[4], device=torch.device("cpu"))

    assert table.dtype == torch.int32
    assert table.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Group 6: forward with mocked attention (CPU, no flash-attn needed)
# ---------------------------------------------------------------------------

def _run_forward_mocked(attn, hidden_states, position_ids, md, bm, captured=None):
    """
    Run forward with _attention monkey-patched to avoid flash-attn. If
    ``captured`` is a dict, the kwargs passed to _attention are stored in it.
    """
    def fake_attention(q, k_cache, v_cache, **kw):
        if captured is not None:
            captured.update(kw)
        return torch.zeros_like(q)

    original = attn._attention
    attn._attention = fake_attention
    try:
        return attn(hidden_states, position_ids, md, bm)
    finally:
        attn._attention = original


def test_forward_output_shape_prefill():
    attn = make_paged_attention()
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 6)

    num_tokens = 6
    hidden = torch.randn(num_tokens, HIDDEN_SIZE)
    pos, md = make_metadata(bm, [(0, 0, num_tokens)])

    out = _run_forward_mocked(attn, hidden, pos, md, bm)

    assert out.shape == (num_tokens, HIDDEN_SIZE)


def test_forward_output_shape_decode():
    attn = make_paged_attention()
    bm = make_manager()
    _register_and_prefill(bm, seq_id=0, num_tokens=5, layer_idx=0)
    bm.ensure_capacity(0, 6)

    hidden = torch.randn(1, HIDDEN_SIZE)
    pos, md = make_metadata(bm, [(0, 5, 1)])

    out = _run_forward_mocked(attn, hidden, pos, md, bm)

    assert out.shape == (1, HIDDEN_SIZE)


def test_forward_writes_to_cache():
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 3)

    num_tokens = 3
    hidden = torch.randn(num_tokens, HIDDEN_SIZE)
    pos, md = make_metadata(bm, [(0, 0, num_tokens)])

    _run_forward_mocked(attn, hidden, pos, md, bm)

    # Cache should have been written to (non-zero values from projections).
    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    table = bm.get_block_table(0)
    pid = table[0]
    # At least slot 0 should be non-zero.
    assert not torch.all(k_cache[pid, 0] == 0)
    assert not torch.all(v_cache[pid, 0] == 0)


def test_forward_mixed_batch():
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()

    # Seq 0: fresh prefill of 4 tokens.
    bm.register_sequence(0)
    bm.ensure_capacity(0, 4)
    # Seq 1: decode 1 token, already has 3 cached.
    _register_and_prefill(bm, seq_id=1, num_tokens=3, layer_idx=0)

    total_tokens = 5  # 4 + 1
    hidden = torch.randn(total_tokens, HIDDEN_SIZE)
    pos, md = make_metadata(bm, [(0, 0, 4), (1, 3, 1)])
    assert pos.tolist() == [0, 1, 2, 3, 3]

    out = _run_forward_mocked(attn, hidden, pos, md, bm)

    assert out.shape == (total_tokens, HIDDEN_SIZE)
    k_cache, _ = bm.get_kv_cache(layer_idx=0)
    # Seq 0: 1 block, last prefill token written to slot 3.
    assert bm.num_blocks_for_sequence(0) == 1
    assert not torch.all(k_cache[bm.get_block_table(0)[0], 3] == 0)
    # Seq 1: still 1 block, decode token written to slot 3.
    assert bm.num_blocks_for_sequence(1) == 1
    assert not torch.all(k_cache[bm.get_block_table(1)[0], 3] == 0)


def test_forward_passes_metadata_through_unchanged():
    """Attention consumes the per-step metadata as-is; nothing is rebuilt per layer."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)
    bm.ensure_capacity(0, 4)
    _register_and_prefill(bm, seq_id=1, num_tokens=3, layer_idx=0)

    hidden = torch.randn(5, HIDDEN_SIZE)
    pos, md = make_metadata(bm, [(0, 0, 4), (1, 3, 1)])

    captured = {}
    _run_forward_mocked(attn, hidden, pos, md, bm, captured=captured)

    assert captured["cu_seqlens_q"] is md.cu_seqlens_q
    assert captured["seqused_k"] is md.seqused_k
    assert captured["block_table"] is md.block_table
    assert captured["max_seqlen_q"] == md.max_seqlen_q
    assert captured["max_seqlen_k"] == md.max_seqlen_k


# ---------------------------------------------------------------------------
# Group 7: GPU + flash-attn tests
# ---------------------------------------------------------------------------

_has_cuda = torch.cuda.is_available()
try:
    from distr_inference.attention import load_flash_attn_varlen_func

    load_flash_attn_varlen_func()
    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False

requires_gpu = pytest.mark.skipif(
    not (_has_cuda and _has_flash_attn),
    reason="Requires CUDA GPU and vLLM's paged flash-attn (vllm-flash-attn or vllm)",
)


@requires_gpu
def test_attention_output_shape_gpu():
    cfg = KVBlockConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=16, dtype=torch.bfloat16, device="cuda",
    )
    bm = BlockManager(num_blocks=16, config=cfg)
    attn = PagedAttention(
        layer_idx=0, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        q_proj=nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        kv_proj=nn.Linear(HIDDEN_SIZE, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        o_proj=nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False).cuda().to(torch.bfloat16),
    ).cuda()

    bm.register_sequence(0)
    bm.ensure_capacity(0, 6)

    num_tokens = 6
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos, md = make_metadata(bm, [(0, 0, num_tokens)], device="cuda")

    out = attn(hidden, pos, md, bm)

    assert out.shape == (num_tokens, HIDDEN_SIZE)
    assert torch.isfinite(out).all()


@requires_gpu
def test_forward_end_to_end_gpu():
    cfg = KVBlockConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=16, dtype=torch.bfloat16, device="cuda",
    )
    bm = BlockManager(num_blocks=16, config=cfg)
    attn = PagedAttention(
        layer_idx=0, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        q_proj=nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        kv_proj=nn.Linear(HIDDEN_SIZE, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        o_proj=nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False).cuda().to(torch.bfloat16),
    ).cuda()

    # Prefill seq 0 with 4 tokens.
    bm.register_sequence(0)
    bm.ensure_capacity(0, 4)
    hidden_pf = torch.randn(4, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos_pf, md_pf = make_metadata(bm, [(0, 0, 4)], device="cuda")
    out_pf = attn(hidden_pf, pos_pf, md_pf, bm)
    assert out_pf.shape == (4, HIDDEN_SIZE)

    # Decode 1 new token.
    bm.ensure_capacity(0, 5)
    hidden_dec = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos_dec, md_dec = make_metadata(bm, [(0, 4, 1)], device="cuda")
    out_dec = attn(hidden_dec, pos_dec, md_dec, bm)
    assert out_dec.shape == (1, HIDDEN_SIZE)
    assert torch.isfinite(out_dec).all()
