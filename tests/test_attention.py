"""Tests for PagedAttention in paged_attention.py."""

import pytest
import torch
import torch.nn as nn

from distr_inference.attention import PagedAttention, _rotate_half
from distr_inference.block_manager import BlockManager
from distr_inference.kv_cache import KVBlockConfig


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
        if bm.get_num_filled(physical_id) <= slot_idx:
            bm.set_num_filled(physical_id, slot_idx + 1)


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

def test_write_kv_prefill_single_seq():
    """6 tokens into empty cache with block_size=4 → 2 blocks."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)

    num_tokens = 6
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)

    attn._write_kv_to_paged_cache(k, v, cu_seqlens, bm, seq_ids=[0], seq_lens_before=[0])

    # 2 blocks allocated: first full (4), second partial (2).
    assert bm.num_blocks_for_sequence(0) == 2
    table = bm.get_block_table(0)
    assert bm.get_num_filled(table[0]) == BLOCK_SIZE
    assert bm.get_num_filled(table[1]) == 2

    # Verify values in cache.
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
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    attn._write_kv_to_paged_cache(k_new, v_new, cu_seqlens, bm, seq_ids=[0], seq_lens_before=[3])

    # Still 1 block, now with 4 filled slots.
    table = bm.get_block_table(0)
    assert bm.num_blocks_for_sequence(0) == 1
    assert bm.get_num_filled(table[0]) == 4

    k_cache, v_cache = bm.get_kv_cache(layer_idx=0)
    pid = table[0]
    assert torch.allclose(k_cache[pid, 3], k_new[0])
    assert torch.allclose(v_cache[pid, 3], v_new[0])


def test_write_kv_decode_crosses_block_boundary():
    """Decode with full first block → new block allocated."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    _register_and_prefill(bm, seq_id=0, num_tokens=4, layer_idx=0)

    assert bm.num_blocks_for_sequence(0) == 1

    k_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    v_new = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    attn._write_kv_to_paged_cache(k_new, v_new, cu_seqlens, bm, seq_ids=[0], seq_lens_before=[4])

    assert bm.num_blocks_for_sequence(0) == 2
    table = bm.get_block_table(0)
    assert bm.get_num_filled(table[1]) == 1

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
    # Seq 1: already has 5 tokens (2 blocks: one full, one with 1 slot).
    _register_and_prefill(bm, seq_id=1, num_tokens=5, layer_idx=0)

    k = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)  # 3 + 1 tokens
    v = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, 3, 4], dtype=torch.int32)

    attn._write_kv_to_paged_cache(
        k, v, cu_seqlens, bm,
        seq_ids=[0, 1],
        seq_lens_before=[0, 5],
    )

    # Seq 0: 1 block with 3 filled.
    table0 = bm.get_block_table(0)
    assert bm.num_blocks_for_sequence(0) == 1
    assert bm.get_num_filled(table0[0]) == 3

    # Seq 1: still 2 blocks, second block now has 2 filled.
    table1 = bm.get_block_table(1)
    assert bm.num_blocks_for_sequence(1) == 2
    assert bm.get_num_filled(table1[1]) == 2


def test_write_kv_spans_multiple_blocks():
    """9 tokens → 3 blocks (4 + 4 + 1)."""
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)

    num_tokens = 9
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)

    attn._write_kv_to_paged_cache(k, v, cu_seqlens, bm, seq_ids=[0], seq_lens_before=[0])

    assert bm.num_blocks_for_sequence(0) == 3
    table = bm.get_block_table(0)
    assert bm.get_num_filled(table[0]) == BLOCK_SIZE
    assert bm.get_num_filled(table[1]) == BLOCK_SIZE
    assert bm.get_num_filled(table[2]) == 1


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

def _mock_flash_attn(q, k, v, **kwargs):
    """Mock that returns zeros shaped like q."""
    return torch.zeros_like(q)


def _run_forward_mocked(attn, hidden_states, position_ids, cu_seqlens, bm, seq_ids, seq_lens):
    """Run forward with _attention monkey-patched to avoid flash-attn."""
    original = attn._attention
    attn._attention = lambda q, k_cache, v_cache, **kw: torch.zeros_like(q)
    try:
        return attn(hidden_states, position_ids, cu_seqlens, bm, seq_ids, seq_lens)
    finally:
        attn._attention = original


def test_forward_output_shape_prefill():
    attn = make_paged_attention()
    bm = make_manager()
    bm.register_sequence(0)

    num_tokens = 6
    hidden = torch.randn(num_tokens, HIDDEN_SIZE)
    pos = torch.arange(num_tokens)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)

    out = _run_forward_mocked(attn, hidden, pos, cu_seqlens, bm, seq_ids=[0], seq_lens=[0])

    assert out.shape == (num_tokens, HIDDEN_SIZE)


def test_forward_output_shape_decode():
    attn = make_paged_attention()
    bm = make_manager()
    _register_and_prefill(bm, seq_id=0, num_tokens=5, layer_idx=0)

    hidden = torch.randn(1, HIDDEN_SIZE)
    pos = torch.tensor([5])
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    out = _run_forward_mocked(attn, hidden, pos, cu_seqlens, bm, seq_ids=[0], seq_lens=[5])

    assert out.shape == (1, HIDDEN_SIZE)


def test_forward_writes_to_cache():
    attn = make_paged_attention(layer_idx=0)
    bm = make_manager()
    bm.register_sequence(0)

    num_tokens = 3
    hidden = torch.randn(num_tokens, HIDDEN_SIZE)
    pos = torch.arange(num_tokens)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)

    _run_forward_mocked(attn, hidden, pos, cu_seqlens, bm, seq_ids=[0], seq_lens=[0])

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
    # Seq 1: decode 1 token, already has 3 cached.
    _register_and_prefill(bm, seq_id=1, num_tokens=3, layer_idx=0)

    total_tokens = 5  # 4 + 1
    hidden = torch.randn(total_tokens, HIDDEN_SIZE)
    pos = torch.cat([torch.arange(4), torch.tensor([3])])
    cu_seqlens = torch.tensor([0, 4, 5], dtype=torch.int32)

    out = _run_forward_mocked(
        attn, hidden, pos, cu_seqlens, bm,
        seq_ids=[0, 1],
        seq_lens=[0, 3],
    )

    assert out.shape == (total_tokens, HIDDEN_SIZE)
    # Seq 0 should now have 1 block with 4 filled.
    assert bm.num_blocks_for_sequence(0) == 1
    assert bm.get_num_filled(bm.get_block_table(0)[0]) == BLOCK_SIZE
    # Seq 1 should still have 1 block, now with 4 filled.
    assert bm.num_blocks_for_sequence(1) == 1
    assert bm.get_num_filled(bm.get_block_table(1)[0]) == BLOCK_SIZE


# ---------------------------------------------------------------------------
# Group 7: GPU + flash-attn tests
# ---------------------------------------------------------------------------

_has_cuda = torch.cuda.is_available()
try:
    from flash_attn import flash_attn_varlen_func as _real_fn  # noqa: F401
    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False

requires_gpu = pytest.mark.skipif(
    not (_has_cuda and _has_flash_attn),
    reason="Requires CUDA GPU and flash-attn",
)


@requires_gpu
def test_attention_output_shape_gpu():
    cfg = KVBlockConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=256, dtype=torch.bfloat16, device="cuda",
    )
    bm = BlockManager(num_blocks=16, config=cfg)
    attn = PagedAttention(
        layer_idx=0, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        q_proj=nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        kv_proj=nn.Linear(HIDDEN_SIZE, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False).cuda().to(torch.bfloat16),
        o_proj=nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False).cuda().to(torch.bfloat16),
    ).cuda()

    bm.register_sequence(0)

    num_tokens = 6
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(num_tokens, device="cuda")
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")

    out = attn(hidden, pos, cu_seqlens, bm, seq_ids=[0], seq_lens=[0])

    assert out.shape == (num_tokens, HIDDEN_SIZE)
    assert torch.isfinite(out).all()


@requires_gpu
def test_forward_end_to_end_gpu():
    cfg = KVBlockConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=256, dtype=torch.bfloat16, device="cuda",
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
    hidden_pf = torch.randn(4, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos_pf = torch.arange(4, device="cuda")
    cu_pf = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    out_pf = attn(hidden_pf, pos_pf, cu_pf, bm, seq_ids=[0], seq_lens=[0])
    assert out_pf.shape == (4, HIDDEN_SIZE)

    # Decode 1 new token.
    hidden_dec = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    pos_dec = torch.tensor([4], device="cuda")
    cu_dec = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    out_dec = attn(hidden_dec, pos_dec, cu_dec, bm, seq_ids=[0], seq_lens=[4])
    assert out_dec.shape == (1, HIDDEN_SIZE)
    assert torch.isfinite(out_dec).all()
