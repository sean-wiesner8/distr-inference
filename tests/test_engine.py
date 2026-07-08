"""
End-to-end engine loop tests on CPU.

Uses a stub model (no CUDA / flash-attn) that returns deterministic logits, so
the full submission -> step -> completion loop, input packing, sampling,
KV-watermark advancement, and eviction are all exercised without a GPU.
"""

import torch

from distr_inference.block_manager import BlockManager
from distr_inference.engine import LLMEngine
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.scheduler import SchedulerConfig
from distr_inference.sequence import SamplingParams


VOCAB = 16
PEAK_TOKEN = 5

CONFIG = KVBlockConfig(
    num_layers=2,
    num_kv_heads=4,
    head_dim=32,
    block_size=4,
    dtype=torch.float32,
    device="cpu",
)


class StubModel:
    """Returns logits whose argmax is always PEAK_TOKEN; records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, position_ids, cu_seqlens, bm, seq_ids, seq_lens):
        self.calls.append(
            {
                "input_ids": input_ids.tolist(),
                "position_ids": position_ids.tolist(),
                "cu_seqlens": cu_seqlens.tolist(),
                "seq_ids": list(seq_ids),
                "seq_lens": list(seq_lens),
            }
        )
        total_tokens = input_ids.shape[0]
        logits = torch.zeros(total_tokens, VOCAB)
        logits[:, PEAK_TOKEN] = 10.0
        return logits


def make_engine(num_blocks=64, max_num_seqs=8, max_num_batched_tokens=1024):
    bm = BlockManager(num_blocks=num_blocks, config=CONFIG)
    model = StubModel()
    engine = LLMEngine.build(
        model,
        bm,
        SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        device="cpu",
    )
    return engine, model, bm


def greedy(**kw):
    return SamplingParams(temperature=0.0, **kw)


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

def test_single_request_runs_to_completion():
    engine, _, bm = make_engine()
    engine.add_request([1, 2, 3], greedy(max_tokens=4))

    finished = engine.run_to_completion()

    assert len(finished) == 1
    seq = finished[0]
    assert seq.done
    assert seq.output_token_ids == [PEAK_TOKEN] * 4   # greedy on the peaked logits
    assert not engine.has_unfinished()
    assert bm.num_used_blocks == 0                     # all blocks released


def test_first_step_is_prefill_then_decode():
    engine, model, _ = make_engine()
    engine.add_request([1, 2, 3], greedy(max_tokens=3))

    engine.step()   # prefill
    engine.step()   # first decode

    prefill = model.calls[0]
    assert prefill["input_ids"] == [1, 2, 3]           # whole prompt
    assert prefill["position_ids"] == [0, 1, 2]
    assert prefill["cu_seqlens"] == [0, 3]
    assert prefill["seq_lens"] == [0]                  # nothing cached yet

    decode = model.calls[1]
    assert decode["input_ids"] == [PEAK_TOKEN]         # only the new token
    assert decode["position_ids"] == [3]
    assert decode["cu_seqlens"] == [0, 1]
    assert decode["seq_lens"] == [3]                   # prompt now cached


# ---------------------------------------------------------------------------
# Continuous batching across multiple requests
# ---------------------------------------------------------------------------

def test_multiple_requests_all_complete():
    engine, _, bm = make_engine()
    lengths = {}
    for i, mt in enumerate([2, 5, 3]):
        sid = engine.add_request([1, 2], greedy(max_tokens=mt))
        lengths[sid] = mt

    finished = engine.run_to_completion()

    assert len(finished) == 3
    for seq in finished:
        assert seq.num_output_tokens == lengths[seq.seq_id]
        assert seq.done
    assert bm.num_used_blocks == 0


def test_late_arrival_joins_running_batch():
    engine, model, _ = make_engine()
    engine.add_request([1, 2, 3, 4], greedy(max_tokens=6))
    engine.step()   # seq 0 prefill
    engine.step()   # seq 0 decode

    # New request mid-flight; next step should batch decode(0) + prefill(1).
    engine.add_request([7, 8], greedy(max_tokens=2))
    engine.step()

    mixed = model.calls[-1]
    assert mixed["seq_ids"] == [0, 1]
    assert mixed["seq_lens"] == [5, 0]                 # seq0 mid-decode, seq1 fresh
    assert mixed["cu_seqlens"] == [0, 1, 3]            # 1 decode tok + 2 prompt toks

    engine.run_to_completion()
    assert not engine.has_unfinished()


def test_finished_sequences_free_blocks_for_waiting():
    # Tight cache: 2 blocks (block_size 4). Two 4-token prompts fill it; a third
    # can only be admitted after one finishes and frees its block.
    engine, _, bm = make_engine(num_blocks=2)
    engine.add_request([1, 2, 3, 4], greedy(max_tokens=1))   # finishes fast
    engine.add_request([1, 2, 3, 4], greedy(max_tokens=1))
    engine.add_request([1, 2, 3, 4], greedy(max_tokens=1))   # must wait for a block

    finished = engine.run_to_completion()

    assert len(finished) == 3
    assert bm.num_used_blocks == 0
