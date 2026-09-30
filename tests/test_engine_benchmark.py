"""
End-to-end engine benchmark on the real paged-attention model.

Measures the continuous-batching LLMEngine over Llama-3.2-1B (or an override
via DISTR_INFERENCE_MODEL_ID) in two configurations:

  1. Single request — one prompt generated alone; the latency baseline.
  2. Batched — BATCH_SIZE prompts submitted together and decoded in the same
     forward passes.

For each run we report per-request TTFT / ITL (via ``compute_metrics``),
aggregate throughput, and peak GPU memory. Run with ``-s`` to see the table.

Timing is taken at step granularity: every sequence in a step's batch gets
exactly one token from that step's forward pass, so each of them is stamped
with the step's end time (after a CUDA sync, so the stamp covers the kernel
work rather than just the launch).

The assertions are deliberately loose sanity checks rather than hardware-
specific latency targets: every request must produce its full MAX_TOKENS, and
batching must raise throughput by at least MIN_BATCH_SPEEDUP over the single
request. Decode on a 1B model is memory-bound, so a working batched path
scales close to linearly; falling under the floor means batching has
regressed into something close to serial execution.

Requires CUDA + HF auth for the model, and vLLM's paged flash-attn kernel.
Skipped otherwise.
"""

import os
import time

import pytest
import torch

try:
    from distr_inference.attention import load_flash_attn_varlen_func

    load_flash_attn_varlen_func()
except ImportError:
    pytest.skip(
        "vLLM's paged flash-attn required (vllm-flash-attn or vllm)",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("CUDA required for paged attention", allow_module_level=True)

from transformers import AutoConfig, AutoTokenizer

from distr_inference.block_manager import BlockManager
from distr_inference.config import DEVICE, DTYPE
from distr_inference.engine import LLMEngine, default_sampler
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.metrics import compute_metrics
from distr_inference.model import LlamaModel
from distr_inference.scheduler import SchedulerConfig
from distr_inference.sequence import SamplingParams
from distr_inference.weight_loader import load_llama_weights


MODEL_ID = os.environ.get("DISTR_INFERENCE_MODEL_ID", "meta-llama/Llama-3.2-1B")
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "The largest planet in the solar system is",
    "A group of crows is called a",
    "The speed of light is approximately",
    "Photosynthesis converts sunlight into",
    "The first person to walk on the moon was",
]
BATCH_SIZE = len(PROMPTS)
MAX_TOKENS = 64
BLOCK_SIZE = 16
# 8 requests x ceil((~10 prompt + 64 output) / 16) = 5 blocks each, plus slack.
NUM_BLOCKS = 64

# Batched throughput must be at least this multiple of single-request
# throughput. Well under the near-linear scaling expected at batch 8, so it
# only trips on a real regression, not on GPU-to-GPU variance.
MIN_BATCH_SPEEDUP = 2.0


@pytest.fixture(scope="module")
def hf_cfg():
    return AutoConfig.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def model(hf_cfg):
    m = LlamaModel(hf_cfg)
    load_llama_weights(m, MODEL_ID)
    m.to(DTYPE).to(DEVICE).eval()
    return m


def make_block_manager(hf_cfg):
    head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // hf_cfg.num_attention_heads
    kv_cfg = KVBlockConfig(
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
        device=str(DEVICE),
    )
    return BlockManager(num_blocks=NUM_BLOCKS, config=kv_cfg)


def encode(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# Timed run
# ---------------------------------------------------------------------------

def run_timed(model, hf_cfg, prompt_ids):
    """
    Submit every prompt at t=0 and step the engine to completion.

    Returns ``(per_request_metrics, aggregate)`` where each per-request entry
    is a :func:`compute_metrics` dict and ``aggregate`` holds wall time, total
    generated tokens, and overall throughput.
    """
    stepped = []

    def recording_sampler(seq, logits):
        stepped.append(seq.seq_id)
        return default_sampler(seq, logits)

    engine = LLMEngine.build(
        model,
        make_block_manager(hf_cfg),
        SchedulerConfig(max_num_seqs=BATCH_SIZE, max_num_batched_tokens=2048),
        device=DEVICE,
        sampler=recording_sampler,
    )
    # ignore_eos: every request must run exactly MAX_TOKENS so runs are
    # comparable regardless of what the model emits.
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stamps = {engine.add_request(ids, sp): [] for ids in prompt_ids}

    finished = []
    with torch.no_grad():
        while engine.has_unfinished():
            stepped.clear()
            finished.extend(engine.step())
            torch.cuda.synchronize()
            now = time.perf_counter()
            for sid in stepped:
                stamps[sid].append(now)
    wall = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    assert len(finished) == len(prompt_ids)
    for seq in finished:
        assert seq.num_output_tokens == MAX_TOKENS

    per_request = []
    for sid, ts in stamps.items():
        # compute_metrics expects [TTFT, ITL_1, ITL_2, ...] in seconds.
        token_times = [ts[0] - start] + [b - a for a, b in zip(ts, ts[1:])]
        per_request.append(compute_metrics(
            token_times, ts[-1] - start, len(ts), peak_alloc, peak_reserved,
        ))

    total_tokens = sum(m["num_tokens"] for m in per_request)
    aggregate = {
        "wall_s": wall,
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": total_tokens / wall,
        "peak_memory_allocated_gb": peak_alloc,
        "peak_memory_reserved_gb": peak_reserved,
    }
    return per_request, aggregate


def format_report(name, per_request, aggregate):
    ttft = [m["ttft_ms"] for m in per_request]
    itl = [m["itl_mean_ms"] for m in per_request]
    p99 = [m["itl_p99_ms"] for m in per_request]
    return "\n".join([
        f"{name} ({len(per_request)} request(s) x {MAX_TOKENS} tokens)",
        f"  TTFT mean / max:      {sum(ttft) / len(ttft):8.2f} / {max(ttft):8.2f} ms",
        f"  ITL mean / worst p99: {sum(itl) / len(itl):8.2f} / {max(p99):8.2f} ms",
        f"  Throughput:           {aggregate['throughput_tokens_per_sec']:8.1f} tokens/s",
        f"  Wall time:            {aggregate['wall_s']:8.3f} s",
        f"  Peak mem alloc/resv:  {aggregate['peak_memory_allocated_gb']:8.2f} / "
        f"{aggregate['peak_memory_reserved_gb']:.2f} GB",
    ])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_engine_benchmark(model, hf_cfg, tokenizer):
    prompt_ids = [encode(tokenizer, p) for p in PROMPTS]

    # Warm-up: first-call CUDA context setup, cuBLAS heuristics, and kernel
    # loading would otherwise land in the single-request TTFT.
    run_timed(model, hf_cfg, prompt_ids)

    single, single_agg = run_timed(model, hf_cfg, prompt_ids[:1])
    batched, batched_agg = run_timed(model, hf_cfg, prompt_ids)

    speedup = batched_agg["throughput_tokens_per_sec"] / single_agg["throughput_tokens_per_sec"]
    report = "\n".join([
        "",
        format_report("single", single, single_agg),
        format_report("batched", batched, batched_agg),
        f"batched / single throughput: {speedup:.2f}x",
    ])
    print(report)  # visible with -s

    assert speedup >= MIN_BATCH_SPEEDUP, (
        f"batched throughput only {speedup:.2f}x single "
        f"(floor {MIN_BATCH_SPEEDUP}x):{report}"
    )
