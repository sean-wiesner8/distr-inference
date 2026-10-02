"""
Profile LLMEngine decode steps on the real model with torch.profiler.

Answers "where does a decode step's time go?" before optimizing it:

  * wall time per step vs. GPU kernel time per step — the gap is time the
    GPU sits idle waiting on the host (Python, launches, syncs),
  * kernel launches per step,
  * host<->device syncs and copies per step,
  * the top ops by host (CPU) time and by GPU time.

Usage (from the repo root, on a CUDA machine with the paged flash-attn kernel):

    python scripts/profile_decode_step.py                     # batch 8, short prompts
    python scripts/profile_decode_step.py --batch-size 1
    python scripts/profile_decode_step.py --context 2000      # decode against a long context
    python scripts/profile_decode_step.py --stack             # attribute host time to source lines
    python scripts/profile_decode_step.py --trace decode.json # open in https://ui.perfetto.dev

The prompts are prefilled in one step, then WARMUP decode steps run untimed,
then --steps decode steps are timed without the profiler (so profiler
overhead doesn't inflate wall time), then --steps more run under the profiler.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoTokenizer

from distr_inference.block_manager import BlockManager
from distr_inference.config import DEVICE, DTYPE
from distr_inference.engine import LLMEngine
from distr_inference.kv_cache import KVBlockConfig
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
BLOCK_SIZE = 16
WARMUP = 10

# Host-side profiler events that mean "the CPU waited on the GPU" or "data
# crossed the PCIe bus". Names are CUDA runtime/driver API calls.
SYNC_EVENTS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
COPY_EVENTS = ("cudaMemcpyAsync", "cudaMemcpy")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=0,
                   help="prompt length in tokens; 0 = the short natural prompts (~8 tokens)")
    p.add_argument("--steps", type=int, default=20, help="decode steps to time and to profile")
    p.add_argument("--stack", action="store_true",
                   help="record Python stacks and group host time by source location")
    p.add_argument("--trace", help="write a Chrome/Perfetto trace to this path")
    p.add_argument("--rows", type=int, default=25, help="rows per op table")
    return p.parse_args()


def build_prompts(tokenizer, batch_size, context):
    def encode(text):
        return tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    if context == 0:
        return [encode(PROMPTS[i % len(PROMPTS)]) for i in range(batch_size)]

    # Fixed-length prompts: repeat the natural prompts and truncate the ids.
    text = " ".join(PROMPTS)
    ids = encode(text)
    while len(ids) < context:
        text = text + " " + text
        ids = encode(text)
    return [ids[:context] for _ in range(batch_size)]


def build_engine(model, hf_cfg, prompt_ids, max_tokens):
    head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // hf_cfg.num_attention_heads
    kv_cfg = KVBlockConfig(
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
        device=str(DEVICE),
    )
    blocks = sum(math.ceil((len(ids) + max_tokens) / BLOCK_SIZE) for ids in prompt_ids)
    total_prompt = sum(len(ids) for ids in prompt_ids)
    return LLMEngine.build(
        model,
        BlockManager(num_blocks=blocks + 8, config=kv_cfg),
        # Budget covers every prompt, so all prefills land in the first step
        # and every later step is pure decode.
        SchedulerConfig(max_num_seqs=len(prompt_ids), max_num_batched_tokens=total_prompt),
        device=DEVICE,
    )


def device_time_us(evt):
    # Renamed from self_cuda_time_total in torch 2.4.
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)


def summarize(prof, steps, wall_ms_per_step):
    events = prof.key_averages()

    gpu_us = sum(device_time_us(e) for e in events if e.device_type == DeviceType.CUDA)
    launches = sum(e.count for e in events if "LaunchKernel" in e.key)
    syncs = sum(e.count for e in events if e.key in SYNC_EVENTS)
    copies = sum(e.count for e in events if e.key in COPY_EVENTS)
    scalar_reads = sum(e.count for e in events if e.key == "aten::_local_scalar_dense")

    gpu_ms = gpu_us / 1000 / steps
    busy = min(gpu_ms / wall_ms_per_step, 1.0) if wall_ms_per_step else 0.0
    print("\nPer decode step")
    print(f"  wall time (unprofiled):       {wall_ms_per_step:8.2f} ms")
    print(f"  GPU kernel time:              {gpu_ms:8.2f} ms   (GPU busy {busy:.0%} of the step)")
    print(f"  kernel launches:              {launches / steps:8.1f}")
    print(f"  host<-device syncs:           {syncs / steps:8.1f}")
    print(f"  host<->device memcpys:        {copies / steps:8.1f}")
    print(f"  scalar reads (.item()/int()): {scalar_reads / steps:8.1f}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    hf_cfg = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LlamaModel(hf_cfg)
    load_llama_weights(model, MODEL_ID)
    model.to(DTYPE).to(DEVICE).eval()

    prompt_ids = build_prompts(tokenizer, args.batch_size, args.context)
    # One prefill step, then warm-up, timed, and profiled decode steps, plus
    # slack so no sequence finishes (and shrinks the batch) mid-measurement.
    max_tokens = 1 + WARMUP + 2 * args.steps + 1
    engine = build_engine(model, hf_cfg, prompt_ids, max_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for ids in prompt_ids:
        engine.add_request(ids, sp)

    print(f"model {MODEL_ID}  batch {args.batch_size}  "
          f"prompt tokens {max(map(len, prompt_ids))}  {torch.cuda.get_device_name()}")

    with torch.no_grad():
        engine.step()  # prefill
        for _ in range(WARMUP):
            engine.step()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            engine.step()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000 / args.steps

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=args.stack,
        ) as prof:
            for _ in range(args.steps):
                with record_function("decode_step"):
                    engine.step()
            torch.cuda.synchronize()

    summarize(prof, args.steps, wall_ms)

    print(f"\nTop ops by host (CPU) self time, totals over {args.steps} steps")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.rows))
    print(f"\nTop ops by GPU self time, totals over {args.steps} steps")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=args.rows))
    if args.stack:
        print("\nHost self time grouped by Python stack (top 5 frames)")
        print(prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=args.rows,
        ))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\ntrace written to {args.trace} (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
