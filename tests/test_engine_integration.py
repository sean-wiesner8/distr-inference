"""
End-to-end engine integration test on the real paged-attention model.

Drives the continuous-batching LLMEngine over Llama-3.2-1B (or an override via
DISTR_INFERENCE_MODEL_ID) with real weights and the flash-attn paged kernel.
Two properties are checked:

  1. Correctness — our full greedy output (prefill + every decode step)
     matches HF greedy generation token-for-token on the same prompt.
  2. Batching invariance — running several prompts *concurrently* (mixed
     prefill/decode, admission, eviction) yields token-for-token identical
     greedy output to running each prompt alone. This is drift-free (it
     compares our engine to itself) and exercises the scheduler end to end.

Requires CUDA + flash-attn + HF auth for the model. Skipped otherwise.
"""

import os

import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for paged attention", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
]
MAX_TOKENS = 10


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


def make_engine(model, hf_cfg):
    head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // hf_cfg.num_attention_heads
    kv_cfg = KVBlockConfig(
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=head_dim,
        block_size=256,  # flash-attn paged kernel requires block_size >= 256 here
        dtype=DTYPE,
        device=str(DEVICE),
    )
    bm = BlockManager(num_blocks=16, config=kv_cfg)
    return LLMEngine.build(
        model,
        bm,
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=2048),
        device=DEVICE,
    )


def encode(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# 1. Full greedy output matches HF reference (prefill + decode)
# ---------------------------------------------------------------------------

def test_greedy_output_matches_hf_reference(model, hf_cfg, tokenizer):
    engine = make_engine(model, hf_cfg)
    prompt_ids = encode(tokenizer, PROMPTS[0])

    engine.add_request(prompt_ids, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    with torch.no_grad():
        finished = engine.run_to_completion()

    assert len(finished) == 1
    seq = finished[0]
    assert seq.num_output_tokens == MAX_TOKENS

    # Reference: HF greedy generation over the same prompt. Comparing the whole
    # continuation (not just the first, prefill-derived token) exercises our
    # decode path — packed single-token steps, KV-cache reads, position offsets.
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, attn_implementation="flash_attention_2",
    ).to(DEVICE).eval()
    ids_2d = torch.tensor([prompt_ids], device=DEVICE)
    with torch.no_grad():
        hf_gen = hf.generate(
            ids_2d, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True,
        )
    hf_out = hf_gen[0, len(prompt_ids):].tolist()

    assert seq.output_token_ids == hf_out, (
        f"greedy output {seq.output_token_ids} != HF reference {hf_out}"
    )


# ---------------------------------------------------------------------------
# 2. Continuous batching is invariant to how requests are grouped
# ---------------------------------------------------------------------------

def test_batched_matches_sequential(model, hf_cfg, tokenizer):
    prompt_ids = [encode(tokenizer, p) for p in PROMPTS]
    sp = lambda: SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Run each prompt alone.
    alone = {}
    for i, ids in enumerate(prompt_ids):
        engine = make_engine(model, hf_cfg)
        engine.add_request(ids, sp())
        with torch.no_grad():
            finished = engine.run_to_completion()
        alone[i] = finished[0].output_token_ids

    # Run all prompts concurrently through one engine.
    engine = make_engine(model, hf_cfg)
    sid_to_idx = {}
    for i, ids in enumerate(prompt_ids):
        sid = engine.add_request(ids, sp())
        sid_to_idx[sid] = i
    with torch.no_grad():
        finished = engine.run_to_completion()

    batched = {sid_to_idx[s.seq_id]: s.output_token_ids for s in finished}

    assert len(batched) == len(PROMPTS)
    for i in range(len(PROMPTS)):
        assert batched[i] == alone[i], (
            f"prompt {i!r}: batched {batched[i]} != sequential {alone[i]}"
        )
