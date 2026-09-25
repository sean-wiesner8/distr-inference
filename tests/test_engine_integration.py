"""
End-to-end engine integration test on the real paged-attention model.

Drives the continuous-batching LLMEngine over Llama-3.2-1B (or an override via
DISTR_INFERENCE_MODEL_ID) with real weights and the flash-attn paged kernel.
Two properties are checked:

  1. Decode correctness — teacher-forcing both our model and HF along the same
     token sequence, our next-token logits at every decode step match HF's
     within bf16 tolerance (same argmax, bounded logit drift). This is the
     decode analog of the prefill logits test. We compare logits rather than
     asserting token-for-token greedy equality because greedy argmax is a step
     function: two numerically-close-but-not-identical attention kernels (our
     paged flash-attn vs HF's eager materialized-softmax attention) inevitably
     flip a near-tie a few steps in, after which the sequences diverge
     irrecoverably.
     Teacher forcing keeps both models on one sequence so errors stay per-step.
  2. Batching invariance — running several prompts *concurrently* yields the
     same per-step logits as running each prompt alone. Requests are admitted
     staggered so later prefills pack into the same forward pass as earlier
     sequences' decodes, and generation runs long enough to cross a KV block
     boundary. Both runs are teacher-forced along the alone run's greedy path:
     bf16 matmuls are not batch-invariant (cuBLAS picks shape-dependent
     reduction orders), so free-running greedy output can legitimately flip a
     near-tie token under batching, after which nothing more is comparable.

Requires CUDA + HF auth for the model. Skipped otherwise. Needs vLLM's paged
flash-attn kernel; the HF reference runs with attn_implementation="eager"
so the comparison target is plain materialized-softmax attention and the test
carries no upstream flash-attn dependency.
"""

import os

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
# Long enough that every prompt (~6-8 tokens) crosses into a second KV block
# mid-decode, exercising lazy block allocation under batching.
MAX_TOKENS = 20
BLOCK_SIZE = 16

# Per-step logit drift budget between our paged kernel and HF's eager attention,
# matching the prefill test's tolerance (see test_model_integration.py). A touch
# looser because we check every decode step, not just the confident last one.
LOGIT_TOL = 3e-1

# Per-logit drift budget between batched and alone runs of our own engine:
# |batched - alone| <= BATCH_ATOL + BATCH_RTOL * |alone|. Only bf16
# batch-variance noise should show up here, which is a few ULPs — and a bf16
# ULP scales with magnitude (7 mantissa bits: 0.125 for logits in [16, 32)),
# so the budget must be relative. BATCH_RTOL = 2**-6 is two ULPs. A real
# batching bug (cross-sequence KV reads, wrong positions / seq_lens) drifts by
# whole logits.
BATCH_RTOL = 2 ** -6
BATCH_ATOL = 2 ** -4


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
    return BlockManager(num_blocks=16, config=kv_cfg)


def make_engine(model, hf_cfg, sampler):
    return LLMEngine.build(
        model,
        make_block_manager(hf_cfg),
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=2048),
        device=DEVICE,
        sampler=sampler,
    )


def encode(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# 1. Decode-step logits match HF reference (teacher-forced)
# ---------------------------------------------------------------------------

def test_decode_logits_match_hf_reference(model, hf_cfg, tokenizer):
    prompt_ids = encode(tokenizer, PROMPTS[0])
    T = len(prompt_ids)

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager",
    ).to(DEVICE).eval()

    # HF greedy continuation defines the token path both models are teacher-
    # forced along, so neither drifts off onto a different sequence.
    ids_2d = torch.tensor([prompt_ids], device=DEVICE)
    with torch.no_grad():
        ref_tokens = hf.generate(
            ids_2d, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True,
        )[0, T:].tolist()
    num_steps = len(ref_tokens)  # may be < MAX_TOKENS if HF emitted EOS early
    assert num_steps >= 2, "need at least one decode step to exercise the decode path"

    # HF reference logits for each generated token, via one teacher-forced
    # forward over prompt+continuation. logits[p] predicts token p+1, so the
    # generated tokens (positions T .. T+num_steps-1) are predicted by the
    # logits at positions T-1 .. T+num_steps-2.
    full_ids = prompt_ids + ref_tokens
    with torch.no_grad():
        hf_logits = hf(torch.tensor([full_ids], device=DEVICE)).logits[0]  # [len, V]
    hf_step_logits = hf_logits[T - 1 : T - 1 + num_steps]                    # [num_steps, V]

    # Our model, teacher-forced one token at a time through the decode path:
    # a packed prefill, then num_steps-1 single-token decode steps that read
    # back the KV written by earlier steps at ever-larger position offsets.
    bm = make_block_manager(hf_cfg)
    bm.register_sequence(0)

    our_step_logits = []
    with torch.no_grad():
        # Prefill over the prompt → prediction for the first generated token.
        input_ids = torch.tensor(prompt_ids, dtype=torch.int64, device=DEVICE)
        position_ids = torch.arange(T, dtype=torch.int32, device=DEVICE)
        cu = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
        our_step_logits.append(model(input_ids, position_ids, cu, bm, [0], [0])[-1])

        # Decode steps: feed each reference token at its absolute position; the
        # KV it writes is read back by every later step.
        for k in range(num_steps - 1):
            pos = T + k  # absolute position of ref_tokens[k], and #tokens cached
            input_ids = torch.tensor([ref_tokens[k]], dtype=torch.int64, device=DEVICE)
            position_ids = torch.tensor([pos], dtype=torch.int32, device=DEVICE)
            cu = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
            our_step_logits.append(model(input_ids, position_ids, cu, bm, [0], [pos])[-1])

    # Per-step comparison. Because both models are on the same sequence, any
    # difference is per-step bf16 kernel drift, not accumulated divergence.
    #
    # We don't require argmax equality: at a genuine near-tie (e.g. "Paris. It"
    # vs "Paris. The") the two kernels' ~1e-2 logit drift flips which token wins
    # even though both are essentially correct. Instead we require our greedy
    # token to land in HF's top-k and the logits to stay within tolerance — a
    # real decode bug fails both (token far outside top-k, drift over budget).
    TOPK = 5
    for step, (ours, ref) in enumerate(zip(our_step_logits, hf_step_logits)):
        ref_topk = ref.topk(TOPK).indices.tolist()
        assert ours.argmax().item() in ref_topk, (
            f"decode step {step}: greedy token {ours.argmax().item()} "
            f"not in HF top-{TOPK} {ref_topk}"
        )
        max_diff = (ours.float() - ref.float()).abs().max().item()
        assert max_diff < LOGIT_TOL, (
            f"decode step {step}: logit drift {max_diff:.4f} exceeds tolerance {LOGIT_TOL}"
        )


# ---------------------------------------------------------------------------
# 2. Continuous batching is invariant to how requests are grouped
# ---------------------------------------------------------------------------

# Engine step at which each prompt is submitted in the batched run. Staggering
# makes later prefills share a forward pass with earlier sequences' decodes.
ADMIT_STEP = {0: 0, 1: 2, 2: 5}


class BatchRecorder:
    """Wraps the model to record each forward pass's ``seq_lens``."""

    def __init__(self, model):
        self.model = model
        self.seq_lens_per_step = []

    def __call__(self, input_ids, position_ids, cu_seqlens, bm, seq_ids, seq_lens):
        self.seq_lens_per_step.append(list(seq_lens))
        return self.model(input_ids, position_ids, cu_seqlens, bm, seq_ids, seq_lens)


def test_batched_matches_sequential(model, hf_cfg, tokenizer):
    prompt_ids = [encode(tokenizer, p) for p in PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Every prompt must fit in one block at prefill and spill into a second
    # during decode, so lazy block allocation runs mid-batch.
    for ids in prompt_ids:
        assert len(ids) < BLOCK_SIZE < len(ids) + MAX_TOKENS

    # --- Alone: greedy per prompt; its tokens define the reference path. ---
    alone_logits = {}
    alone_tokens = {}
    for i, ids in enumerate(prompt_ids):
        logs = []

        def greedy_recording(seq, logits, logs=logs):
            logs.append(logits.float().cpu())
            return int(logits.argmax())

        engine = make_engine(model, hf_cfg, greedy_recording)
        engine.add_request(ids, sp)
        with torch.no_grad():
            finished = engine.run_to_completion()
        alone_logits[i] = logs
        alone_tokens[i] = finished[0].output_token_ids

    # --- Batched: staggered admission, forced along each alone path. ---
    sid_to_idx = {}
    batched_logits = {i: [] for i in range(len(PROMPTS))}

    def forced_recording(seq, logits):
        i = sid_to_idx[seq.seq_id]
        batched_logits[i].append(logits.float().cpu())
        return alone_tokens[i][seq.num_output_tokens]

    recorder = BatchRecorder(model)
    engine = make_engine(recorder, hf_cfg, forced_recording)
    pending = dict(ADMIT_STEP)
    step = 0
    with torch.no_grad():
        while pending or engine.has_unfinished():
            for i in [i for i, s in pending.items() if s <= step]:
                sid_to_idx[engine.add_request(prompt_ids[i], sp)] = i
                del pending[i]
            engine.step()
            step += 1

    # The stagger must have produced at least one mixed prefill+decode batch.
    mixed = [
        lens for lens in recorder.seq_lens_per_step
        if any(n == 0 for n in lens) and any(n > 0 for n in lens)
    ]
    assert mixed, f"no mixed prefill/decode batch: {recorder.seq_lens_per_step}"

    # --- Per-step logits must agree up to bf16 batch-variance noise. ---
    for i in range(len(PROMPTS)):
        assert len(batched_logits[i]) == len(alone_logits[i]) == MAX_TOKENS
        for k, (b, a) in enumerate(zip(batched_logits[i], alone_logits[i])):
            diff = (b - a).abs()
            excess = diff - (BATCH_ATOL + BATCH_RTOL * a.abs())
            worst = int(excess.argmax())
            assert excess[worst] <= 0, (
                f"prompt {i} step {k} (predicting position {len(prompt_ids[i]) + k}): "
                f"token {worst} logit drift {diff[worst]:.4f} (alone {a[worst]:.4f}, "
                f"batched {b[worst]:.4f}) exceeds "
                f"{BATCH_ATOL} + {BATCH_RTOL} * |alone|"
            )
