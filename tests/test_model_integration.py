"""
End-to-end prefill correctness test.

Loads Llama-3.2-1B (or an override via DISTR_INFERENCE_MODEL_ID) into our
LlamaModel via the weight loader and compares the logits of a packed one-
sequence forward pass against the HF reference model on the same prompt.

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
from distr_inference.kv_cache import KVBlockConfig
from distr_inference.model import LlamaModel
from distr_inference.weight_loader import load_llama_weights


MODEL_ID = os.environ.get("DISTR_INFERENCE_MODEL_ID", "meta-llama/Llama-3.2-1B")
PROMPT = "The capital of France is"


def test_prefill_matches_hf_reference():
    hf_cfg = AutoConfig.from_pretrained(MODEL_ID)

    model = LlamaModel(hf_cfg)
    load_llama_weights(model, MODEL_ID)
    model.to(DTYPE).to(DEVICE).eval()

    head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // hf_cfg.num_attention_heads
    kv_cfg = KVBlockConfig(
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=head_dim,
        block_size=16,
        dtype=DTYPE,
        device=str(DEVICE),
    )
    bm = BlockManager(num_blocks=64, config=kv_cfg)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_2d = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)  # [1, T]
    T = ids_2d.shape[1]

    input_ids   = ids_2d[0]                                                  # [T]
    position_ids = torch.arange(T, device=DEVICE, dtype=torch.int32)         # [T]
    cu_seqlens   = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)    # [2]

    bm.register_sequence(0)
    with torch.no_grad():
        our_logits = model(input_ids, position_ids, cu_seqlens, bm, [0], [0])  # [T, V]

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    with torch.no_grad():
        hf_logits = hf(ids_2d).logits[0]  # [T, V]

    assert our_logits.shape == hf_logits.shape

    last_diff = (our_logits[-1].float() - hf_logits[-1].float()).abs().max().item()
    assert last_diff < 5e-2, f"last-position logit drift {last_diff:.4f} exceeds bf16 tolerance"

    assert our_logits[-1].argmax() == hf_logits[-1].argmax(), (
        "argmax of last-position logits disagrees with HF reference"
    )
