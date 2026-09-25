"""
Load Llama weights from a HuggingFace checkpoint into our :class:`LlamaModel`.

Single-GPU, bf16 only. Mutates the model in place. Kept separate from
``model.py`` so the model definition stays pure structure and the loader can
be swapped (sharded checkpoints, quantization, different model families)
without touching the forward pass.

Key layout assumed (HF Llama):
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.mlp.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight
    model.norm.weight
    lm_head.weight        (omitted when tie_word_embeddings=True)
"""

from __future__ import annotations

import warnings
from typing import Dict, Set

import torch
from transformers import AutoModelForCausalLM

from .model import LlamaModel


@torch.no_grad()
def load_llama_weights(model: LlamaModel, model_id_or_path: str) -> None:
    """
    Copy weights from a HF Llama checkpoint into ``model`` in place.

    Loads the HF model on CPU in bf16, then copies tensor-by-tensor. The
    caller is responsible for moving ``model`` to the target device after
    this returns.
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    state_dict: Dict[str, torch.Tensor] = hf_model.state_dict()

    hf_cfg = hf_model.config
    cfg = model.hf_config
    for field in (
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
    ):
        if getattr(hf_cfg, field) != getattr(cfg, field):
            raise ValueError(
                f"Config mismatch on {field}: "
                f"checkpoint={getattr(hf_cfg, field)} model={getattr(cfg, field)}"
            )

    consumed: Set[str] = set()

    def copy(dst: torch.Tensor, key: str) -> None:
        src = state_dict[key]
        if dst.shape != src.shape:
            raise ValueError(
                f"Shape mismatch at {key}: dst={tuple(dst.shape)} src={tuple(src.shape)}"
            )
        dst.copy_(src)
        consumed.add(key)

    copy(model.embed_tokens.weight.data, "model.embed_tokens.weight")

    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"

        copy(layer.input_layernorm.inner.weight.data,          f"{prefix}.input_layernorm.weight")
        copy(layer.post_attention_layernorm.inner.weight.data, f"{prefix}.post_attention_layernorm.weight")

        copy(layer.self_attn.q_proj.weight.data, f"{prefix}.self_attn.q_proj.weight")
        copy(layer.self_attn.o_proj.weight.data, f"{prefix}.self_attn.o_proj.weight")

        k_key = f"{prefix}.self_attn.k_proj.weight"
        v_key = f"{prefix}.self_attn.v_proj.weight"
        fused_kv = torch.cat([state_dict[k_key], state_dict[v_key]], dim=0).contiguous()
        if layer.self_attn.kv_proj.weight.shape != fused_kv.shape:
            raise ValueError(
                f"Shape mismatch on fused KV at layer {i}: "
                f"dst={tuple(layer.self_attn.kv_proj.weight.shape)} src={tuple(fused_kv.shape)}"
            )
        layer.self_attn.kv_proj.weight.data.copy_(fused_kv)
        consumed.add(k_key)
        consumed.add(v_key)

        copy(layer.mlp.inner.gate_proj.weight.data, f"{prefix}.mlp.gate_proj.weight")
        copy(layer.mlp.inner.up_proj.weight.data,   f"{prefix}.mlp.up_proj.weight")
        copy(layer.mlp.inner.down_proj.weight.data, f"{prefix}.mlp.down_proj.weight")

    copy(model.norm.inner.weight.data, "model.norm.weight")

    if not cfg.tie_word_embeddings:
        copy(model.lm_head.weight.data, "lm_head.weight")
    elif "lm_head.weight" in state_dict:
        consumed.add("lm_head.weight")

    leftover = set(state_dict.keys()) - consumed
    if leftover:
        warnings.warn(
            f"Checkpoint keys not consumed by loader: {sorted(leftover)}",
            stacklevel=2,
        )

    del state_dict
    del hf_model
