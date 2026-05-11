"""
Model configuration for the inference engine.

Targets the Llama-3.2-1B checkpoint
(https://huggingface.co/meta-llama/Llama-3.2-1B). Field values are read from
the checkpoint's ``config.json`` via ``transformers.AutoConfig`` rather than
hardcoded, so swapping to another Llama variant is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig


LLAMA_3_2_1B_ID = "meta-llama/Llama-3.2-1B"

# Single-GPU, bf16 policy. Every weight, activation, and KV cache tensor
# lives on this device with this dtype. RMSNorm's internal reduction is the
# only fp32 island.
DTYPE: torch.dtype = torch.bfloat16
DEVICE: torch.device = torch.device("cuda:0")


@dataclass(frozen=True)
class RopeScaling:
    rope_type: str
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_scaling: Optional[RopeScaling] = None

    @classmethod
    def from_hf(cls, model_id_or_path: str) -> "ModelConfig":
        """Build a ModelConfig from a HuggingFace checkpoint's config.json."""
        hf = AutoConfig.from_pretrained(model_id_or_path)

        rope_scaling = None
        if getattr(hf, "rope_scaling", None) is not None:
            rs = hf.rope_scaling
            rope_scaling = RopeScaling(
                rope_type=rs["rope_type"],
                factor=rs["factor"],
                low_freq_factor=rs["low_freq_factor"],
                high_freq_factor=rs["high_freq_factor"],
                original_max_position_embeddings=rs["original_max_position_embeddings"],
            )

        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads

        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            max_position_embeddings=hf.max_position_embeddings,
            tie_word_embeddings=hf.tie_word_embeddings,
            rope_scaling=rope_scaling,
        )


def load_llama_3_2_1b_config() -> ModelConfig:
    return ModelConfig.from_hf(LLAMA_3_2_1B_ID)
