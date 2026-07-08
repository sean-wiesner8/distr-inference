"""
Token sampling for the inference engine.

A single naive, per-row sampler shared by every sequence in a step. Sampling
is done one row at a time so heterogeneous ``SamplingParams`` (different
temperature / top-k per request) are trivial to support; vectorizing across
the batch is a later optimization.
"""

from __future__ import annotations

import torch

from .sequence import SamplingParams


def sample_token(logits: torch.Tensor, params: SamplingParams) -> int:
    """
    Sample one token id from a 1-D logits row under ``params``.

    Parameters
    ----------
    logits : [vocab_size] logits for a single position.
    params : Sampling configuration for the owning sequence.

    Returns
    -------
    The sampled token id.
    """
    # temperature == 0 → greedy / argmax (deterministic).
    if params.temperature == 0.0:
        return int(torch.argmax(logits))

    logits = logits.float() / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, logits.numel())
        top_vals, top_idx = torch.topk(logits, k)
        probs = torch.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(top_idx[choice])

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
