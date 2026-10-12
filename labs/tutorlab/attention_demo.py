"""Inspectable single-head self-attention, without changing the shared GPT.

The explicit attention matrix is for teaching, not a memory-efficient kernel.
Only square self-attention is supported; no padding, dropout, or KV cache.
"""
import math

import torch
import torch.nn as nn


def attention(q, k, v, causal=True):
    """Return (output, weights) for (..., T, width) floating tensors.

    Q/K widths agree; V can have a different width. Leading dimensions must
    agree exactly, deliberately avoiding accidental batch broadcasting.
    """
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError('expected (..., tokens, features) tensors')
    if q.shape != k.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError('Q/K shapes and Q/V token and batch dimensions must agree')
    if min(q.shape[-2:]) < 1 or v.shape[-1] < 1:
        raise ValueError('tokens and feature widths must be nonempty')
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        allowed = torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class SingleHead(nn.Module):
    """Learnable token-wise projections followed by one attention head."""

    def __init__(self, input_width, head_width, value_width=None):
        super().__init__()
        value_width = head_width if value_width is None else value_width
        if min(input_width, head_width, value_width) < 1:
            raise ValueError('feature widths must be positive')
        self.query = nn.Linear(input_width, head_width, bias=False)
        self.key = nn.Linear(input_width, head_width, bias=False)
        self.value = nn.Linear(input_width, value_width, bias=False)

    def forward(self, x, causal=True):
        return attention(self.query(x), self.key(x), self.value(x), causal=causal)


def worked_example():
    """Hand-chosen features and selector projections, NOT a trained language model.

    Columns hold Q features, K features, V features. Scaled scores in every
    unmasked row are log(1), log(2), log(4), so all results are hand-checkable.
    """
    x = torch.tensor([[1, 0, 0, 0, 2, 0],
                      [1, 0, math.sqrt(2) * math.log(2), 0, 0, 4],
                      [1, 0, math.sqrt(2) * math.log(4), 0, 6, 2]], dtype=torch.float64)
    # fork_rng keeps this deterministic fixture from changing callers' RNG state.
    with torch.random.fork_rng():
        torch.manual_seed(4)
        head = SingleHead(6, 2).double()
    with torch.no_grad():
        for projection, start in ((head.query, 0), (head.key, 2), (head.value, 4)):
            projection.weight.zero_()
            projection.weight[:, start:start + 2].copy_(torch.eye(2, dtype=x.dtype))
    return x, head
