"""Inspectable multi-head self-attention built from chapter 4's unchanged head.

Equal-width heads, no bias, dropout, padding or KV cache. The explicit matrices
are educational, not a memory-efficient implementation. Inputs are (B, T, C).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tutorlab.attention_demo import SingleHead, attention


class MultiHead(nn.Module):
    """Independent full-input projections, concatenation, then output mixing."""

    def __init__(self, width, num_heads):
        super().__init__()
        if width < 1 or num_heads < 1 or width % num_heads:
            raise ValueError('positive width must be divisible by positive num_heads')
        self.width, self.num_heads = width, num_heads
        self.head_width = width // num_heads
        self.heads = nn.ModuleList([SingleHead(width, self.head_width)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(width, width, bias=False)

    def _validate(self, x):
        if x.ndim != 3 or x.shape[-1] != self.width or min(x.shape[:2]) < 1:
            raise ValueError('expected nonempty (batch, tokens, configured width) input')

    def forward(self, x, causal=True):
        """Return output (B,T,C), weights (B,H,T,T), concatenation (B,T,C)."""
        self._validate(x)
        results = [head(x, causal=causal) for head in self.heads]
        cat = torch.cat([out for out, _ in results], dim=-1)
        weights = torch.stack([weights for _, weights in results], dim=1)
        return self.proj(cat), weights, cat

    def packed_weights(self):
        """Q-all-heads, K-all-heads, V-all-heads in PyTorch Linear storage order.

        torch.cat preserves autograd links to the original independent weights.
        This is an inspectable equivalence demonstration, not a speed benchmark.
        """
        return torch.cat([getattr(head, name).weight
                          for name in ('query', 'key', 'value')
                          for head in self.heads], dim=0)

    def packed_forward(self, x, causal=True):
        """Same parameters and operation, expressed with a batched head axis."""
        self._validate(x)
        batch, tokens, width = x.shape
        q, k, v = F.linear(x, self.packed_weights()).split(width, dim=-1)
        q, k, v = [z.reshape(batch, tokens, self.num_heads, self.head_width)
                   .transpose(1, 2) for z in (q, k, v)]
        out, weights = attention(q, k, v, causal=causal)
        cat = out.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.proj(cat), weights, cat


def worked_example():
    """Two opposite routing patterns from manually chosen weights, NOT training.

    X columns: constant query feature, log-key feature, two value features.
    Both heads see all four input coordinates; selectors simplify arithmetic.
    """
    x = torch.tensor([[[1, 0, 2, 0], [1, math.log(2), 0, 4],
                       [1, math.log(4), 6, 2]]], dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = MultiHead(4, 2).double()
    with torch.no_grad():
        for sign, head in zip((1, -1), model.heads):
            for layer in (head.query, head.key, head.value):
                layer.weight.zero_()
            head.query.weight[0, 0] = 1
            head.key.weight[0, 1] = sign * math.sqrt(2)
            head.value.weight[:, 2:].copy_(torch.eye(2, dtype=x.dtype))
        model.proj.weight.copy_(torch.tensor([[1, 0, 1, 0], [0, 1, 0, -1],
                                              [1, 0, -1, 0], [0, 1, 0, 1]], dtype=x.dtype))
    return x, model
