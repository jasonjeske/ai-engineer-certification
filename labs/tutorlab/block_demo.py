"""Inspectable pre-norm block, reusing the course GPT's attention and MLP.

New teaching utilities only: existing tutorlab modules are unchanged.
Inputs are floating-point (batch, tokens, channels); no padding is supported.
"""
import torch
import torch.nn as nn

from tutorlab.gpt import CausalSelfAttention, MLP


def layer_norm_parts(x, eps=1e-5):
    """Normalize the last axis, using population variance and no learned affine.

    Returns normalized values, per-row mean, and per-row variance.
    Keeping the reduced axis makes broadcasting explicit.
    """
    if x.ndim < 1 or x.shape[-1] < 1 or not x.is_floating_point():
        raise ValueError('expected floating-point tensor with nonempty last axis')
    if eps <= 0:
        raise ValueError('eps must be positive')
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized, mean, variance


class InspectableBlock(nn.Module):
    """Same parameter names and equations as gpt.Block, plus a trace method."""

    def __init__(self, cfg):
        super().__init__()
        if min(cfg.n_embd, cfg.n_head, cfg.block_size) < 1 or cfg.n_embd % cfg.n_head:
            raise ValueError('positive width must divide into positive head count; context must be positive')
        self.width, self.context = cfg.n_embd, cfg.block_size
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def trace(self, x):
        if x.ndim != 3 or x.shape[-1] != self.width or min(x.shape[:2]) < 1:
            raise ValueError('expected nonempty (batch, tokens, configured width) input')
        if x.shape[1] > self.context:
            raise ValueError('sequence exceeds configured context')
        norm1 = self.ln1(x)
        attention_update = self.attn(norm1)
        after_attention = x + attention_update
        norm2 = self.ln2(after_attention)
        mlp_update = self.mlp(norm2)
        output = after_attention + mlp_update
        return {'norm1': norm1, 'attention_update': attention_update,
                'after_attention': after_attention, 'norm2': norm2,
                'mlp_update': mlp_update, 'output': output}

    def forward(self, x):
        return self.trace(x)['output']


def scalar_path_gradient(depth, scale, residual):
    """Controlled scalar chain, NOT a measurement of transformer training."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('depth must be a positive integer')
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    y = x
    for _ in range(depth):
        update = scale * y
        y = y + update if residual else update
    y.backward()
    return x.grad.item()
