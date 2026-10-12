"""Inspectable assembly of tutorlab.gpt; reuses its modules without changing them.

trace_forward returns live tensors, so gradients work. In training mode dropout
is stochastic; use eval() when comparing separate forward calls.
"""
import torch


def trace_forward(model, idx):
    """Return ordered intermediate tensors for a nonempty (batch, time) ID tensor."""
    if idx.ndim != 2 or idx.shape[0] < 1 or idx.shape[1] < 1:
        raise ValueError('idx must be a nonempty (batch, time) tensor')
    if idx.shape[1] > model.cfg.block_size:
        raise ValueError('sequence longer than block_size')
    pos = torch.arange(idx.shape[1], device=idx.device)
    trace = {'tokens': model.tok_emb(idx), 'positions': model.pos_emb(pos)}
    x = model.drop(trace['tokens'] + trace['positions'])
    trace['residual_in'] = x
    for i, block in enumerate(model.blocks):
        x = block(x)
        trace[f'block_{i}'] = x
    trace['normalized'] = model.ln_f(x)
    trace['logits'] = model.head(trace['normalized'])
    return trace


def parameter_budget(cfg):
    """Unique trainable parameter counts for the exact tied-head tutorlab GPT.

Includes linear biases and affine LayerNorm; excludes masks (buffers).
Not a generic transformer estimator.
"""
    c = cfg.n_embd
    per_block = 12 * c * c + 13 * c
    return {
        'token_table_and_tied_head': cfg.vocab_size * c,
        'position_table': cfg.block_size * c,
        'blocks': cfg.n_layer * per_block,
        'final_norm': 2 * c,
    }
