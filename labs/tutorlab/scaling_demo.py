"""Accounting tools for the exact dense, tied-head tutorlab GPT.

Counts are analytic, not performance predictions. The allocation example is a
constructed dimensionless objective, NOT a fit to any measured model losses.
"""
import math

from .gpt_assembly import parameter_budget


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def validate_config(cfg):
    for name in ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd'):
        _positive_int(name, getattr(cfg, name))
    if cfg.n_embd % cfg.n_head:
        raise ValueError('n_embd must be divisible by n_head')


def parameter_counts(cfg, tied=True):
    """Exact unique parameter count and leading-matrix approximation.

Includes affine LayerNorm, linear biases, learned positions and optional
untied bias-free output head. Head count partitions a fixed embedding width.
"""
    validate_config(cfg)
    c = cfg.n_embd
    parts = parameter_budget(cfg).copy()
    parts['untied_head'] = 0 if tied else cfg.vocab_size * c
    exact = sum(parts.values())
    leading = (cfg.vocab_size * c + cfg.block_size * c
               + 12 * cfg.n_layer * c * c + parts['untied_head'])
    return {'parts': parts, 'exact': exact, 'leading': leading,
            'omitted_bias_and_norm': exact - leading}


def weight_bytes(parameters, bytes_per_value=4):
    """Logical weight storage only; excludes metadata, buffers and runtime."""
    _positive_int('parameters', parameters)
    _positive_int('bytes_per_value', bytes_per_value)
    return parameters * bytes_per_value


def fp32_adam_bytes(parameters):
    """Weights + gradients + two Adam moment arrays, all float32.

Excludes scalar steps, activations, temporary tensors, masks and allocator
state. Not a measured peak, nor a mixed-precision training recipe.
"""
    return 4 * weight_bytes(parameters, 4)


def attention_score_elements(batch, heads, tokens):
    """One explicitly materialized (batch, heads, tokens, tokens) matrix."""
    for name, value in (('batch', batch), ('heads', heads), ('tokens', tokens)):
        _positive_int(name, value)
    return batch * heads * tokens * tokens


def kv_cache_bytes(cfg, batch, tokens, bytes_per_value=2):
    """Hypothetical full multi-head K/V cache, all layers, not GQA/MQA.

This estimates a cached implementation. tutorlab.gpt.generate does not cache.
No allocator overhead or extra capacity; prompt must fit learned positions.
"""
    validate_config(cfg)
    for name, value in (('batch', batch), ('tokens', tokens),
                        ('bytes_per_value', bytes_per_value)):
        _positive_int(name, value)
    if tokens > cfg.block_size:
        raise ValueError('tokens exceed learned position capacity')
    return 2 * cfg.n_layer * batch * tokens * cfg.n_embd * bytes_per_value


def dense_train_flops(parameters, tokens):
    """6*N*D rough dense-training heuristic, not a profiler.

Here N is supplied by the caller and D is processed tokens, including repeats.
Ignoring attention's context term and embedding-lookup details is especially
rough for small models or long contexts. No inference-cost claim is made.
"""
    _positive_int('parameters', parameters)
    _positive_int('tokens', tokens)
    return 6 * parameters * tokens


def illustrative_loss(model_units, data_units):
    """Chosen objective 1.5 + n**(-0.5) + d**(-0.5), NOT observed loss."""
    for value in (model_units, data_units):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('model and data units must be finite and positive')
    return 1.5 + model_units ** -0.5 + data_units ** -0.5
