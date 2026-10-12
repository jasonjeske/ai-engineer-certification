"""Inspectable CPU decoding for the generation chapter, without changing GPT.

Policy: temperature then exact top-k then top-p on the remaining distribution.
Ties go to the lower vocabulary ID. Zero temperature means explicit greedy.
The generation loop accepts one nonempty, unpadded prompt, not a serving batch.
"""
import math
from dataclasses import dataclass

import torch


def _validate_options(temperature, top_k, top_p):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError('top_k must be None or a positive integer')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')


def probabilities(logits, temperature=1.0, top_k=None, top_p=1.0):
    """Return float64 probabilities for a CPU vector or matrix of logits.

    Negative infinity is a mask; NaN, positive infinity and all-masked rows
    are errors. Inputs are never mutated. Greedy ignores valid filter options.
    Float64 is for a small teaching example, not a production recommendation.
    """
    _validate_options(temperature, top_k, top_p)
    if logits.device.type != 'cpu' or logits.ndim not in (1, 2) or logits.numel() == 0:
        raise ValueError('logits must be a nonempty CPU vector or matrix')
    if not logits.is_floating_point():
        raise ValueError('logits must be floating point')
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError('logits cannot contain NaN or positive infinity')
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError('each row needs at least one finite logit')
    scores = logits.to(torch.float64)
    if temperature == 0:
        result = torch.zeros_like(scores)
        return result.scatter(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
    # Subtract before dividing to keep the maximum at zero, even for tiny T.
    scores = (scores - scores.amax(dim=-1, keepdim=True)) / temperature
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    if top_k is not None:
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, order[..., :min(top_k, scores.size(-1))], True)
        scores = scores.masked_fill(~keep, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        ranked = probs.gather(-1, order)
        cumulative = ranked.cumsum(dim=-1)
        # Keep a candidate when mass BEFORE it is below the threshold.
        # This includes the candidate that reaches/crosses the threshold.
        before = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], dim=-1)
        keep_ranked = before < top_p
        keep = torch.zeros_like(keep_ranked).scatter(-1, order, keep_ranked)
        probs = probs.masked_fill(~keep, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def entropy_bits(probs):
    """Shannon entropy of a normalized probability vector/matrix, in bits."""
    # Replace zero only inside the logarithm: its contribution stays zero.
    safe = torch.where(probs > 0, probs, torch.ones_like(probs))
    return -(probs * safe.log2()).sum(dim=-1)


@dataclass
class GenerationResult:
    ids: torch.Tensor
    context_lengths: list[int]
    stop_reason: str


@torch.inference_mode()
def generate(model, prompt, max_new_tokens, *, temperature=1.0, top_k=None,
             top_p=1.0, seed=1337, eos_id=None):
    """Recompute the context each step, choose the last position, then append.

    model follows tutorlab.gpt.GPT's (logits, loss) interface and cfg fields.
    Requires model.eval(); never updates weights, mode or the input prompt.
    EOS is checked on newly generated IDs, retained in the returned sequence.
    A token budget is still required. No KV cache or multi-sequence padding.
    """
    _validate_options(temperature, top_k, top_p)
    if model.training:
        raise ValueError('call model.eval() before generation')
    if (prompt.device.type != 'cpu' or prompt.dtype != torch.long or prompt.ndim != 2
            or prompt.shape[0] != 1 or prompt.shape[1] == 0):
        raise ValueError('prompt must be a nonempty CPU long tensor of shape (1, time)')
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if ((prompt < 0) | (prompt >= model.cfg.vocab_size)).any():
        raise ValueError('prompt contains an out-of-vocabulary ID')
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < model.cfg.vocab_size):
        raise ValueError('eos_id must be None or a valid vocabulary ID')
    if type(model.cfg.block_size) is not int or model.cfg.block_size < 1:
        raise ValueError('block_size must be positive')
    ids = prompt.clone()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    lengths = []
    for _ in range(max_new_tokens):
        ctx = ids[:, -model.cfg.block_size:]
        logits, _ = model(ctx)
        probs = probabilities(logits[:, -1, :], temperature, top_k, top_p)
        if temperature == 0:
            nxt = probs.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(probs, 1, generator=rng)
        ids = torch.cat([ids, nxt], dim=1)
        lengths.append(ctx.size(1))
        if eos_id is not None and nxt.item() == eos_id:
            return GenerationResult(ids, lengths, 'eos')
    return GenerationResult(ids, lengths, 'length')
