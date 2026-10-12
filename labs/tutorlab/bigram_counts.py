"""Count-based bigrams: separate fitting, smoothing, scoring and sampling.

CPU float64 throughout. Independent sequences do not create cross-boundary
pairs. Callers own tokenization, boundary tokens, train/validation/test splits.
"""
import math

import torch


def _ids(values, vocab_size):
    ids = torch.as_tensor(values)
    if ids.ndim != 1:
        raise ValueError('token IDs must be a one-dimensional sequence')
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('token IDs must be integers')
    ids = ids.to(device='cpu', dtype=torch.long)
    if (ids < 0).any() or (ids >= vocab_size).any():
        raise ValueError('token ID outside vocabulary')
    return ids


def count_pairs(sequences, vocab_size):
    """Count adjacent IDs within each sequence, never across sequence boundaries."""
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')
    counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float64)
    for sequence in sequences:
        ids = _ids(sequence, vocab_size)
        # The flat index i * V + j names cell [i, j]. bincount accumulates repeats.
        flat = ids[:-1] * vocab_size + ids[1:]
        counts += torch.bincount(flat, minlength=vocab_size ** 2).reshape(vocab_size, vocab_size)
    return counts


def probabilities(counts, alpha=1.0):
    """Add alpha to every cell, then normalize each row. Zero rows require alpha > 0."""
    counts = torch.as_tensor(counts, dtype=torch.float64, device='cpu')
    if counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[0] != counts.shape[1]:
        raise ValueError('counts must be a nonempty square matrix')
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    weights = counts + alpha
    totals = weights.sum(dim=1, keepdim=True)
    if (totals == 0).any():
        raise ValueError('unobserved context row needs positive smoothing')
    return weights / totals


def _distribution(probs):
    probs = torch.as_tensor(probs, dtype=torch.float64, device='cpu')
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[0] != probs.shape[1]:
        raise ValueError('probabilities must be a nonempty square matrix')
    if not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError('probabilities must be finite and nonnegative')
    if not torch.allclose(probs.sum(1), torch.ones(probs.shape[0], dtype=torch.float64), atol=1e-10, rtol=1e-10):
        raise ValueError('probability rows must sum to one')
    return probs


def mean_nll(probs, sequences):
    """Mean negative natural log probability per within-sequence transition.

A zero-probability target returns infinity, not an arbitrary clamped score.
The first token of each sequence is given, not predicted.
"""
    probs = _distribution(probs)
    losses = []
    for sequence in sequences:
        ids = _ids(sequence, probs.shape[0])
        if len(ids) > 1:
            losses.append(-probs[ids[:-1], ids[1:]].log())
    if not losses:
        raise ValueError('scoring needs at least one transition')
    return torch.cat(losses).mean().item()


def generate(probs, start_id, new_tokens, seed=7, greedy=False):
    """Return the start token followed by new_tokens draws (or greedy choices)."""
    probs = _distribution(probs)
    start = _ids([start_id], probs.shape[0]).item()
    if not isinstance(new_tokens, int) or new_tokens < 0:
        raise ValueError('new_tokens must be a nonnegative integer')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    out = [start]
    for _ in range(new_tokens):
        row = probs[out[-1]]
        nxt = row.argmax().item() if greedy else torch.multinomial(row, 1, generator=rng).item()
        out.append(nxt)
    return out
