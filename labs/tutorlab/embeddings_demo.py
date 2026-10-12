"""Small embedding experiments for chapter 2, not a pretrained language model.

Cosine deliberately rejects zero vectors: a vector without direction has no
mathematical cosine. PyTorch's epsilon convention is demonstrated separately.
The training experiment fits supplied context distributions, not natural text.
"""
import torch
import torch.nn as nn


def cosine_matrix(rows):
    """Pairwise cosines for a nonempty floating (items, dimensions) tensor.

    Normalize each row, then take every pair's dot product. Reject invalid
    values rather than returning a convincing but meaningless search score.
    """
    if rows.ndim != 2 or 0 in rows.shape or not rows.is_floating_point():
        raise ValueError('rows must be a nonempty floating matrix')
    if not torch.isfinite(rows).all():
        raise ValueError('rows must be finite')
    lengths = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    if not torch.isfinite(lengths).all() or (lengths == 0).any():
        raise ValueError('each row must have a finite, nonzero length')
    unit = rows / lengths
    return unit @ unit.T


def nearest_other(rows, index):
    """Return (row index, cosine), excluding the query itself.

    Ties follow torch.argmax: the first maximum wins, not a semantic preference.
    """
    if rows.ndim != 2 or len(rows) < 2 or not 0 <= index < len(rows):
        raise ValueError('need at least two rows and a valid query index')
    scores = cosine_matrix(rows)[index].clone()
    scores[index] = -torch.inf
    winner = int(scores.argmax())
    return winner, float(scores[winner].detach())


def fit_context_vectors(counts, steps=80, lr=0.2, seed=42):
    """Fit a fresh lookup table to normalized, explicitly supplied counts.

    The loss is half the SUM of squared errors over all entries. Thus each
    gradient equals weight minus target and one step scales error by (1-lr).
    This is a supervised geometry demonstration, NOT word2vec or GPT training.
    Returns the learned table, its initial weights, targets, and loss history.
    """
    if counts.ndim != 2 or 0 in counts.shape or not counts.is_floating_point():
        raise ValueError('counts must be a nonempty floating matrix')
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    totals = counts.sum(dim=1, keepdim=True)
    if not torch.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError('each token needs a finite, positive count total')
    if not isinstance(steps, int) or steps < 0 or not 0 < lr <= 1:
        raise ValueError('steps must be nonnegative and 0 < lr <= 1')
    if counts.device.type != 'cpu':
        raise ValueError('this deterministic teaching experiment runs on CPU')
    targets = (counts / totals).detach()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        table = nn.Embedding(counts.shape[0], counts.shape[1], dtype=counts.dtype)
    initial = table.weight.detach().clone()
    ids = torch.arange(len(counts), dtype=torch.long)
    history = []
    for step in range(steps + 1):
        loss = 0.5 * (table(ids) - targets).square().sum()
        history.append(float(loss.detach()))
        if step == steps:
            break
        table.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            table.weight -= lr * table.weight.grad
    return table, initial, targets, history
