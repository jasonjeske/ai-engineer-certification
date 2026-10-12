"""Small, inspectable training primitives for chapter 8. No data downloads.

Bigram shares GPT's (logits, optional loss) interface. Evaluation restores the
caller's mode; it neither steps the optimizer nor clears existing gradients.
"""
import torch
from torch import nn
from torch.nn import functional as F


def shifted_batch(tokens):
    """Turn (B, T+1) integer IDs into contiguous next-token inputs and targets."""
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError('tokens must have shape (B, T+1), with B >= 1 and T >= 1')
    if tokens.dtype != torch.long:
        raise TypeError('tokens must use torch.long IDs')
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class Bigram(nn.Module):
    """Each input ID selects a trainable row of vocabulary logits, not counts."""
    def __init__(self, vocab_size):
        super().__init__()
        if vocab_size < 2:
            raise ValueError('vocab_size must be at least 2')
        self.table = nn.Embedding(vocab_size, vocab_size)
        # Zero logits give exactly uniform probabilities and a visible baseline.
        nn.init.zeros_(self.table.weight)

    def forward(self, idx, targets=None):
        logits = self.table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def mean_ce_logit_gradient(logits, targets):
    """Analytic d(mean cross entropy)/d(logits), unweighted hard targets only."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel():
        raise ValueError('expected logits (N, V) and targets (N,)')
    if targets.numel() == 0:
        raise ValueError('at least one target is required')
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(logits.dtype)
    return (logits.softmax(-1) - one_hot) / targets.numel()


@torch.no_grad()
def evaluate(model, x, y):
    """Measure a fixed batch without recording a graph; restore root mode."""
    was_training = model.training
    model.eval()
    try:
        _, loss = model(x, y)
        return loss.item()
    finally:
        model.train(was_training)


def train_step(model, optimizer, x, y):
    """One update; return the PRE-update batch loss, not a post-step metric."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
