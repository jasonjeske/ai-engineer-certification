"""Chapter 8: gradients, batch alignment, update ownership and evaluation."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.training_loop_demo import (
    Bigram, evaluate, mean_ce_logit_gradient, shifted_batch, train_step,
)


def test_shift_alignment_and_contiguous_storage():
    ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    x, y = shifted_batch(ids)
    assert x.tolist() == [[0, 1, 2], [3, 2, 1]]
    assert y.tolist() == [[1, 2, 3], [2, 1, 0]]
    assert x.is_contiguous() and y.is_contiguous()


@pytest.mark.parametrize('ids', [torch.tensor([0, 1]), torch.zeros(1, 1, dtype=torch.long),
                                 torch.zeros(0, 3, dtype=torch.long)])
def test_reject_invalid_batch_shapes(ids):
    with pytest.raises(ValueError):
        shifted_batch(ids)


def test_reject_float_ids():
    with pytest.raises(TypeError):
        shifted_batch(torch.zeros(1, 3))


def test_uniform_bigram_baseline_and_optional_loss():
    model = Bigram(3).double()
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0]]))
    logits, loss = model(x, y)
    assert logits.shape == (1, 3, 3)
    assert abs(loss.item() - math.log(3)) < 1e-12
    assert model(x)[1] is None


def test_bad_vocabulary_rejected():
    with pytest.raises(ValueError):
        Bigram(1)


def test_mean_gradient_matches_autograd_and_finite_difference():
    z = torch.tensor([[0.2, -0.8, 1.1], [-1.0, 0.0, 2.0]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([2, 0])
    F.cross_entropy(z, y).backward()
    expected = mean_ce_logit_gradient(z.detach(), y)
    torch.testing.assert_close(z.grad, expected, atol=1e-12, rtol=1e-12)
    for row in range(z.shape[0]):
        for col in range(z.shape[1]):
            plus, minus = z.detach().clone(), z.detach().clone()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            numeric = (F.cross_entropy(plus, y) - F.cross_entropy(minus, y)) / 2e-6
            assert abs(numeric.item() - z.grad[row, col].item()) < 1e-8
    torch.testing.assert_close(z.grad.sum(-1), torch.zeros(2, dtype=torch.double), atol=1e-12, rtol=0)


@pytest.mark.parametrize('z,y', [(torch.zeros(1, 2, 3), torch.tensor([0])),
                                (torch.zeros(2, 3), torch.tensor([0])),
                                (torch.zeros(0, 3), torch.tensor([], dtype=torch.long))])
def test_bad_gradient_shapes_rejected(z, y):
    with pytest.raises(ValueError):
        mean_ce_logit_gradient(z, y)


def test_backward_accumulates_but_does_not_change_weights():
    m = Bigram(3).double()
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0]]))
    before = m.table.weight.detach().clone()
    m(x, y)[1].backward()
    first = m.table.weight.grad.clone()
    m(x, y)[1].backward()
    assert torch.equal(m.table.weight, before)
    torch.testing.assert_close(m.table.weight.grad, 2 * first)


def test_train_step_clears_stale_grad_and_returns_pre_update_loss():
    m = Bigram(3).double()
    ref = copy.deepcopy(m)
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0]]))
    pre = evaluate(m, x, y)
    m.table.weight.grad = torch.full_like(m.table.weight, 999.0)
    opt = torch.optim.SGD(m.parameters(), lr=0.3)
    returned = train_step(m, opt, x, y)
    ref(x, y)[1].backward()
    expected = ref.table.weight.detach() - 0.3 * ref.table.weight.grad
    torch.testing.assert_close(m.table.weight, expected)
    assert returned == pre
    assert evaluate(m, x, y) < returned


def test_zero_learning_rate_changes_no_weights():
    m = Bigram(3).double()
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0]]))
    before = m.table.weight.detach().clone()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    train_step(m, opt, x, y)
    assert torch.equal(m.table.weight, before)


@pytest.mark.parametrize('training', [True, False])
def test_evaluate_restores_mode_and_preserves_weights_and_gradients(training):
    torch.manual_seed(808)
    m = GPT(GPTConfig(vocab_size=3, block_size=3, n_layer=1, n_head=1, n_embd=8, dropout=0.5))
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0]]))
    m(x, y)[1].backward()
    m.train(training)
    weights = [p.detach().clone() for p in m.parameters()]
    grads = [p.grad.clone() for p in m.parameters()]
    result = evaluate(m, x, y)
    assert result == evaluate(m, x, y)  # dropout disabled during both calls
    assert m.training == training
    assert all(torch.equal(a, b) for a, b in zip(weights, m.parameters()))
    assert all(torch.equal(a, p.grad) for a, p in zip(grads, m.parameters()))


def test_evaluate_restores_mode_even_when_forward_fails():
    m = Bigram(3).train()
    with pytest.raises(IndexError):
        evaluate(m, torch.tensor([[3]]), torch.tensor([[0]]))
    assert m.training


def test_unequal_microbatches_match_full_mean_when_token_weighted():
    m = Bigram(3).double()
    micro = copy.deepcopy(m)
    x, y = shifted_batch(torch.tensor([[0, 1, 0, 2, 0, 1]]))
    m(x, y)[1].backward()
    for start, end in [(0, 2), (2, 5)]:
        loss = micro(x[:, start:end], y[:, start:end])[1]
        (loss * ((end - start) / y.numel())).backward()
    torch.testing.assert_close(m.table.weight.grad, micro.table.weight.grad, atol=1e-12, rtol=1e-12)


def test_same_step_interface_trains_shared_gpt():
    torch.manual_seed(808)
    m = GPT(GPTConfig(vocab_size=3, block_size=3, n_layer=1, n_head=1, n_embd=8))
    x, y = shifted_batch(torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]))
    before = evaluate(m, x, y)
    opt = torch.optim.AdamW(m.parameters(), lr=0.02, weight_decay=0.0)
    for _ in range(50):
        train_step(m, opt, x, y)
    assert evaluate(m, x, y) < 0.1
    assert evaluate(m, x, y) < before
