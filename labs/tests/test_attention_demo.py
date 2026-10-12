"""Single-head correctness, negative cases, and independent PyTorch parity."""
import pytest
import torch
import torch.nn.functional as F

from tutorlab.attention_demo import SingleHead, attention, worked_example


def test_hand_calculation():
    x, head = worked_example()
    out, weights = head(x)
    expected = torch.tensor([[1, 0, 0], [1/3, 2/3, 0], [1/7, 2/7, 4/7]], dtype=x.dtype)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(out[-1], torch.tensor([26/7, 16/7], dtype=x.dtype))


@pytest.mark.parametrize('causal', [False, True])
def test_batched_parity_and_gradients(causal):
    g = torch.Generator().manual_seed(4)
    q = torch.randn(2, 3, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 5, 2, generator=g, dtype=torch.float64, requires_grad=True)
    out, weights = attention(q, k, v, causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.0)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    grads = torch.autograd.grad(out.square().sum(), (q, k, v), retain_graph=True)
    refs = torch.autograd.grad(ref.square().sum(), (q, k, v))
    for grad, expected in zip(grads, refs):
        torch.testing.assert_close(grad, expected)


def test_prefix_invariance_and_noncausal_negative_control():
    x, head = worked_example()
    changed = x.clone()
    changed[-1, 2] += 2
    changed[-1, 4:] += 10
    out, weights = head(x)
    later, _ = head(changed)
    torch.testing.assert_close(out[:-1], later[:-1], rtol=0, atol=0)
    assert not torch.allclose(out[-1], later[-1])
    assert torch.count_nonzero(weights.triu(1)) == 0
    unmasked, _ = head(x, causal=False)
    leaking, _ = head(changed, causal=False)
    assert not torch.allclose(unmasked[:-1], leaking[:-1])


def test_single_token_returns_value():
    q = torch.tensor([[2., -3.]])
    v = torch.tensor([[9., 4., -1.]])
    out, weights = attention(q, -q, v)
    torch.testing.assert_close(out, v)
    assert weights.item() == 1


def test_equal_scores_average_and_value_change_preserves_weights():
    x, head = worked_example()
    q, k, v = head.query(x), head.key(x), head.value(x)
    out, _ = attention(torch.zeros_like(q), k, v)
    torch.testing.assert_close(out[-1], v.mean(0))
    _, before = attention(q, k, v)
    _, after = attention(q, k, v + 10)
    torch.testing.assert_close(before, after)


def test_wrong_masks_are_detectable():
    x, head = worked_example()
    scores = head.query(x) @ head.key(x).T / (2 ** 0.5)
    allowed = torch.ones(3, 3, dtype=torch.bool).tril()
    zero_mask = scores.masked_fill(~allowed, 0).softmax(-1)
    post_mask = scores.softmax(-1) * allowed
    assert torch.count_nonzero(zero_mask.triu(1)) > 0
    assert not torch.allclose(post_mask.sum(-1), torch.ones(3, dtype=x.dtype))
    assert torch.isnan(scores.fill_(float('-inf')).softmax(-1)).all()


@pytest.mark.parametrize('case', ['rank', 'tokens', 'width', 'batch', 'empty'])
def test_bad_shapes_fail_clearly(case):
    q, k, v = torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 4)
    if case == 'rank':
        q = q[0]
    elif case == 'tokens':
        v = v[:2]
    elif case == 'width':
        k = k[:, :1]
    elif case == 'batch':
        q = q.unsqueeze(0)
    else:
        q, k, v = q[:0], k[:0], v[:0]
    with pytest.raises(ValueError):
        attention(q, k, v)


def test_learnable_projections_have_gradients():
    torch.manual_seed(4)
    head = SingleHead(6, 2).double()
    x = torch.randn(2, 4, 6, dtype=torch.float64)
    out, _ = head(x)
    out.square().mean().backward()
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
