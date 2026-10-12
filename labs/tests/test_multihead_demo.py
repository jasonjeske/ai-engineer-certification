"""Multi-head correctness against hand arithmetic and independent PyTorch code."""
import copy

import pytest
import torch

from tutorlab.multihead_demo import MultiHead, worked_example


def test_worked_example_and_output_mixing():
    x, m = worked_example()
    out, weights, cat = m(x)
    torch.testing.assert_close(weights[0, :, -1], torch.tensor(
        [[1/7, 2/7, 4/7], [4/7, 2/7, 1/7]], dtype=x.dtype))
    torch.testing.assert_close(cat[0, -1], torch.tensor([26/7, 16/7, 2, 10/7], dtype=x.dtype))
    torch.testing.assert_close(out[0, -1], torch.tensor([40/7, 6/7, 12/7, 26/7], dtype=x.dtype))
    assert not torch.allclose(out, cat)


@pytest.mark.parametrize('heads', [1, 2, 4])
@pytest.mark.parametrize('causal', [False, True])
def test_packed_and_pytorch_output_weight_and_gradient_parity(heads, causal):
    torch.manual_seed(51)
    m = MultiHead(8, heads).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    y, weights, cat = m(x, causal)
    yp, wp, cp = m.packed_forward(x, causal)
    for actual, expected in ((yp, y), (wp, weights), (cp, cat)):
        torch.testing.assert_close(actual, expected)
    parameters = tuple(m.parameters())
    loop_grads = torch.autograd.grad(y.square().sum(), (x,) + parameters, retain_graph=True)
    packed_grads = torch.autograd.grad(yp.square().sum(), (x,) + parameters)
    for actual, expected in zip(packed_grads, loop_grads):
        torch.testing.assert_close(actual, expected)
    ref = torch.nn.MultiheadAttention(8, heads, bias=False, dropout=0.0,
                                     batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        ref.in_proj_weight.copy_(m.packed_weights())
        ref.out_proj.weight.copy_(m.proj.weight)
    blocked = torch.ones(5, 5, dtype=torch.bool).triu(1) if causal else None
    yr, wr = ref(x, x, x, attn_mask=blocked, average_attn_weights=False)
    torch.testing.assert_close(yr, y)
    torch.testing.assert_close(wr, weights)
    gx, gi, go = torch.autograd.grad(yr.square().sum(), (x, ref.in_proj_weight, ref.out_proj.weight))
    torch.testing.assert_close(gx, loop_grads[0])
    named = dict(zip(dict(m.named_parameters()), loop_grads[1:]))
    for projection_index, name in enumerate(('query', 'key', 'value')):
        for h in range(heads):
            start = projection_index * 8 + h * m.head_width
            torch.testing.assert_close(gi[start:start + m.head_width], named[f'heads.{h}.{name}.weight'])
    torch.testing.assert_close(go, named['proj.weight'])


def test_causality_and_batch_isolation():
    x, m = worked_example()
    batch = x.repeat(2, 1, 1)
    original, weights, _ = m(batch)
    changed = batch.clone()
    changed[0, -1, 1:] += 3
    altered = m(changed)[0]
    torch.testing.assert_close(original[0, :-1], altered[0, :-1], rtol=0, atol=0)
    torch.testing.assert_close(original[1], altered[1], rtol=0, atol=0)
    assert not torch.allclose(original[0, -1], altered[0, -1])
    assert not torch.allclose(m(batch, False)[0][0, :-1], m(changed, False)[0][0, :-1])
    assert torch.count_nonzero(weights.triu(1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_single_token_and_noncontiguous_input():
    torch.manual_seed(5)
    m = MultiHead(8, 2).double()
    x = torch.randn(2, 8, 3, dtype=torch.float64).transpose(1, 2)
    assert not x.is_contiguous()
    torch.testing.assert_close(m(x)[0], m.packed_forward(x)[0])
    single = x[:, :1]
    out, weights, _ = m(single)
    values = torch.cat([h.value(single) for h in m.heads], dim=-1)
    torch.testing.assert_close(out, m.proj(values))
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_heads_are_independent_and_use_full_input_width():
    torch.manual_seed(5)
    m = MultiHead(8, 2).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    assert m.heads[0].query.weight.shape == (4, 8)
    assert m.heads[0].query.weight.data_ptr() != m.heads[1].query.weight.data_ptr()
    _, w0, c0 = m(x)
    with torch.no_grad():
        m.heads[1].value.weight.add_(0.5)
    _, w1, c1 = m(x)
    torch.testing.assert_close(w0, w1)
    torch.testing.assert_close(c0[..., :4], c1[..., :4])
    assert not torch.allclose(c0[..., 4:], c1[..., 4:])


def test_head_permutation_requires_matching_projection_columns():
    x, m = worked_example()
    swapped = copy.deepcopy(m)
    swapped.heads = torch.nn.ModuleList([swapped.heads[1], swapped.heads[0]])
    assert not torch.allclose(m(x)[0], swapped(x)[0])
    with torch.no_grad():
        swapped.proj.weight.copy_(m.proj.weight[:, [2, 3, 0, 1]])
    torch.testing.assert_close(m(x)[0], swapped(x)[0])


def test_fixed_width_parameters_and_registered_gradients():
    torch.manual_seed(5)
    for heads in (1, 2, 4):
        m = MultiHead(8, heads).double()
        assert sum(p.numel() for p in m.parameters()) == 4 * 8 * 8
        x = torch.randn(2, 4, 8, dtype=torch.float64)
        m(x)[0].square().sum().backward()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum() > 0


@pytest.mark.parametrize('width,heads', [(0, 2), (8, 0), (7, 2), (4, 8), (-4, 2)])
def test_invalid_configuration(width, heads):
    with pytest.raises(ValueError, match='divisible'):
        MultiHead(width, heads)


@pytest.mark.parametrize('shape', [(3, 8), (2, 3, 7), (2, 0, 8), (0, 3, 8)])
def test_invalid_inputs(shape):
    m = MultiHead(8, 2)
    for run in (m, m.packed_forward):
        with pytest.raises(ValueError, match='nonempty'):
            run(torch.zeros(shape))
