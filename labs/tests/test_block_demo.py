"""Chapter 6: normalization, block wiring, gradients, locality and causality."""
import pytest
import torch
import torch.nn.functional as F

from tutorlab.block_demo import InspectableBlock, layer_norm_parts, scalar_path_gradient
from tutorlab.gpt import Block, GPTConfig


def fixture():
    torch.manual_seed(6)
    cfg = GPTConfig(block_size=4, n_embd=8, n_head=2, dropout=0.0)
    return cfg, InspectableBlock(cfg).double(), torch.randn(2, 4, 8, dtype=torch.float64)


def test_manual_norm_matches_values_and_gradients():
    x = torch.tensor([[1., 3., 5., 7.], [2., 2., 2., 2.]], dtype=torch.float64, requires_grad=True)
    ours, mean, var = layer_norm_parts(x)
    theirs = F.layer_norm(x, (4,), eps=1e-5)
    torch.testing.assert_close(ours, theirs)
    torch.testing.assert_close(mean[:, 0], torch.tensor([4., 2.], dtype=x.dtype))
    torch.testing.assert_close(var[:, 0], torch.tensor([5., 0.], dtype=x.dtype))
    probe = torch.arange(8, dtype=x.dtype).reshape_as(x)
    a = torch.autograd.grad((ours * probe).sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad((theirs * probe).sum(), x)[0]
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()


def test_norm_affine_is_not_standardized_output():
    x = torch.tensor([1., 3., 5., 7.], dtype=torch.float64)
    z, _, _ = layer_norm_parts(x)
    out = F.layer_norm(x, (4,), torch.full_like(x, 2), torch.full_like(x, 3))
    torch.testing.assert_close(out, 2 * z + 3)
    assert out.mean().item() == pytest.approx(3)
    assert out.var(unbiased=False).item() == pytest.approx(4, abs=1e-5)


def test_trace_matches_existing_block_and_updated_second_input():
    cfg, model, x = fixture()
    reference = Block(cfg).double()
    reference.load_state_dict(model.state_dict())
    trace = model.trace(x)
    torch.testing.assert_close(trace['output'], reference(x))
    torch.testing.assert_close(trace['norm2'], model.ln2(x + trace['attention_update']))
    assert not torch.allclose(trace['norm2'], model.ln2(x))
    assert all(z.shape == x.shape for z in trace.values())


def test_mlp_is_row_local_and_shared():
    _, model, x = fixture()
    changed = x.clone()
    changed[:, -1, 0] += 10
    torch.testing.assert_close(model.mlp(x)[:, :-1], model.mlp(changed)[:, :-1], rtol=0, atol=0)
    a = x[:, :1].expand(-1, 4, -1).clone()
    y = model.mlp(a)
    torch.testing.assert_close(y[:, 0], y[:, -1])


def test_block_and_stack_cannot_read_future():
    cfg, model, x = fixture()
    stack = torch.nn.Sequential(model, InspectableBlock(cfg).double())
    changed = x.clone()
    changed[:, -1, 0] += 10
    for net in (model, stack):
        a, b = net(x), net(changed)
        torch.testing.assert_close(a[:, :-1], b[:, :-1], rtol=0, atol=0)
        assert not torch.allclose(a[:, -1], b[:, -1])


def test_global_normalization_negative_control_leaks():
    _, model, x = fixture()
    changed = x.clone()
    changed[:, -1, 0] += 10
    right = (model.ln1(x)[:, :-1] - model.ln1(changed)[:, :-1]).abs().max()
    wrong = (F.layer_norm(x, (4, 8))[:, :-1] - F.layer_norm(changed, (4, 8))[:, :-1]).abs().max()
    assert right == 0
    assert wrong > 0.1


def test_zero_branch_outputs_give_identity_and_identity_gradient():
    _, model, x = fixture()
    with torch.no_grad():
        for layer in (model.attn.proj, model.mlp.net[2]):
            layer.weight.zero_()
            layer.bias.zero_()
    x.requires_grad_()
    torch.testing.assert_close(model(x), x, rtol=0, atol=0)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_every_parameter_has_finite_gradient():
    _, model, x = fixture()
    x.requires_grad_()
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert model.attn.qkv.weight.grad.abs().sum() > 0
    assert model.mlp.net[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('shape', [(2, 4, 7), (4, 8), (0, 4, 8), (2, 0, 8), (2, 5, 8)])
def test_bad_input_shapes_rejected(shape):
    _, model, _ = fixture()
    with pytest.raises(ValueError):
        model(torch.zeros(shape, dtype=torch.float64))


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        InspectableBlock(GPTConfig(n_embd=7, n_head=2))


def test_norm_validation():
    for x, eps in [(torch.ones(2, dtype=torch.long), 1e-5), (torch.ones(2), 0), (torch.empty(0), 1e-5)]:
        with pytest.raises(ValueError):
            layer_norm_parts(x, eps)


def test_scalar_gradient_control_and_cancellation():
    assert scalar_path_gradient(8, 0.5, False) == 0.5 ** 8
    assert scalar_path_gradient(8, 0.5, True) == 1.5 ** 8
    assert scalar_path_gradient(8, -1, True) == 0
    with pytest.raises(ValueError):
        scalar_path_gradient(0, 0.5, True)
