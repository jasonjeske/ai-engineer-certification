"""Assembly contracts: independent path, parameter accounting and causal boundaries."""
import io

import pytest
import torch
import torch.nn.functional as F

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.gpt_assembly import parameter_budget, trace_forward


def tiny(layers=3, heads=2, dropout=0.0):
    torch.manual_seed(99)
    return GPT(GPTConfig(vocab_size=9, block_size=6, n_layer=layers,
                         n_head=heads, n_embd=8, dropout=dropout)).eval()


@pytest.mark.parametrize('layers,heads', [(0, 1), (1, 2), (3, 2), (3, 4)])
def test_trace_shapes_and_exact_budget(layers, heads):
    model = tiny(layers, heads)
    idx = torch.tensor([[1, 0, 3, 7, 2, 0], [2, 0, 1, 7, 0, 3]])
    trace = trace_forward(model, idx)
    assert trace['tokens'].shape == (2, 6, 8)
    assert trace['positions'].shape == (6, 8)
    assert trace['logits'].shape == (2, 6, 9)
    for i in range(layers):
        assert trace[f'block_{i}'].shape == (2, 6, 8)
    torch.testing.assert_close(trace['logits'], model(idx)[0], rtol=0, atol=0)
    assert model.num_params() == sum(parameter_budget(model.cfg).values())


def test_trace_keeps_gradients_and_shifted_loss():
    model = tiny()
    data = torch.tensor([[1, 0, 3, 7, 2, 0, 8]])
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    logits = trace_forward(model, x)['logits']
    loss = F.cross_entropy(logits.reshape(-1, 9), y.reshape(-1))
    torch.testing.assert_close(loss, model(x, y)[1])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(block.attn.qkv.weight.grad.norm() > 0 for block in model.blocks)


def test_independent_blocks_are_registered_and_head_is_tied():
    model = tiny()
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len({id(b.attn.qkv.weight) for b in model.blocks}) == 3
    assert model.head.weight is model.tok_emb.weight
    names = dict(model.named_parameters())
    assert all(f'blocks.{i}.attn.qkv.weight' in names for i in range(3))
    assert 'head.weight' not in names  # same Parameter counted once
    assert not any('mask' in name for name in names)


def test_checkpoint_roundtrip_preserves_logits_and_tie():
    model = tiny()
    x = torch.tensor([[1, 0, 3]])
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded = tiny()
    loaded.load_state_dict(torch.load(stream, weights_only=True))
    assert loaded.head.weight is loaded.tok_emb.weight
    torch.testing.assert_close(loaded(x)[0], model(x)[0], rtol=0, atol=0)


def test_suffix_cannot_change_past_and_batch_rows_do_not_mix():
    model = tiny(dropout=0.2)  # eval disables dropout even when configured
    x = torch.tensor([[1, 0, 3, 7, 2, 0], [2, 0, 1, 7, 0, 3]])
    changed = x.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 9
    a = trace_forward(model, x)['logits']
    b = trace_forward(model, changed)['logits']
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:])
    torch.testing.assert_close(a[:1], model(x[:1])[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(a[:, :3], model(x[:, :3])[0], rtol=0, atol=1e-6)


@pytest.mark.parametrize('shape', [(6,), (1, 0), (0, 3), (1, 7)])
def test_trace_rejects_bad_shape_or_context(shape):
    with pytest.raises(ValueError):
        trace_forward(tiny(), torch.zeros(shape, dtype=torch.long))


def test_gpt_rejects_context_overflow_and_invalid_head_width():
    with pytest.raises(AssertionError, match='longer than block_size'):
        tiny()(torch.zeros((1, 7), dtype=torch.long))
    with pytest.raises(AssertionError):
        tiny(heads=3)


def test_positions_distinguish_identical_token_lookups():
    trace = trace_forward(tiny(), torch.tensor([[0, 0]]))
    torch.testing.assert_close(trace['tokens'][:, 0], trace['tokens'][:, 1], rtol=0, atol=0)
    assert not torch.allclose(trace['residual_in'][:, 0], trace['residual_in'][:, 1])
