"""Exact accounting, real tensor checks and guardrails for scaling_demo."""
from dataclasses import replace

import pytest
import torch

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.scaling_demo import (attention_score_elements, dense_train_flops,
                                  fp32_adam_bytes, illustrative_loss,
                                  kv_cache_bytes, parameter_counts, weight_bytes)


def tiny():
    return GPTConfig(vocab_size=9, block_size=6, n_layer=3, n_head=2, n_embd=8)


@pytest.mark.parametrize('kwargs', [{}, {'n_layer': 1}, {'n_embd': 16},
                                   {'vocab_size': 18}, {'block_size': 12}, {'n_head': 4}])
def test_exact_formula_matches_actual_model(kwargs):
    cfg = replace(tiny(), **kwargs)
    m = GPT(cfg)
    counts = parameter_counts(cfg)
    assert counts['exact'] == sum(p.numel() for p in m.parameters())
    assert counts['omitted_bias_and_norm'] == 13 * cfg.n_layer * cfg.n_embd + 2 * cfg.n_embd


def test_untied_clone_adds_exactly_one_vocabulary_matrix():
    cfg = tiny()
    m = GPT(cfg)
    assert m.head.weight is m.tok_emb.weight
    m.head.weight = torch.nn.Parameter(m.head.weight.detach().clone())
    assert m.head.weight is not m.tok_emb.weight
    assert m.num_params() == parameter_counts(cfg, tied=False)['exact']
    assert m.num_params() - parameter_counts(cfg)['exact'] == cfg.vocab_size * cfg.n_embd


def test_gpt2_meta_exact_count_and_no_real_storage():
    cfg = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    with torch.device('meta'):
        m = GPT(cfg)
    assert parameter_counts(cfg)['exact'] == m.num_params() == 124439808
    assert all(p.is_meta for p in m.parameters())
    assert parameter_counts(cfg)['leading'] == 124318464


def test_weight_bytes_matches_actual_tensor_storage():
    m = GPT(tiny())
    actual = sum(p.numel() * p.element_size() for p in m.parameters())
    assert weight_bytes(m.num_params()) == actual
    assert fp32_adam_bytes(m.num_params()) == 4 * actual


def test_fp32_adam_arrays_match_after_real_step():
    torch.manual_seed(2)
    m = GPT(tiny())
    opt = torch.optim.AdamW(m.parameters())
    x = torch.tensor([[0, 1, 2, 3, 4, 5]])
    _, loss = m(x, (x + 1) % 9)
    loss.backward()
    opt.step()
    tensors = list(m.parameters()) + [p.grad for p in m.parameters()]
    tensors += [state[name] for state in opt.state.values() for name in ('exp_avg', 'exp_avg_sq')]
    actual = sum(t.numel() * t.element_size() for t in tensors)
    assert fp32_adam_bytes(m.num_params()) == actual


def test_heads_repartition_width_without_changing_parameters():
    assert len({parameter_counts(replace(tiny(), n_head=h))['exact'] for h in (1, 2, 4, 8)}) == 1


def test_attention_and_cache_match_tensor_shapes():
    cfg = tiny()
    score = torch.empty(2, cfg.n_head, 3, 3)
    # Two tensors per layer, K and V, each B,H,T,head_size.
    kv = torch.empty(cfg.n_layer, 2, 2, cfg.n_head, 3, cfg.n_embd // cfg.n_head, dtype=torch.float16)
    assert attention_score_elements(2, cfg.n_head, 3) == score.numel()
    assert kv_cache_bytes(cfg, 2, 3, 2) == kv.numel() * kv.element_size()
    assert attention_score_elements(2, cfg.n_head, 6) == 4 * score.numel()
    assert kv_cache_bytes(cfg, 2, 6, 2) == 2 * kv.numel() * kv.element_size()


def test_cache_rejects_context_overflow():
    with pytest.raises(ValueError, match='capacity'):
        kv_cache_bytes(tiny(), 1, 7)


@pytest.mark.parametrize('kwargs', [{'n_head': 3}, {'n_embd': 0}, {'n_layer': -1},
                                   {'vocab_size': True}, {'block_size': 2.5}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        parameter_counts(replace(tiny(), **kwargs))


@pytest.mark.parametrize('fn,args', [(weight_bytes, (0,)), (weight_bytes, (8, 0)),
                                    (attention_score_elements, (1, 0, 3)),
                                    (dense_train_flops, (10, -2)),
                                    (kv_cache_bytes, (tiny(), 0, 2))])
def test_invalid_accounting_inputs(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_fixed_compute_product_and_joint_doubling():
    base = dense_train_flops(100, 200)
    assert dense_train_flops(200, 100) == base
    assert dense_train_flops(200, 400) == 4 * base


def test_constructed_loss_has_balanced_fixed_product_minimum():
    values = [(n, illustrative_loss(n, 16 / n)) for n in (1, 2, 4, 8, 16)]
    assert min(values, key=lambda row: row[1]) == (4, 2.5)
    assert illustrative_loss(1, 16) == illustrative_loss(16, 1)
    assert illustrative_loss(4, 4) < illustrative_loss(4, 2)


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_illustrative_objective_rejects_invalid_units(value):
    with pytest.raises(ValueError):
        illustrative_loss(value, 1)
