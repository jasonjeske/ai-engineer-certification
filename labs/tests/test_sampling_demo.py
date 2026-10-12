"""Generation policy, boundary cases and real GPT integration tests."""
from types import SimpleNamespace

import pytest
import torch

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.sampling_demo import entropy_bits, generate, probabilities


@pytest.fixture
def logits():
    return torch.tensor([4., 2., 1., 1.], dtype=torch.float64).log()


def test_temperature_matches_hand_calculation(logits):
    expected = torch.tensor([16., 4., 1., 1.], dtype=torch.float64) / 22
    torch.testing.assert_close(probabilities(logits, 0.5), expected)
    torch.testing.assert_close(probabilities(logits), torch.tensor([.5, .25, .125, .125], dtype=torch.float64))
    assert entropy_bits(probabilities(logits, .5)) < entropy_bits(probabilities(logits))
    assert entropy_bits(probabilities(logits)) < entropy_bits(probabilities(logits, 2))


def test_shift_invariance_and_no_mutation(logits):
    original = logits.clone()
    torch.testing.assert_close(probabilities(logits + 1000), probabilities(logits))
    probabilities(logits, top_k=2, top_p=.7)
    assert torch.equal(original, logits)


def test_extreme_logits_are_stable():
    p = probabilities(torch.tensor([1e30, -1e30]), temperature=1e-300)
    assert torch.equal(p, torch.tensor([1., 0.], dtype=torch.float64))


def test_greedy_ties_and_exact_k():
    tied = torch.zeros(4)
    assert probabilities(tied, 0).tolist() == [1., 0., 0., 0.]
    assert probabilities(tied, top_k=2).tolist() == [.5, .5, 0., 0.]


def test_topk_renormalizes_and_large_k_is_identity(logits):
    torch.testing.assert_close(probabilities(logits, top_k=2), torch.tensor([2/3, 1/3, 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(probabilities(logits, top_k=99), probabilities(logits))


def test_nucleus_includes_crossing_token(logits):
    p = probabilities(logits, top_p=.7)
    assert (p > 0).tolist() == [True, True, False, False]
    torch.testing.assert_close(p, probabilities(logits, top_k=2))


def test_nucleus_exact_boundary_and_tiny_threshold():
    logits = torch.zeros(4)
    assert probabilities(logits, top_p=.5).tolist() == [.5, .5, 0., 0.]
    assert probabilities(logits, top_p=1e-12).tolist() == [1., 0., 0., 0.]
    assert probabilities(logits, top_p=1.).tolist() == [.25, .25, .25, .25]


def test_nucleus_adapts_and_rows_are_independent(logits):
    rows = torch.stack([logits, torch.zeros_like(logits)])
    p = probabilities(rows, top_p=.7)
    assert (p > 0).sum(dim=-1).tolist() == [2, 3]
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(2, dtype=torch.float64))


def test_filter_order_is_documented():
    logits = torch.tensor([.6, .25, .15], dtype=torch.float64).log()
    # Top-k first makes leading mass .6/.85 > .7, so top-p retains one.
    assert (probabilities(logits, top_k=2, top_p=.7) > 0).sum() == 1
    assert (probabilities(logits, top_p=.7) > 0).sum() == 2


def test_masked_tokens_never_sampled():
    p = probabilities(torch.tensor([0., float('-inf'), -2.]))
    draw = torch.multinomial(p, 1000, replacement=True, generator=torch.Generator().manual_seed(9))
    assert 1 not in draw.tolist()


@pytest.mark.parametrize('options', [
    {'temperature': -1}, {'temperature': float('nan')}, {'temperature': float('inf')},
    {'top_k': 0}, {'top_k': -2}, {'top_k': 1.5}, {'top_k': True},
    {'top_p': 0}, {'top_p': 1.1}, {'top_p': float('nan')},
])
def test_invalid_options_rejected(logits, options):
    with pytest.raises(ValueError):
        probabilities(logits, **options)


@pytest.mark.parametrize('values', [[], [float('-inf')], [float('nan'), 0.], [float('inf'), 0.]])
def test_invalid_logits_rejected(values):
    with pytest.raises(ValueError):
        probabilities(torch.tensor(values))


class NextIDModel(torch.nn.Module):
    """Explicit fixture, not a trained LM: position j predicts (id[j]+1)%V."""
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=3, vocab_size=5)
        self.seen = []

    def forward(self, idx):
        self.seen.append(idx.clone())
        logits = torch.full((*idx.shape, 5), float('-inf'))
        logits.scatter_(-1, ((idx + 1) % 5).unsqueeze(-1), 0.)
        assert not torch.is_grad_enabled()
        return logits, None


def test_last_position_feedback_and_crop():
    model = NextIDModel().eval()
    prompt = torch.tensor([[0, 1, 2, 3]])
    result = generate(model, prompt, 3, temperature=0)
    assert result.ids.tolist() == [[0, 1, 2, 3, 4, 0, 1]]
    assert [x.tolist() for x in model.seen] == [[[1, 2, 3]], [[2, 3, 4]], [[3, 4, 0]]]
    assert result.context_lengths == [3, 3, 3]
    assert result.stop_reason == 'length'
    assert prompt.tolist() == [[0, 1, 2, 3]]


def test_eos_and_zero_budget():
    model = NextIDModel().eval()
    prompt = torch.tensor([[0, 1]])
    result = generate(model, prompt, 9, eos_id=3)
    assert result.ids.tolist() == [[0, 1, 2, 3]]
    assert result.stop_reason == 'eos'
    calls = len(model.seen)
    zero = generate(model, prompt, 0)
    assert torch.equal(zero.ids, prompt) and len(model.seen) == calls
    assert zero.context_lengths == []
    # An EOS in the prompt is context, not a newly generated stop token.
    assert generate(model, torch.tensor([[3]]), 1, eos_id=3).ids.tolist() == [[3, 4]]


@pytest.mark.parametrize('prompt,budget,eos', [
    (torch.empty((1, 0), dtype=torch.long), 1, None),
    (torch.tensor([[0], [1]]), 1, None),
    (torch.tensor([[0.]]), 1, None),
    (torch.tensor([[-1]]), 1, None),
    (torch.tensor([[5]]), 1, None),
    (torch.tensor([[0]]), -1, None),
    (torch.tensor([[0]]), 1.5, None),
    (torch.tensor([[0]]), 1, 5),
])
def test_invalid_generation_requests(prompt, budget, eos):
    with pytest.raises(ValueError):
        generate(NextIDModel().eval(), prompt, budget, eos_id=eos)


def test_training_mode_is_not_silently_changed():
    model = NextIDModel()
    with pytest.raises(ValueError, match='eval'):
        generate(model, torch.tensor([[0]]), 1)
    assert model.training


def test_seeded_real_gpt_and_frozen_weights():
    torch.manual_seed(4)
    model = GPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=.2)).eval()
    prompt = torch.tensor([[1, 2]])
    original = {n: p.clone() for n, p in model.named_parameters()}
    global_rng = torch.random.get_rng_state().clone()
    a = generate(model, prompt, 8, seed=19)
    b = generate(model, prompt, 8, seed=19)
    assert torch.equal(a.ids, b.ids)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert a.ids.shape == (1, 10)
    assert a.context_lengths == [2, 3, 4, 4, 4, 4, 4, 4]
    for name, param in model.named_parameters():
        assert torch.equal(param, original[name]) and param.grad is None
    greedy = generate(model, prompt, 8, temperature=0)
    top1 = generate(model, prompt, 8, top_k=1)
    assert torch.equal(greedy.ids, top1.ids)
