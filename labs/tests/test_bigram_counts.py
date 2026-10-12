"""Regression tests for the chapter 3 count model, including failure cases."""
import math

import pytest
import torch
import torch.nn.functional as F

from tutorlab.bigram_counts import count_pairs, generate, mean_nll, probabilities


def toy():
    return count_pairs([[0, 1, 0, 2, 0, 1, 0]], 3)


def test_counts_match_hand_and_keep_boundaries():
    assert torch.equal(toy(), torch.tensor([[0, 2, 1], [2, 0, 0], [1, 0, 0]]))
    counts = count_pairs([[0, 1], [2, 0], [], [1]], 3)
    assert counts.sum() == 2
    assert counts[1, 2] == 0
    assert count_pairs([[0, 0, 0, 0]], 1).item() == 3


def test_smoothing_and_zero_rows():
    p = probabilities(toy())
    assert torch.allclose(p[0], torch.tensor([1 / 6, 3 / 6, 2 / 6], dtype=torch.float64))
    assert torch.allclose(p.sum(1), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(probabilities(torch.zeros(3, 3))[2], torch.full((3,), 1 / 3, dtype=torch.float64))
    with pytest.raises(ValueError, match='unobserved'):
        probabilities(torch.zeros(3, 3), alpha=0)


def test_nll_matches_hand_and_cross_entropy():
    p = probabilities(toy())
    ids = torch.tensor([0, 1, 0])
    loss = mean_nll(p, [ids])
    assert loss == pytest.approx(-math.log(0.5 * 0.6) / 2)
    assert loss == pytest.approx(F.cross_entropy(p.log()[ids[:-1]], ids[1:]).item())
    assert math.isinf(mean_nll(probabilities(toy(), alpha=0), [[0, 1, 2]]))
    with pytest.raises(ValueError, match='transition'):
        mean_nll(p, [[], [0]])


def test_sampling_is_reproducible_and_does_not_fit():
    p = probabilities(toy())
    before = p.clone()
    a = generate(p, 0, 40, seed=7)
    assert a == generate(p, 0, 40, seed=7)
    assert len(a) == 41 and a[0] == 0 and set(a) <= {0, 1, 2}
    assert generate(p, 0, 6, greedy=True) == [0, 1, 0, 1, 0, 1, 0]
    assert torch.equal(p, before)
    assert generate(p, 0, 0) == [0]


@pytest.mark.parametrize('ids', [[-1, 0], [0, 3], [0.5, 1.0], [[0, 1]]])
def test_bad_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        count_pairs([ids], 3)


@pytest.mark.parametrize('counts,alpha', [([[1, 2, 3]], 1), ([[1, -1], [0, 1]], 1),
                                         ([[float('nan')]], 1), ([[1]], -1), ([[1]], float('inf'))])
def test_invalid_counts_and_alpha(counts, alpha):
    with pytest.raises(ValueError):
        probabilities(counts, alpha)


def test_bad_distributions_and_sampling_limits():
    with pytest.raises(ValueError, match='sum to one'):
        mean_nll(torch.ones(2, 2), [[0, 1]])
    with pytest.raises(ValueError):
        generate(probabilities(toy()), 3, 1)
    with pytest.raises(ValueError):
        generate(probabilities(toy()), 0, -1)


def test_rows_not_columns_and_no_implicit_wraparound():
    c = count_pairs([[0, 1, 0, 1, 2]], 3)
    assert c[2, 0] == 0 and c.sum() == 4
    p = probabilities(c)
    assert torch.allclose(p.sum(1), torch.ones(3, dtype=torch.float64))
    assert not torch.allclose(p.sum(0), torch.ones(3, dtype=torch.float64))
