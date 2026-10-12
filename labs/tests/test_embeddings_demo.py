"""Chapter 2 geometry and supervised lookup-table regression checks."""
import pytest
import torch
import torch.nn.functional as F

from tutorlab.embeddings_demo import cosine_matrix, fit_context_vectors, nearest_other


def test_cosine_matches_pytorch_and_scale_invariance():
    rows = torch.tensor([[3., 4.], [4., 3.], [-3., -4.]], dtype=torch.double)
    result = cosine_matrix(rows)
    expected = F.cosine_similarity(rows[:, None, :], rows[None, :, :], dim=-1)
    assert torch.allclose(result, expected)
    assert result[0, 1].item() == pytest.approx(0.96)
    assert result[0, 2].item() == pytest.approx(-1)
    scaled = rows * torch.tensor([[2.], [5.], [0.5]])
    assert torch.allclose(result, cosine_matrix(scaled))


@pytest.mark.parametrize('rows', [torch.zeros(2, 3), torch.ones(3), torch.ones(2, 2, dtype=torch.long),
                                  torch.empty(0, 2), torch.tensor([[float('nan'), 1.]]),
                                  torch.tensor([[float('inf'), 1.]])])
def test_cosine_rejects_undefined_inputs(rows):
    with pytest.raises(ValueError):
        cosine_matrix(rows)


def test_nearest_excludes_self_and_handles_duplicate_rows():
    rows = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    assert nearest_other(rows, 0) == (1, 1.0)
    with pytest.raises(ValueError):
        nearest_other(rows[:1], 0)
    with pytest.raises(ValueError):
        nearest_other(rows, -1)


def test_training_update_is_exact_and_converges():
    counts = torch.tensor([[8., 2., 0.], [7., 3., 0.], [0., 0., 10.]], dtype=torch.double)
    table, initial, targets, history = fit_context_vectors(counts, steps=1, lr=0.2)
    assert torch.allclose(table.weight, initial - 0.2 * (initial - targets))
    assert history[1] == pytest.approx(history[0] * 0.8 ** 2)
    table, _, targets, history = fit_context_vectors(counts)
    assert torch.allclose(table.weight, targets, atol=1e-6)
    assert all(a >= b for a, b in zip(history, history[1:]))
    assert nearest_other(table.weight.detach(), 0)[0] == 1


def test_training_reproducible_without_consuming_global_rng():
    counts = torch.eye(3)
    rng_before = torch.get_rng_state().clone()
    first, _, _, history = fit_context_vectors(counts)
    second, _, _, other_history = fit_context_vectors(counts)
    assert torch.equal(first.weight, second.weight)
    assert history == other_history
    assert torch.equal(rng_before, torch.get_rng_state())


@pytest.mark.parametrize('counts', [torch.zeros(2, 2), torch.tensor([[1., -1.]]),
                                    torch.tensor([[float('nan'), 1.]]), torch.ones(3),
                                    torch.empty(0, 2), torch.ones(2, 2, dtype=torch.long)])
def test_training_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        fit_context_vectors(counts)


def test_lookup_equals_one_hot_and_repeated_gradients_accumulate():
    rows = torch.arange(24., dtype=torch.double).view(6, 4).requires_grad_()
    ids = torch.tensor([0, 0, 1])
    one_hot = F.one_hot(ids, num_classes=len(rows)).to(rows.dtype)
    assert torch.equal(rows[ids], one_hot @ rows)
    rows[ids].sum().backward()
    assert torch.equal(rows.grad[0], torch.full((4,), 2., dtype=torch.double))
    assert torch.equal(rows.grad[1], torch.ones(4, dtype=torch.double))
    assert torch.count_nonzero(rows.grad[2:]) == 0
