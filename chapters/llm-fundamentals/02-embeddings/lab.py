"""Lab: turn token IDs into vectors, measure geometry, and change it by training.

Run from the project root:
  env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/02-embeddings/lab.py
Or from labs/:
  env -u PYTHONPATH ../.venv/bin/python ../chapters/llm-fundamentals/02-embeddings/lab.py
The path setup also works from any directory. Offline, CPU only, expected under
10 seconds after imports. Context counts below are a hand-authored teaching
fixture, not observations from a corpus or a pretrained model.
"""
import pathlib
import sys

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tutorlab.embeddings_demo import cosine_matrix, fit_context_vectors, nearest_other  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(42)
VOCAB = ['cat', 'dog', 'run', 'walk', 'the', 'a']
STOI = {token: i for i, token in enumerate(VOCAB)}
D = 4


def embed(table, token):
    if token not in STOI:
        raise ValueError(f'token not in vocabulary: {token}')
    return table(torch.tensor(STOI[token], dtype=torch.long))


# --- Part 1: lookup is just selecting rows ------------------------------------
print('Part 1: a random, untrained table (seed 42)')
table = nn.Embedding(len(VOCAB), D)
for token in VOCAB:
    print(f'  {token:4} id={STOI[token]} row={[round(v, 3) for v in embed(table, token).tolist()]}')
ids = torch.tensor([STOI[t] for t in ['the', 'cat', 'run']], dtype=torch.long)
vectors = table(ids)
one_hot = F.one_hot(ids, num_classes=len(VOCAB)).to(table.weight.dtype)
batched = table(torch.stack([ids, ids]))
print(f'  ids {ids.tolist()} -> shape {tuple(vectors.shape)}')
print(f'  batch shape {tuple(batched.shape)}; table parameters {table.weight.numel()}')
print(f'  lookup equals one_hot @ weight: {torch.equal(vectors, one_hot @ table.weight)}')
unknown_rejected = False
try:
    embed(table, 'kitten')
except ValueError as e:
    unknown_rejected = True
    print(f'  unknown rejected: {e}')

# --- Part 2: cosine by hand, then checked against PyTorch ----------------------
print('\nPart 2: hand geometry, not learned meanings')
u, v = torch.tensor([3., 4.]), torch.tensor([4., 3.])
dot = float(u @ v)
nu, nv = float(u.norm()), float(v.norm())
cos = dot / (nu * nv)
print(f'  u={u.tolist()} v={v.tolist()} dot={dot:.1f} norms={nu:.1f},{nv:.1f} cosine={cos:.3f}')
print(f'  cosine(u, 2*u)={float(F.cosine_similarity(u, 2*u, dim=0)):.3f}; dot(u, 2*u)={float(u @ (2*u)):.1f}')
print(f'  cosine(u, -u)={float(F.cosine_similarity(u, -u, dim=0)):.3f}')
print(f'  cosine([1,0], [0,1])={float(F.cosine_similarity(torch.tensor([1., 0.]), torch.tensor([0., 1.]), dim=0)):.3f}')
zero_rejected = False
try:
    cosine_matrix(torch.zeros(1, 2))
except ValueError:
    zero_rejected = True
print(f'  zero rejected by our math helper: {zero_rejected}; PyTorch epsilon convention={float(F.cosine_similarity(torch.zeros(2), u, dim=0)):.1f}')
random_cos = cosine_matrix(table.weight.detach())
for a, b in [('cat', 'dog'), ('cat', 'run'), ('the', 'a')]:
    print(f'  untrained cosine({a}, {b})={float(random_cos[STOI[a], STOI[b]]):.6f}')
print(f'  self cosine is one even before training: {torch.allclose(random_cos.diag(), torch.ones(len(VOCAB)))}')

# --- Part 3: repeated IDs accumulate gradients --------------------------------
print('\nPart 3: trace one backward pass (sum of selected entries)')
grad_table = nn.Embedding(len(VOCAB), D)
repeated = torch.tensor([0, 0, 1])
grad_table(repeated).sum().backward()
grad = grad_table.weight.grad
for i, token in enumerate(VOCAB):
    print(f'  grad {token:4} {grad[i].tolist()}')

# --- Part 4: supervised training on explicit synthetic context counts ---------
print('\nPart 4: fit a hand-authored context fixture, not natural language')
print('  context columns: purrs, barks, quickly, slowly')
counts = torch.tensor([[8., 2., 0., 0.], [7., 3., 0., 0.],
                       [0., 0., 8., 2.], [0., 0., 7., 3.],
                       [3., 3., 2., 2.], [3., 3., 2., 2.]])
trained, initial, targets, losses = fit_context_vectors(counts, steps=80, lr=0.2, seed=42)
for i, token in enumerate(VOCAB):
    print(f'  {token:4} counts={counts[i].tolist()} target={[round(x, 3) for x in targets[i].tolist()]}')
for step in [0, 1, 10, 40, 80]:
    print(f'  step {step:2d} half-squared-error sum={losses[step]:.10f}')
w0, target = float(initial[0, 0]), float(targets[0, 0])
g0 = w0 - target
print(f'  first coordinate: weight={w0:.6f} target={target:.6f} gradient={g0:.6f} next={w0 - 0.2*g0:.6f}')
learned = trained.weight.detach()
print(f'  max target error={float((learned-targets).abs().max()):.9f}')
learned_cos = cosine_matrix(learned)
for a, b in [('cat', 'dog'), ('cat', 'run'), ('the', 'a')]:
    print(f'  trained cosine({a}, {b})={float(learned_cos[STOI[a], STOI[b]]):.6f}')
for token in VOCAB:
    before, _ = nearest_other(initial, STOI[token])
    after, score = nearest_other(learned, STOI[token])
    print(f'  nearest other to {token:4}: before={VOCAB[before]:4} after={VOCAB[after]:4} cosine={score:.6f}')

# --- Part 5: IDs and coordinate names are not meanings ------------------------
print('\nPart 5: relabel IDs and rotate every row consistently')
permutation = torch.tensor([2, 5, 0, 4, 1, 3])
old_to_new = torch.argsort(permutation)
relabel_ok = torch.equal(learned[ids], learned[permutation][old_to_new[ids]])
rotation = torch.tensor([[0., -1., 0., 0.], [1., 0., 0., 0.],
                         [0., 0., 0., -1.], [0., 0., 1., 0.]])
rotation_ok = torch.allclose(cosine_matrix(learned @ rotation), learned_cos, atol=1e-6)
print(f'  consistent ID relabel preserves vectors: {relabel_ok}')
print(f'  orthogonal rotation preserves all cosines: {rotation_ok}')
print('  repeated token IDs return identical static vectors, regardless of neighbors')
example_v, example_d = 50_000, 768
parameters = example_v * example_d
print(f'  hypothetical table: V={example_v:,} D={example_d} parameters={parameters:,}')
print(f'  float32 weights only: {parameters * 4:,} bytes = {parameters * 4 / 1e6:.1f} MB')

# TRY THIS: replace dog's counts with run's counts. Predict cat's closest other
#   row, then inspect the ranking. Existing semantic-pair assertions should fail.
# TRY THIS: set steps=0. Predict which checks fail and why self cosine still passes.
# TRY THIS: repeat cat three times in Part 3. Predict its gradient before running.


def check():
    assert vectors.shape == (3, D) and batched.shape == (2, 3, D)
    assert torch.equal(vectors, one_hot @ table.weight)
    assert unknown_rejected and zero_rejected
    assert abs(cos - float(F.cosine_similarity(u, v, dim=0))) < 1e-6
    assert abs(cos - 0.96) < 1e-6
    assert torch.allclose(random_cos.diag(), torch.ones(len(VOCAB)), atol=1e-6)
    assert torch.equal(grad[0], torch.full((D,), 2.))
    assert torch.equal(grad[1], torch.ones(D)) and torch.count_nonzero(grad[2:]) == 0
    assert torch.equal(initial, table.weight.detach()), 'training starts from same random table'
    assert losses[-1] < 1e-10 and max((learned - targets).abs().flatten()) < 1e-6
    assert losses[1] < losses[0] and all(a >= b for a, b in zip(losses, losses[1:]))
    assert nearest_other(learned, STOI['cat'])[0] == STOI['dog']
    assert nearest_other(learned, STOI['run'])[0] == STOI['walk']
    assert learned_cos[0, 1] > 0.98 and abs(float(learned_cos[0, 2])) < 1e-5
    assert relabel_ok and rotation_ok
    assert torch.equal(trained(torch.tensor([0, 0]))[0], trained(torch.tensor([0, 0]))[1])


check()
print('\nCHECK PASSED')
