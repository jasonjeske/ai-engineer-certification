"""Lab: self-attention as measurable, content-dependent mixing.

Goal: hand-calculate a head, verify learned projection plumbing, and catch leakage.
Run from the project: env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/04-self-attention/lab.py
Or from labs: env -u PYTHONPATH ../.venv/bin/python ../chapters/llm-fundamentals/04-self-attention/lab.py
Offline CPU lab; expected runtime under 10 seconds. No training or downloads.
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'labs'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tutorlab.attention_demo import SingleHead, attention, worked_example  # noqa: E402

torch.manual_seed(4)
torch.set_num_threads(1)


def matrix(name, tensor):
    print(f'  {name}')
    for row in tensor.detach().tolist():
        print('   [' + ', '.join(f'{number:.6f}' for number in row) + ']')


# --- Part 1: from uniform mixing to a hand-checkable head ---
print('Part 1: a constructed example, not learned language semantics')
x, head = worked_example()
q, k, v = head.query(x), head.key(x), head.value(x)
out, weights = head(x)
print(f'  X {tuple(x.shape)}, Q {tuple(q.shape)}, K {tuple(k.shape)}, V {tuple(v.shape)}')
matrix('input features X', x)
matrix('values V', v)
uniform_out, uniform_weights = attention(torch.zeros_like(q), k, v)
matrix('equal-score weights', uniform_weights)
matrix('equal-score outputs', uniform_out)

# --- Part 2: score, scale, mask, normalize, mix ---
print('\nPart 2: exact-ratio attention')
scores = q @ k.T / math.sqrt(q.shape[-1])
print(f'  head width {q.shape[-1]}, divisor {math.sqrt(q.shape[-1]):.6f}')
matrix('scaled scores before mask', scores)
matrix('exp of final score row', scores[-1].exp().unsqueeze(0))
print(f'  final exponential sum {scores[-1].exp().sum().item():.6f}')
matrix('causal weights', weights)
matrix('outputs', out)
matrix('row sums', weights.sum(-1).unsqueeze(0))
print(f'  future weight maximum {weights.triu(1).abs().max().item():.6f}')

# --- Part 3: adversarial correctness tests, not model attacks ---
print('\nPart 3: change a future input and detect implementation mistakes')
changed = x.clone()
changed[-1, 2] += 2
changed[-1, 4:] += 10
later, _ = head(changed)
prefix_delta = (out[:-1] - later[:-1]).abs().max().item()
unmasked, _ = head(x, causal=False)
leaking, _ = head(changed, causal=False)
leak_delta = (unmasked[:-1] - leaking[:-1]).abs().max().item()
print(f'  causal prefix max change {prefix_delta:.6f}')
print(f'  unmasked prefix max change {leak_delta:.6f}')
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
wrong_zero = scores.masked_fill(~allowed, 0).softmax(-1)
wrong_after = scores.softmax(-1) * allowed
matrix('WRONG zero-score mask, first weight row', wrong_zero[:1])
matrix('WRONG mask after softmax, row sums', wrong_after.sum(-1).unsqueeze(0))
shifted = (scores + 1000).softmax(-1)
print(f'  stable softmax shift invariance {torch.allclose(shifted, scores.softmax(-1))}')
value_only = v.clone()
value_only[0, 0] += 7
changed_value, same_weights = attention(q, k, value_only)
print(f'  value-only change keeps weights {torch.equal(weights, same_weights)}')
print(f'  final output first-coordinate change {(changed_value[-1, 0] - out[-1, 0]).item():.6f}')

# --- Part 4: real nn.Linear parameters and an independent implementation ---
print('\nPart 4: random learnable projections, no semantic training')
random_head = SingleHead(6, 2).double()
random_x = torch.randn(2, 4, 6, dtype=torch.float64)
random_out, random_weights = random_head(random_x)
ref = F.scaled_dot_product_attention(random_head.query(random_x), random_head.key(random_x),
                                     random_head.value(random_x), dropout_p=0.0, is_causal=True)
error = (random_out - ref).abs().max().item()
print(f'  batch input {tuple(random_x.shape)}, output {tuple(random_out.shape)}, weights {tuple(random_weights.shape)}')
print(f'  trainable projection parameters {sum(p.numel() for p in random_head.parameters())}')
print(f'  PyTorch SDPA max absolute error {error:.3e}')
random_out.square().mean().backward()
for name, p in random_head.named_parameters():
    print(f'  {name} finite nonzero gradient {bool(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)}')

# --- Part 5: what the explicit score matrix costs ---
print('\nPart 5: one dense float32 score matrix, not whole-model memory')
for length in (128, 256):
    print(f'  T={length}: {length * length:,} entries, {length * length * 4:,} bytes')

# TRY THIS: Set all query vectors to zero. Predict the weights and output first.
# TRY THIS: Change only one value. Predict which weights and output coordinates change.
# TRY THIS: Remove the mask. Predict whether the prefix-invariance assertion survives.


def check():
    expected = torch.tensor([[1., 0, 0], [1/3, 2/3, 0], [1/7, 2/7, 4/7]], dtype=x.dtype)
    assert torch.allclose(weights, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out[-1], torch.tensor([26/7, 16/7], dtype=x.dtype))
    assert torch.allclose(weights.sum(-1), torch.ones(3, dtype=x.dtype))
    assert torch.count_nonzero(weights.triu(1)) == 0
    assert torch.equal(out[0], v[0])
    assert torch.allclose(uniform_out[-1], v.mean(0))
    assert prefix_delta == 0 and leak_delta > 1
    assert torch.count_nonzero(wrong_zero.triu(1)) > 0
    assert not torch.allclose(wrong_after.sum(-1), torch.ones(3, dtype=x.dtype))
    assert torch.equal(weights, same_weights)
    assert torch.allclose(changed_value[-1, 0] - out[-1, 0], torch.tensor(1., dtype=x.dtype))
    assert torch.allclose(shifted, scores.softmax(-1))
    assert torch.allclose(random_out, ref, atol=1e-12, rtol=1e-12)
    for p in random_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


check()
print('CHECK PASSED')
