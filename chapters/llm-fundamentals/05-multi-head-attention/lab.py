"""Lab: independent attention routes, concatenation, and output mixing.

Goal: calculate two heads by hand, then prove loop/packed/PyTorch equivalence.
Run from root: env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/05-multi-head-attention/lab.py
Or: cd labs && ../.venv/bin/python ../chapters/llm-fundamentals/05-multi-head-attention/lab.py
Works from any directory. Offline CPU, seeded, expected runtime under 10 seconds.
"""
import pathlib
import sys

import torch

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))
from tutorlab.multihead_demo import MultiHead, worked_example  # noqa: E402

torch.manual_seed(5)
torch.set_num_threads(1)


def vector(label, values):
    print(f'  {label} [' + ', '.join(f'{v:.6f}' for v in values.detach().tolist()) + ']')


# --- Part 1: two views of the same full input ---
print('Part 1: hand-chosen projections, not trained semantic heads')
x, model = worked_example()
out, weights, cat = model(x)
print(f'  input {tuple(x.shape)}, heads {model.num_heads}, head width {model.head_width}')
print(f'  each query projection weight {tuple(model.heads[0].query.weight.shape)}')
print('  X rows [constant query, log-key, value first, value second]')
for row in x[0]:
    vector('X', row)
for index, head in enumerate(model.heads):
    scores = head.query(x) @ head.key(x).transpose(-2, -1) / model.head_width ** 0.5
    vector(f'head {index} final scores', scores[0, -1])
    vector(f'head {index} final weights', weights[0, index, -1])
    vector(f'head {index} final output', cat[0, -1, index * 2:(index + 1) * 2])

# --- Part 2: concatenate, then mix features within each position ---
print('\nPart 2: concatenate and output-project')
print(f'  weights {tuple(weights.shape)}, concatenation {tuple(cat.shape)}, output {tuple(out.shape)}')
vector('final concatenation', cat[0, -1])
vector('final projected output', out[0, -1])
print('  output projection stored rows')
for row in model.proj.weight:
    vector('W', row)
summed = cat[..., :2] + cat[..., 2:]
vector('sum instead of concat loses separate routes', summed[0, -1])
ablated = cat.clone()
ablated[..., 2:] = 0
contribution = out - model.proj(ablated)
vector('head 1 contribution after output projection', contribution[0, -1])

# --- Part 3: interventions and invariants ---
print('\nPart 3: test information flow, not only shapes')
changed = x.clone()
changed[:, -1, 1:] += 3
later = model(changed)[0]
causal_change = (out[:, :-1] - later[:, :-1]).abs().max().item()
noncausal_change = (model(x, causal=False)[0][:, :-1] -
                    model(changed, causal=False)[0][:, :-1]).abs().max().item()
print(f'  causal prefix max change {causal_change:.6f}')
print(f'  unmasked prefix changes {noncausal_change > 0.0}')
print(f'  future nonzero weights {torch.count_nonzero(weights.triu(1)).item()}')
print(f'  normalized rows {torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))}')
zero_query = x.clone()
zero_query[..., 0] = 0
uniform_out, uniform_weights, uniform_cat = model(zero_query)
vector('zero-query final weights, both heads', uniform_weights[0, 0, -1])
vector('zero-query final concatenation', uniform_cat[0, -1])
vector('zero-query final projected output', uniform_out[0, -1])

# --- Part 4: packed implementation and independent library reference ---
print('\nPart 4: trainable random projections and numerical parity')
random_model = MultiHead(8, 2).double()
batch = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
loop, loop_weights, loop_cat = random_model(batch)
packed, packed_weights, packed_cat = random_model.packed_forward(batch)
reference = torch.nn.MultiheadAttention(8, 2, bias=False, dropout=0.0,
                                        batch_first=True, dtype=torch.float64)
with torch.no_grad():
    reference.in_proj_weight.copy_(random_model.packed_weights())
    reference.out_proj.weight.copy_(random_model.proj.weight)
# MultiheadAttention Boolean masks: True means BLOCKED, unlike SDPA.
blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)
ref, ref_weights = reference(batch, batch, batch, attn_mask=blocked,
                             need_weights=True, average_attn_weights=False)
print(f'  random input {tuple(batch.shape)}, per-head weights {tuple(loop_weights.shape)}')
print(f'  loop matches packed {torch.allclose(loop, packed, atol=1e-12, rtol=1e-12)}')
print(f'  loop matches PyTorch {torch.allclose(loop, ref, atol=1e-12, rtol=1e-12)}')
print(f'  per-head weights match PyTorch {torch.allclose(loop_weights, ref_weights, atol=1e-12, rtol=1e-12)}')
loop.square().mean().backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() and
              p.grad.abs().sum() > 0 for p in random_model.parameters())
print(f'  all projection tensors have finite nonzero gradients {grad_ok}')

# --- Part 5: parameters versus explicit score storage ---
print('\nPart 5: fixed C=8, B=2, T=4, no biases')
counts = []
for heads in (1, 2, 4):
    m = MultiHead(8, heads)
    count = sum(p.numel() for p in m.parameters())
    counts.append(count)
    score_entries = 2 * heads * 4 * 4
    print(f'  heads {heads}, head width {8 // heads}, parameters {count}, score entries {score_entries}')
print('  one float32 score tensor, B=2, H=2')
for tokens in (128, 256):
    entries = 2 * 2 * tokens * tokens
    print(f'  T={tokens}, entries {entries:,}, bytes {entries * 4:,}')

# TRY THIS: set the query feature to zero and predict both final mixtures.
# TRY THIS: zero the second head's concatenated block, then predict its signed contribution.
# TRY THIS: choose four heads at fixed width eight; predict parameters and score storage.


def check():
    expected_weights = torch.tensor([[1/7, 2/7, 4/7], [4/7, 2/7, 1/7]], dtype=x.dtype)
    assert torch.allclose(weights[0, :, -1], expected_weights)
    assert torch.allclose(cat[0, -1], torch.tensor([26/7, 16/7, 14/7, 10/7], dtype=x.dtype))
    assert torch.allclose(out[0, -1], torch.tensor([40/7, 6/7, 12/7, 26/7], dtype=x.dtype))
    assert out.shape == x.shape and weights.shape == (1, 2, 3, 3)
    assert torch.count_nonzero(weights.triu(1)) == 0
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert causal_change == 0.0 and noncausal_change > 0.0
    assert torch.allclose(uniform_weights[0, :, -1], torch.full((2, 3), 1/3, dtype=x.dtype))
    assert torch.allclose(uniform_out[0, -1], torch.tensor([16/3, 0, 0, 4], dtype=x.dtype))
    assert torch.allclose(contribution[0, -1], torch.tensor([2, -10/7, -2, 10/7], dtype=x.dtype))
    assert torch.allclose(loop, packed, atol=1e-12, rtol=1e-12)
    assert torch.allclose(loop, ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(loop_weights, ref_weights, atol=1e-12, rtol=1e-12)
    assert torch.allclose(loop_weights, packed_weights) and torch.allclose(loop_cat, packed_cat)
    assert grad_ok and counts == [256, 256, 256]


check()
print('\nCHECK PASSED')
