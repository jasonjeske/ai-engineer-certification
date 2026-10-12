"""Goal: trace one gradient update, train a neural bigram, then train a tiny GPT.

Run from project root:
  env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/08-the-training-loop/lab.py
Or from labs/: ../.venv/bin/python ../chapters/llm-fundamentals/08-the-training-loop/lab.py
Offline CPU lab, expected runtime under 10 seconds. No pretrained weights.
Toy sentences intentionally share patterns: held-out results are not language benchmarks.
"""
import math
import pathlib
import sys

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from tutorlab.gpt import GPT, GPTConfig  # noqa: E402
from tutorlab.training_loop_demo import (  # noqa: E402
    Bigram, evaluate, mean_ce_logit_gradient, shifted_batch, train_step,
)

torch.set_num_threads(1)
torch.manual_seed(808)
torch.use_deterministic_algorithms(True)

# --- Part 1: follow one update by hand -----------------------------------------
print('Part 1: one example, raw logits -> loss -> gradient -> update')
z = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
target = torch.tensor([1])
loss = F.cross_entropy(z, target)
loss.backward()
analytic = mean_ce_logit_gradient(z.detach(), target)
grad = z.grad.clone()
updated = z.detach() - 0.3 * grad
post = F.cross_entropy(updated, target)
eps = 1e-6
plus, minus = z.detach().clone(), z.detach().clone()
plus[0, 1] += eps
minus[0, 1] -= eps
finite_diff = ((F.cross_entropy(plus, target) - F.cross_entropy(minus, target)) / (2 * eps)).item()
print(f'  logits={z.detach().tolist()} target={target.tolist()} lr=0.3')
print(f'  probability={z.detach().softmax(-1).tolist()} loss={loss.item():.6f}')
print(f'  gradient={[round(v, 6) for v in grad[0].tolist()]}')
print(f'  updated logits={[round(v, 6) for v in updated[0].tolist()]}')
print(f'  target probability after={updated.softmax(-1)[0, 1].item():.6f} loss after={post.item():.6f}')
print(f'  finite-difference target gradient={finite_diff:.6f}')

# --- Part 2: backward does not update weights; gradients accumulate ------------
print('\nPart 2: gradient buffers are not weights')
w = torch.nn.Parameter(torch.zeros(1, 3, dtype=torch.double))
opt = torch.optim.SGD([w], lr=0.3)
original = w.detach().clone()
F.cross_entropy(w, target).backward()
first_grad = w.grad.clone()
unchanged = torch.equal(w, original)
F.cross_entropy(w, target).backward()  # fresh graph, unchanged weights, no reset
second_grad = w.grad.clone()
opt.zero_grad(set_to_none=True)
cleared = w.grad is None
F.cross_entropy(w, target).backward()
opt.step()
print(f'  weights unchanged by backward={unchanged}')
print(f'  two backwards without reset: gradient ratio={(second_grad / first_grad)[0].tolist()}')
print(f'  reset set grad to None={cleared}; one clean SGD step matches hand update={torch.allclose(w, updated)}')

# --- Part 3: align data and inspect the bigram optimum -------------------------
print('\nPart 3: adjacent characters become supervised examples')
TRAIN = 'a cat sat. a cat ran. a rat ran. a rat sat.'
VAL = 'a rat sat. a cat ran.'
VOCAB = ''.join(sorted(set(TRAIN)))
stoi = {c: i for i, c in enumerate(VOCAB)}

def encode(text):
    return torch.tensor([[stoi[c] for c in text]], dtype=torch.long)

x, y = shifted_batch(encode(TRAIN))
vx, vy = shifted_batch(encode(VAL))
counts = torch.bincount((x * len(VOCAB) + y).flatten(), minlength=len(VOCAB) ** 2).reshape(len(VOCAB), -1).double()
row_totals = counts.sum(1, keepdim=True)
empirical = counts / row_totals.clamp_min(1)
observed = counts > 0
floor = -(counts[observed] * empirical[observed].log()).sum().item() / y.numel()
print(f'  vocabulary={VOCAB!r}; V={len(VOCAB)}; train targets={y.numel()}; validation targets={vy.numel()}')
print(f'  first input IDs={x[0, :8].tolist()}; first target IDs={y[0, :8].tolist()}')
print(f'  empirical bigram training-loss infimum={floor:.6f}')
print('  validation is a separate toy string with shared phrases, not independent language evidence')

# --- Part 4: train the neural bigram ------------------------------------------
print('\nPart 4: learn a probability table instead of counting it')
bigram = Bigram(len(VOCAB)).double()
bopt = torch.optim.AdamW(bigram.parameters(), lr=0.1, weight_decay=0.0)
bigram_history = []
for step in range(201):
    if step in (0, 1, 10, 50, 200):
        row = (step, evaluate(bigram, x, y), evaluate(bigram, vx, vy))
        bigram_history.append(row)
        print(f'  after {step:3d} updates: train={row[1]:.6f} validation={row[2]:.6f}')
    if step < 200:
        train_step(bigram, bopt, x, y)
learned = bigram.table.weight.detach().softmax(-1)
a = stoi['a']
print(f'  successors of a (empirical / learned): ' + ' '.join(f'{c!r}:{empirical[a,j]:.4f}/{learned[a,j]:.4f}' for j,c in enumerate(VOCAB)))
print(f'  max row probability error={float((learned - empirical).abs().max()):.6f}')

# --- Part 5: fit is not generalization ----------------------------------------
print('\nPart 5: controlled mismatch, not a natural-language benchmark')
fit = Bigram(2).double()
fopt = torch.optim.SGD(fit.parameters(), lr=1.0)
fx, fy = shifted_batch(torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0]]))
sx, sy = shifted_batch(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]))
mismatch_before = (evaluate(fit, fx, fy), evaluate(fit, sx, sy))
for _ in range(40):
    train_step(fit, fopt, fx, fy)
mismatch_after = (evaluate(fit, fx, fy), evaluate(fit, sx, sy))
print(f'  before: alternating train={mismatch_before[0]:.6f} repeated-zero stress={mismatch_before[1]:.6f}')
print(f'  after:  alternating train={mismatch_after[0]:.6f} repeated-zero stress={mismatch_after[1]:.6f}')

# --- Part 6: the same loop trains the assembled GPT ----------------------------
print('\nPart 6: replace the table with the shared GPT')
gx, gy = shifted_batch(torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2], [1, 2, 0, 1, 2, 0, 1, 2, 0]]))
gpt = GPT(GPTConfig(vocab_size=3, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
gopt = torch.optim.AdamW(gpt.parameters(), lr=0.01, weight_decay=0.0)
gpt_before = evaluate(gpt, gx, gy)
attn_before = gpt.blocks[0].attn.qkv.weight.detach().clone()
for _ in range(80):
    train_step(gpt, gopt, gx, gy)
gpt_after = evaluate(gpt, gx, gy)
attn_change = (gpt.blocks[0].attn.qkv.weight.detach() - attn_before).abs().max().item()
weights_before_eval = {k: v.detach().clone() for k, v in gpt.state_dict().items()}
mode_before_eval = gpt.training
evaluate(gpt, gx, gy)
eval_unchanged = all(torch.equal(v, weights_before_eval[k]) for k, v in gpt.state_dict().items())
print(f'  shapes: inputs={tuple(gx.shape)} logits={tuple(gpt(gx)[0].shape)} targets={tuple(gy.shape)}')
print(f'  params={gpt.num_params()}; updates=80; training loss {gpt_before:.6f} -> {gpt_after:.6f}')
print(f'  attention weight max change={attn_change:.6f}')
print(f'  evaluation preserved state={eval_unchanged}; restored train mode={gpt.training == mode_before_eval}')
print('  tiny GPT result is a memorization smoke test, not a held-out score')

# TRY THIS: Set the bigram learning rate to 0.0. Predict its final training loss.
# TRY THIS: Remove the gradient reset in Part 2. Predict the SGD weight change.
# TRY THIS: Reverse the stress string's pattern. Predict which transition is rewarded.


def check():
    assert torch.allclose(grad, analytic, atol=1e-12)
    assert abs(finite_diff - grad[0, 1].item()) < 1e-8
    assert post.item() < loss.item()
    assert unchanged and cleared
    assert torch.allclose(second_grad, 2 * first_grad)
    assert torch.allclose(w.detach(), updated)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert abs(bigram_history[0][1] - math.log(len(VOCAB))) < 1e-12
    assert floor <= bigram_history[-1][1] < floor + 0.01
    assert (learned - empirical).abs().max().item() < 0.01
    assert bigram_history[-1][2] < bigram_history[0][2]
    assert mismatch_after[0] < mismatch_before[0]
    assert mismatch_after[1] > mismatch_before[1]
    assert gpt_after < 0.05 and gpt_after < gpt_before
    assert attn_change > 0
    assert eval_unchanged and gpt.training == mode_before_eval


check()
print('CHECK PASSED')
