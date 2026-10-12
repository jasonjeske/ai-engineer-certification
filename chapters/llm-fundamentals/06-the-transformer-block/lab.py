"""Lab: build and inspect a pre-norm transformer block, not a trained LLM.

Goal: verify residuals, feature-only LayerNorm, row-local MLP and causal stacking.
Run from project root: env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/06-the-transformer-block/lab.py
Or: cd labs && ../.venv/bin/python ../chapters/llm-fundamentals/06-the-transformer-block/lab.py
The resolved path works from any directory. Offline CPU run expected under 10 s.
"""
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'labs'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tutorlab.block_demo import InspectableBlock, layer_norm_parts, scalar_path_gradient  # noqa: E402
from tutorlab.gpt import Block, GPTConfig  # noqa: E402

torch.manual_seed(6)
torch.set_num_threads(1)
DTYPE = torch.float64


def values(x):
    return [round(v, 6) for v in x.detach().flatten().tolist()]


# --- Part 1: add a correction instead of replacing the state -------------------
print('Part 1: residual arithmetic and a controlled gradient chain')
state = torch.tensor([1., 3.], dtype=DTYPE)
update = torch.tensor([0.5, -0.5], dtype=DTYPE)
print('  state:', values(state), 'update:', values(update), 'sum:', values(state + update))
plain_grad = scalar_path_gradient(8, 0.5, False)
residual_grad = scalar_path_gradient(8, 0.5, True)
cancel_grad = scalar_path_gradient(8, -1.0, True)
print(f'  8 scalar layers, branch scale 0.5: plain gradient={plain_grad:.8f}, residual gradient={residual_grad:.8f}')
print(f'  residual branch scale -1: gradient={cancel_grad:.1f} (cancellation is possible)')

# --- Part 2: normalize features, not tokens -----------------------------------
print('\nPart 2: layer normalization by hand')
row = torch.tensor([1., 3., 5., 7.], dtype=DTYPE)
z, mean, variance = layer_norm_parts(row)
reference_norm = F.layer_norm(row, (4,), eps=1e-5)
affine = 2 * z + 3
constant, _, _ = layer_norm_parts(torch.full((4,), 2., dtype=DTYPE))
print('  row:', values(row), 'centered:', values(row - mean))
print(f'  mean={mean.item():.6f}, population variance={variance.item():.6f}, eps=0.00001')
print('  normalized:', values(z))
print(f'  normalized mean={z.mean().item():.6f}, variance={z.var(unbiased=False).item():.6f}')
print('  gamma=2, beta=3:', values(affine))
print(f'  affine mean={affine.mean().item():.6f}, variance={affine.var(unbiased=False).item():.6f}')
print('  constant row normalized:', values(constant))

# --- Part 3: build the block and expose every intermediate --------------------
print('\nPart 3: one block, with two heads and a fourfold MLP expansion')
cfg = GPTConfig(block_size=4, n_embd=8, n_head=2, dropout=0.0)
block = InspectableBlock(cfg).double().eval()
x = torch.randn(2, 4, 8, dtype=DTYPE)
trace = block.trace(x)
reference = Block(cfg).double().eval()
reference.load_state_dict(block.state_dict())
parity_error = (trace['output'] - reference(x)).abs().max().item()
print(f'  input={tuple(x.shape)}, heads={cfg.n_head}, head width={cfg.n_embd // cfg.n_head}')
for name, tensor in trace.items():
    print(f'  {name:17} shape={tuple(tensor.shape)} rms={tensor.square().mean().sqrt().item():.6f}')
counts = {name: sum(p.numel() for p in module.parameters()) for name, module in
          [('attention', block.attn), ('MLP', block.mlp), ('norm1', block.ln1), ('norm2', block.ln2)]}
print('  parameters:', counts, 'total=', sum(counts.values()))
print(f'  existing tutorlab.gpt.Block max error={parity_error:.6f}')

# --- Part 4: the MLP transforms each row with the same weights ----------------
print('\nPart 4: position-wise nonlinear computation')
u = trace['norm2'].detach()
hidden = block.mlp.net[0](u)
activated = block.mlp.net[1](hidden)
changed_u = u.clone()
changed_u[:, -1, 0] += 10
mlp_error = (block.mlp(u)[:, :-1] - block.mlp(changed_u)[:, :-1]).abs().max().item()
print(f'  MLP shapes: {tuple(u.shape)} -> {tuple(hidden.shape)} -> {tuple(block.mlp(u).shape)}')
print('  GELU([-1, 0, 1]):', values(F.gelu(torch.tensor([-1., 0., 1.], dtype=DTYPE))))
print(f'  changing last row: earlier MLP rows max difference={mlp_error:.6f}')

# --- Part 5: causal stacking and a deliberately wrong normalization -----------
print('\nPart 5: a future-change test and a negative control')
stack = torch.nn.Sequential(block, InspectableBlock(cfg).double().eval())
changed = x.clone()
changed[:, -1, 0] += 10
out, changed_out = stack(x), stack(changed)
prefix_error = (out[:, :-1] - changed_out[:, :-1]).abs().max().item()
future_change = (out[:, -1] - changed_out[:, -1]).abs().max().item()
wrong_norm = F.layer_norm(x, (4, 8))
wrong_changed = F.layer_norm(changed, (4, 8))
wrong_error = (wrong_norm[:, :-1] - wrong_changed[:, :-1]).abs().max().item()
print(f'  two blocks output shape={tuple(out.shape)}')
print(f'  correct block stack earlier rows max difference={prefix_error:.6f}')
print(f'  changed final row max difference={future_change:.6f}')
print(f'  WRONG norm over (tokens, features): earlier rows max difference={wrong_error:.6f}')

# --- Part 6: identity path and backward pass ----------------------------------
print('\nPart 6: zero corrections, then a real backward pass')
identity = copy.deepcopy(block)
with torch.no_grad():
    for layer in (identity.attn.proj, identity.mlp.net[2]):
        layer.weight.zero_()
        layer.bias.zero_()
identity_x = x.detach().clone().requires_grad_()
identity_out = identity(identity_x)
identity_out.sum().backward()
identity_error = (identity_out - identity_x).abs().max().item()
identity_grad_error = (identity_x.grad - 1).abs().max().item()
print(f'  zero branch outputs: identity error={identity_error:.6f}, gradient-to-ones error={identity_grad_error:.6f}')
probe_x = x.detach().clone().requires_grad_()
loss = block(probe_x).square().mean()  # diagnostic scalar, NOT a next-token training loss
loss.backward()
finite_grads = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in block.parameters())
print(f'  diagnostic mean-square loss={loss.item():.6f}, input gradient norm={probe_x.grad.norm().item():.6f}')
print(f'  every parameter gradient present and finite={finite_grads}')

# TRY THIS: Change gamma and beta above. Predict the normalized and affine means separately.
# TRY THIS: Change the future perturbation to an earlier row. Which later rows may change?
# TRY THIS: In a copy, remove one residual addition. Predict whether the zero-branch identity check survives.


def check():
    assert torch.equal(state + update, torch.tensor([1.5, 2.5], dtype=DTYPE))
    assert plain_grad == 0.5 ** 8 and residual_grad == 1.5 ** 8
    assert cancel_grad == 0.0
    assert torch.allclose(z, reference_norm, atol=1e-12, rtol=1e-12)
    assert torch.equal(constant, torch.zeros_like(constant))
    assert abs(affine.mean().item() - 3) < 1e-12
    assert parity_error < 1e-12 and trace['output'].shape == x.shape
    assert torch.allclose(trace['norm2'], block.ln2(trace['after_attention']))
    assert mlp_error == 0.0
    assert prefix_error == 0.0 and future_change > 1.0
    assert wrong_error > 0.1, 'negative control must detect future leakage'
    assert identity_error == 0.0 and identity_grad_error == 0.0
    assert finite_grads and torch.isfinite(probe_x.grad).all()
    assert block.attn.qkv.weight.grad.abs().sum() > 0
    assert block.mlp.net[0].weight.grad.abs().sum() > 0


check()
print('\nCHECK PASSED')
