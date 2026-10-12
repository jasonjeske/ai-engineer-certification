"""Lab: Generation, sampling text from your model.

Goal: inspect temperature/top-k/top-p, then decode the real local Shakespeare GPT.
Run from root: env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/09-generation/lab.py
Or: cd labs && ../.venv/bin/python ../chapters/llm-fundamentals/09-generation/lab.py
Paths are resolved from this file, so any current directory works.
Expected runtime: under 60 seconds on CPU; offline, no training or downloads.
"""
import pathlib
import sys
from types import SimpleNamespace

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))

import torch  # noqa: E402

from tutorlab.sampling_demo import entropy_bits, generate, probabilities  # noqa: E402
from tutorlab.train import load  # noqa: E402

# CPU, one thread and local generators make repeat runs inspectable.
torch.set_num_threads(1)
torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)


def fmt(p):
    return '[' + ', '.join(f'{x:.6f}' for x in p.tolist()) + ']'


# --- Part 1: a distribution you can compute by hand ---
print('Part 1: logits are log weights for tokens A, B, C, D')
weights = torch.tensor([4., 2., 1., 1.], dtype=torch.float64)
logits = weights.log()
print(f'  weights={weights.tolist()} sum={weights.sum().item():.0f}')
print(f'  logits={fmt(logits)}')
sweep = {}
for temp in (.5, 1., 2.):
    p = probabilities(logits, temperature=temp)
    sweep[temp] = p
    print(f'  T={temp:.1f} p={fmt(p)} entropy_bits={entropy_bits(p).item():.6f}')
print(f'  T=0.0 p={fmt(probabilities(logits, temperature=0))} (greedy)')
print(f'  T=0.5 unnormalized={weights.square().tolist()} sum={weights.square().sum().item():.0f}')
print(f'  T=2.0 unnormalized={fmt(weights.sqrt())} sum={weights.sqrt().sum().item():.6f}')

# --- Part 2: support restriction and categorical draws ---
print('\nPart 2: filter then renormalize')
k2 = probabilities(logits, top_k=2)
p7 = probabilities(logits, top_p=.7)
print(f'  top_k=2 p={fmt(k2)}')
print(f'  top_p=0.7 p={fmt(p7)} original_cumulative={fmt(sweep[1.].cumsum(0))}')
flat = probabilities(torch.zeros(4), top_p=.7)
print(f'  flat top_p=0.7 p={fmt(flat)} support={(flat > 0).sum().item()}')
tied = probabilities(torch.zeros(4), top_k=2)
print(f'  tied top_k=2 p={fmt(tied)} (lower IDs win ties)')
order_logits = torch.tensor([.6, .25, .15], dtype=torch.float64).log()
print(f'  order demo base={fmt(probabilities(order_logits))}')
print(f'  top_p=0.7 alone={fmt(probabilities(order_logits, top_p=.7))}')
print(f'  top_k=2 THEN top_p=0.7={fmt(probabilities(order_logits, top_k=2, top_p=.7))}')
rng = torch.Generator().manual_seed(1337)
draws = torch.multinomial(sweep[1.], 20000, replacement=True, generator=rng)
counts = torch.bincount(draws, minlength=4)
print(f'  20000 independent draws counts={counts.tolist()} frequencies={fmt(counts / 20000)}')
filtered_draws = torch.multinomial(k2, 1000, replacement=True, generator=torch.Generator().manual_seed(7))
print(f'  top_k=2 sampled IDs={sorted(set(filtered_draws.tolist()))}')

# --- Part 3: local versus sequence probability ---
print('\nPart 3: a constructed two-step tree, not a model quality score')
print(f'  first A=0.600 B=0.400; best continuation after A=0.510, after B=0.990')
greedy_path, other_path = .6 * .51, .4 * .99
print(f'  greedy path probability={greedy_path:.3f}; alternative={other_path:.3f}')

# --- Part 4: real checkpoint, actual last-position logits ---
print('\nPart 4: local trained character GPT (not a chat model)')
model, tok = load(dev='cpu')
model.eval()
original = {name: p.detach().clone() for name, p in model.named_parameters()}
prompt = torch.tensor([tok.encode('ROMEO:\n')], dtype=torch.long)
with torch.inference_mode():
    all_logits, _ = model(prompt)
    last = all_logits[:, -1, :]
    next_p = probabilities(last)[0]
print(f'  torch={torch.__version__} device=cpu threads={torch.get_num_threads()}')
print(f'  parameters={model.num_params()} vocabulary={tok.vocab_size} block_size={model.cfg.block_size}')
print(f'  prompt={tok.decode(prompt[0].tolist())!r} ids={prompt[0].tolist()}')
print(f'  logits shape={tuple(all_logits.shape)} last shape={tuple(last.shape)}')
for idx in torch.argsort(next_p, descending=True)[:5].tolist():
    print(f'  next character={tok.decode([idx])!r} probability={next_p[idx].item():.6f}')

# --- Part 5: freeze weights, change decoding only ---
print('\nPart 5: same checkpoint and prompt, 96 new characters, seed=1337')
settings = [
    ('greedy', dict(temperature=0)),
    ('cool', dict(temperature=.5)),
    ('warm', dict(temperature=1.2)),
    ('nucleus', dict(temperature=.8, top_k=20, top_p=.9)),
]
results = {}
for name, options in settings:
    result = generate(model, prompt, 96, seed=1337, **options)
    results[name] = result
    print(f'  {name}: {tok.decode(result.ids[0].tolist())!r}')
    print(f'    total_tokens={result.ids.size(1)} stop={result.stop_reason}')
repeat = generate(model, prompt, 96, temperature=.8, top_k=20, top_p=.9, seed=1337)
top1 = generate(model, prompt, 96, top_k=1, seed=42)
long_prompt = torch.tensor([tok.encode('\n' * (model.cfg.block_size + 2))])
cropped = generate(model, long_prompt, 2, temperature=0)
print(f'  repeated nucleus identical={torch.equal(repeat.ids, results["nucleus"].ids)}')
print(f'  top_k=1 equals greedy={torch.equal(top1.ids, results["greedy"].ids)}')
print(f'  long prompt length={long_prompt.size(1)} context_lengths={cropped.context_lengths} returned_length={cropped.ids.size(1)}')

# --- Part 6: stopping with an explicit, untrained fixture ---
class StopFixture(torch.nn.Module):
    """Always emits ID 2, standing in for an EOS-trained model interface."""
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(vocab_size=3, block_size=4)

    def forward(self, idx):
        out = torch.full((*idx.shape, 3), float('-inf'))
        out[..., 2] = 0.
        return out, None


stop = generate(StopFixture().eval(), torch.tensor([[0]]), 10, eos_id=2)
print('\nPart 6: EOS behavior with a labelled fixture, not the Shakespeare checkpoint')
print(f'  requested_new=10 generated_new={len(stop.context_lengths)} ids={stop.ids.tolist()} stop={stop.stop_reason}')
print('  Shakespeare has no special EOS token; its runs stop at the token budget.')
unchanged = all(torch.equal(p, original[n]) for n, p in model.named_parameters())
print(f'  model parameters unchanged={unchanged}; gradients absent={all(p.grad is None for p in model.parameters())}')

# TRY THIS: Set top_k=1 for cool and warm. Predict whether temperature still matters.
# TRY THIS: Change top_p=.7 to .5 in Part 2. Predict the surviving IDs before running.
# TRY THIS: Change seed=1337 in a sampled run. Predict what stays fixed: weights or text?


def check():
    assert torch.allclose(sweep[1.], weights / weights.sum())
    assert torch.allclose(sweep[.5], weights.square() / weights.square().sum())
    assert entropy_bits(sweep[.5]) < entropy_bits(sweep[1.]) < entropy_bits(sweep[2.])
    assert torch.allclose(probabilities(logits + 1000), sweep[1.])
    assert torch.allclose(k2, torch.tensor([2/3, 1/3, 0., 0.], dtype=torch.float64))
    assert torch.allclose(k2, p7)
    assert (filtered_draws < 2).all()
    assert (flat > 0).sum() == 3
    assert torch.max(torch.abs(counts / 20000 - sweep[1.])) < .02
    assert greedy_path < other_path
    assert all_logits.shape == (1, prompt.size(1), tok.vocab_size)
    for result in results.values():
        assert result.ids.size(1) == prompt.size(1) + 96
        assert torch.equal(result.ids[:, :prompt.size(1)], prompt)
        assert ((result.ids >= 0) & (result.ids < tok.vocab_size)).all()
    assert torch.equal(repeat.ids, results['nucleus'].ids)
    assert torch.equal(top1.ids, results['greedy'].ids)
    assert cropped.context_lengths == [model.cfg.block_size] * 2
    assert cropped.ids.size(1) == long_prompt.size(1) + 2
    assert stop.ids.tolist() == [[0, 2]] and stop.stop_reason == 'eos'
    assert unchanged and all(p.grad is None for p in model.parameters())


check()
print('\nCHECK PASSED')
