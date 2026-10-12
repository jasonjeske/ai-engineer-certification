"""Lab: count a real GPT, then separate size, storage and compute.

Run from project root:
  env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/10-scaling-and-what-comes-next/lab.py
Or from labs/:
  env -u PYTHONPATH ../.venv/bin/python ../chapters/llm-fundamentals/10-scaling-and-what-comes-next/lab.py
Offline, seeded CPU work, expected under 10 seconds. GPT-2-shaped counting uses
meta tensors, NOT downloaded or trained GPT-2 weights. The final loss table is
an explicitly constructed teaching function, NOT an empirical scaling result.
"""
import os
import pathlib
import sys
from dataclasses import replace

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))
os.environ['TIKTOKEN_CACHE_DIR'] = str(LABS / 'data' / 'tiktoken-cache')

import tiktoken  # noqa: E402
import torch  # noqa: E402
from tutorlab.gpt import GPT, GPTConfig  # noqa: E402
from tutorlab.scaling_demo import (attention_score_elements, dense_train_flops,
                                  fp32_adam_bytes, illustrative_loss,
                                  kv_cache_bytes, parameter_counts, weight_bytes)  # noqa: E402
from tutorlab.tokenizers import CharTokenizer  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(1337)

# --- Part 1: count actual trainable tensors ------------------------------------
print('Part 1: exact unique parameters, not checkpoint dictionary entries')
toy = GPTConfig(vocab_size=9, block_size=6, n_layer=3, n_head=2, n_embd=8)
model = GPT(toy)
counts = parameter_counts(toy)
for name, n in counts['parts'].items():
    print(f'  {name}: {n:,}')
print(f"  leading estimate={counts['leading']:,} exact={counts['exact']:,} torch={model.num_params():,}")
print(f"  omitted biases and norms={counts['omitted_bias_and_norm']:,}")
x = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
y = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
logits, loss = model(x, y)
loss.backward()
connected = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in model.parameters())
print(f'  real forward logits={tuple(logits.shape)} finite gradients on every parameter={connected}')
untied = GPT(toy)
untied.head.weight = torch.nn.Parameter(untied.head.weight.detach().clone())
print(f'  tied={model.num_params():,} untied={untied.num_params():,} extra={untied.num_params() - model.num_params():,}')
mask_elements = sum(b.numel() for b in model.buffers())
print(f'  causal mask buffer elements={mask_elements} (not parameters)')

# --- Part 2: change one size knob at a time ------------------------------------
print('\nPart 2: one change at a time (same exact architecture)')
variants = [('base', toy), ('layers x2', replace(toy, n_layer=6)),
            ('width x2', replace(toy, n_embd=16)),
            ('vocab x2', replace(toy, vocab_size=18)),
            ('context x2', replace(toy, block_size=12)),
            ('heads x2', replace(toy, n_head=4))]
observed = {}
for name, cfg in variants:
    actual = GPT(cfg).num_params()
    observed[name] = actual
    assert actual == parameter_counts(cfg)['exact']
    print(f'  {name:12} params={actual:6,} ratio={actual / model.num_params():.3f}')

# --- Part 3: GPT-2 scale, no large allocation or weights download ---------------
print('\nPart 3: GPT-2 small shape, metadata only')
gpt2 = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
with torch.device('meta'):
    large = GPT(gpt2)
big_counts = parameter_counts(gpt2)
p = big_counts['exact']
print('  layers=12 width=768 heads=12 vocab=50,257 context=1,024')
print(f"  leading={big_counts['leading']:,} exact={p:,} meta module={large.num_params():,}")
print(f"  token table={big_counts['parts']['token_table_and_tied_head']:,} ({100 * gpt2.vocab_size * gpt2.n_embd / p:.2f}% of total)")
print(f'  untied head would add={gpt2.vocab_size * gpt2.n_embd:,}')
print(f'  weights only: fp32={weight_bytes(p, 4):,} bytes fp16/bf16={weight_bytes(p, 2):,} bytes')
print(f'  fp32 Adam arrays only={fp32_adam_bytes(p):,} bytes (not peak training memory)')

# --- Part 4: runtime memory is not a parameter count ---------------------------
print('\nPart 4: context costs (GPT-2 shape, batch=1)')
score_counts = []
cache_counts = []
for t in (512, 1024):
    score = attention_score_elements(1, gpt2.n_head, t)
    cache = kv_cache_bytes(gpt2, 1, t, 2)
    score_counts.append(score)
    cache_counts.append(cache)
    print(f'  tokens={t}: one-layer score elements={score:,}; hypothetical all-layer 2-byte KV cache={cache:,} bytes')
print('  score ratio=4.0; cache ratio=2.0; learned parameter count unchanged')
print('  tutorlab.generate recomputes context; it does NOT implement this KV cache')

# --- Part 5: a fixed text budget depends on the tokenizer ----------------------
print('\nPart 5: same text, different sequence lengths')
text = (LABS / 'data' / 'tinyshakespeare.txt').read_text()
chars = CharTokenizer(text)
bpe = tiktoken.get_encoding('gpt2')
sample = 'To be, or not to be, that is the question.'
char_ids, bpe_ids = chars.encode(sample), bpe.encode(sample)
print(f'  text={sample!r}')
print(f'  character vocab={chars.vocab_size:,}, tokens={len(char_ids)}')
print(f'  GPT-2 BPE vocab={bpe.n_vocab:,}, tokens={len(bpe_ids)}')
print(f'  at width=768: character table={chars.vocab_size * 768:,}; BPE table={bpe.n_vocab * 768:,}')

# --- Part 6: fixed compute and an explicitly illustrative objective ------------
print('\nPart 6: budget accounting, not a trained scaling-law experiment')
for n, d in ((1_000_000, 100_000_000), (2_000_000, 50_000_000), (2_000_000, 200_000_000)):
    print(f'  N={n:,} D={d:,}: rough 6ND={dense_train_flops(n, d):,} FLOPs')
print('  constructed objective: 1.5 + 1/sqrt(n) + 1/sqrt(d); n*d=16')
allocations = []
for n in (1, 2, 4, 8, 16):
    d = 16 / n
    value = illustrative_loss(n, d)
    allocations.append((n, d, value))
    print(f'  model units={n:2} data units={d:4.1f} illustrative loss={value:.4f}')
best = min(allocations, key=lambda row: row[2])
print(f'  best tested allocation: n={best[0]}, d={best[1]:.1f}; this is NOT a fitted empirical law')

# TRY THIS: Set heads to 1, 2, 4, 8 at width 8. Predict the parameter count.
# TRY THIS: Compare all variants with tied=False. Predict which term is added.
# TRY THIS: Change the constructed model term to 2/sqrt(n). Predict the shift
# in the best fixed-product allocation. Do not call it a measured loss curve.


def check():
    assert counts['exact'] == model.num_params() == 2752
    assert counts['leading'] == 2424
    assert model.head.weight is model.tok_emb.weight
    assert untied.head.weight is not untied.tok_emb.weight
    assert untied.num_params() - model.num_params() == toy.vocab_size * toy.n_embd
    assert parameter_counts(toy, tied=False)['exact'] == untied.num_params()
    assert connected and logits.shape == (1, 6, 9)
    assert mask_elements == toy.n_layer * toy.block_size ** 2
    assert observed['heads x2'] == observed['base']
    assert observed['width x2'] > observed['layers x2'] > observed['base']
    assert large.num_params() == p == 124439808
    assert all(t.is_meta for t in large.parameters())
    assert score_counts[1] == 4 * score_counts[0]
    assert cache_counts[1] == 2 * cache_counts[0]
    assert bpe.decode(bpe_ids) == chars.decode(char_ids) == sample
    assert len(bpe_ids) < len(char_ids)
    assert dense_train_flops(1_000_000, 100_000_000) == dense_train_flops(2_000_000, 50_000_000)
    assert best[:2] == (4, 4.0)


check()
print('\nCHECK PASSED')
