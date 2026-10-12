"""Lab: The bigram model, your first real predictor.

Goal: count adjacent tokens, smooth rows, sample, and measure held-out loss.
Run from project root:
  env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/03-the-bigram-model/lab.py
Or from labs/:
  env -u PYTHONPATH ../.venv/bin/python ../chapters/llm-fundamentals/03-the-bigram-model/lab.py
Offline; expected runtime under 20 seconds on CPU. No files are written.
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
LABS = ROOT / 'labs'
sys.path.insert(0, str(LABS))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tutorlab.bigram_counts import count_pairs, generate, mean_nll, probabilities  # noqa: E402
from tutorlab.tokenizers import CharTokenizer  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(7)

# --- Part 1: count a string you can audit by hand ---
toy_text = 'abacaba'
toy_tok = CharTokenizer(toy_text)
toy_ids = toy_tok.encode(toy_text)
toy_counts = count_pairs([toy_ids], toy_tok.vocab_size)
# Explicit reference loop: the same counting operation without vectorization.
manual_counts = torch.zeros_like(toy_counts)
for current, following in zip(toy_ids, toy_ids[1:]):
    manual_counts[current, following] += 1
print('Part 1: count adjacent pairs')
print(f'  text={toy_text!r} ids={toy_ids} vocabulary={toy_tok.vocab}')
print(f'  tokens={len(toy_ids)} transitions={int(toy_counts.sum())}')
for char, row in zip(toy_tok.vocab, toy_counts):
    print(f'  counts after {char}: {[int(x) for x in row]}')

# --- Part 2: turn counts into conditional probabilities ---
raw = probabilities(toy_counts, alpha=0)
smoothed = probabilities(toy_counts, alpha=1)
print('\nPart 2: normalize rows, not columns')
for char, row in zip(toy_tok.vocab, smoothed):
    print(f'  add-one P(next | {char}): {[round(x, 6) for x in row.tolist()]}')
print(f'  raw P(b | a)={raw[0, 1]:.6f}; add-one={smoothed[0, 1]:.6f}')
print(f'  smoothed row sums: {smoothed.sum(1).tolist()}')
print(f'  denominator for a: {int(toy_counts[0].sum())} + {toy_tok.vocab_size} = {int(toy_counts[0].sum()) + toy_tok.vocab_size}')
print(f'  unseen context with add-one: {probabilities(torch.zeros(3, 3))[0].tolist()}')

# --- Part 3: score the actual next tokens ---
aba = toy_tok.encode('aba')
abc = toy_tok.encode('abc')
aba_loss = mean_nll(smoothed, [aba])
print('\nPart 3: probability and negative log likelihood')
print(f'  P(ba | initial a) = {smoothed[0, 1]:.6f} * {smoothed[1, 0]:.6f} = {math.exp(-2 * aba_loss):.6f}')
print(f'  aba mean NLL={aba_loss:.6f}; perplexity={math.exp(aba_loss):.6f}')
print(f'  abc raw NLL={mean_nll(raw, [abc])}; add-one NLL={mean_nll(smoothed, [abc]):.6f}')
print(f'  uniform toy NLL=ln({toy_tok.vocab_size})={math.log(toy_tok.vocab_size):.6f}')

# --- Part 4: generation feeds a sampled token back in ---
sample = generate(smoothed, 0, 40, seed=7)
greedy = generate(smoothed, 0, 12, greedy=True)
g = torch.Generator().manual_seed(7)
draws = torch.multinomial(smoothed[0], 20_000, replacement=True, generator=g)
frequencies = torch.bincount(draws, minlength=3).double() / len(draws)
print('\nPart 4: sampling versus greedy choice')
print(f'  seed 7, 40 new tokens: {toy_tok.decode(sample)!r}')
print(f'  greedy, 12 new tokens: {toy_tok.decode(greedy)!r}')
print(f'  20000 independent draws after a: {[round(x, 4) for x in frequencies.tolist()]}')
print(f'  contexts ab and cb both end in b: identical row={torch.equal(smoothed[toy_tok.encode("ab")[-1]], smoothed[toy_tok.encode("cb")[-1]])}')

# --- Part 5: train, choose smoothing on validation, evaluate test once ---
text = (LABS / 'data' / 'tinyshakespeare.txt').read_text()
train_text, val_text, test_text = text[:200_000], text[200_000:220_000], text[-20_000:]
# Vocabulary is learned ONLY on training text. Unknown evaluation chars fail explicitly.
tok = CharTokenizer(train_text)
train_ids, val_ids, test_ids = [tok.encode(s) for s in (train_text, val_text, test_text)]
counts = count_pairs([train_ids], tok.vocab_size)
print('\nPart 5: Tiny Shakespeare, disjoint text slices')
print(f'  corpus chars={len(text)}; train={len(train_ids)}, validation={len(val_ids)}, test={len(test_ids)}')
print(f'  vocabulary={tok.vocab_size}; table cells={counts.numel()}; training pairs={int(counts.sum())}')
print(f'  train offsets=[0, 200000); validation=[200000, 220000); test=[{len(text)-20000}, {len(text)})')
val_scores = {}
for alpha in (0.01, 0.1, 1.0, 10.0):
    p = probabilities(counts, alpha)
    val_scores[alpha] = mean_nll(p, [val_ids])
    print(f'  alpha={alpha:g} validation NLL={val_scores[alpha]:.6f}')
best_alpha = min(val_scores, key=val_scores.get)
bigram = probabilities(counts, best_alpha)
uniform = torch.full_like(bigram, 1 / tok.vocab_size)
# Unigram counts target positions to use exactly the same training events as bigram.
unigram_row = (counts.sum(0) + 1) / (counts.sum() + tok.vocab_size)
unigram = unigram_row.expand_as(bigram)
models = {'uniform': uniform, 'unigram': unigram, 'bigram': bigram}
results = {}
print(f'  selected alpha={best_alpha:g} using validation only')
for name, p in models.items():
    train_loss = mean_nll(p, [train_ids])
    test_loss = mean_nll(p, [test_ids])
    results[name] = test_loss
    print(f'  {name:7s} train NLL={train_loss:.6f}; test NLL={test_loss:.6f}; test perplexity={math.exp(test_loss):.6f}')
print(f'  smoothed table row-sum max error={(bigram.sum(1) - 1).abs().max():.3e}')
print('  sample (seed 7, initial newline, 240 new tokens):')
print(repr(tok.decode(generate(bigram, tok.stoi['\n'], 240, seed=7))))

# --- Part 6: the same table can be expressed as logits in an embedding ---
# This is a representation equivalence, NOT gradient training.
logit_table = torch.nn.Embedding.from_pretrained(smoothed.log(), freeze=True)
x = torch.tensor(aba[:-1], dtype=torch.long)
y = torch.tensor(aba[1:], dtype=torch.long)
logits = logit_table(x)
ce_loss = F.cross_entropy(logits, y).item()
print('\nPart 6: bridge to a neural bigram')
print(f'  toy table shape={tuple(logit_table.weight.shape)}; batch logits shape={tuple(logits.shape)}')
print(f'  softmax(log P) equals P: {torch.allclose(logits.softmax(-1), smoothed[x])}')
print(f'  cross_entropy={ce_loss:.6f}; direct NLL={aba_loss:.6f}')
print('  No optimizer ran. A longer-context model needs more than a different table parameterization.')

# TRY THIS: Change alpha=1 to alpha=10 in the toy. Predict P(b | a) before running.
# TRY THIS: Change the sampling seed. Predict whether the test loss changes.
# TRY THIS: Count [[0, 1], [2, 0]] versus [[0, 1, 2, 0]]. Predict the extra pair.


def check():
    assert torch.equal(toy_counts, manual_counts), 'vectorized and explicit counts disagree'
    assert torch.equal(toy_counts, torch.tensor([[0, 2, 1], [2, 0, 0], [1, 0, 0]]))
    assert int(counts.sum()) == len(train_ids) - 1
    assert torch.allclose(smoothed[0], torch.tensor([1 / 6, 3 / 6, 2 / 6], dtype=torch.float64))
    assert torch.allclose(bigram.sum(1), torch.ones(tok.vocab_size, dtype=torch.float64))
    assert abs(aba_loss + math.log(0.5 * 0.6) / 2) < 1e-12
    assert math.isinf(mean_nll(raw, [abc])) and math.isfinite(mean_nll(smoothed, [abc]))
    assert sample == generate(smoothed, 0, 40, seed=7)
    assert (frequencies - smoothed[0]).abs().max() < 0.02
    assert len(text) - 20_000 >= 220_000, 'validation and test overlap'
    assert results['bigram'] < results['unigram'] < results['uniform'], 'held-out baseline claim failed'
    assert abs(ce_loss - aba_loss) < 1e-12
    separate = count_pairs([[0, 1], [2, 0]], 3)
    assert separate[1, 2] == 0 and separate.sum() == 2


if __name__ == '__main__':
    check()
    print('\nCHECK PASSED')
