"""Lab: Tokens, teaching a computer to read.

Goal: build a real byte-level BPE tokenizer (the GPT-2/GPT-4 design), train it on
Shakespeare, and compare it with OpenAI's production tokenizers.

Run:  python lab.py
Runs offline in about 20 seconds. Everything it needs is in labs/ (Tiny Shakespeare
and a local cache of OpenAI's tiktoken vocabularies).
"""
import os
import pathlib
import sys

LABS = pathlib.Path(__file__).resolve().parents[3] / 'labs'
sys.path.insert(0, str(LABS))
os.environ['TIKTOKEN_CACHE_DIR'] = str(LABS / 'data' / 'tiktoken-cache')   # offline: use the local copy

import tiktoken  # noqa: E402

from tutorlab.tokenizers import BPETokenizer, CharTokenizer  # noqa: E402

TEXT = (LABS / 'data' / 'tinyshakespeare.txt').read_text()
TRAIN = TEXT[:200_000]      # train on the first ~200 KB
HELD_OUT = TEXT[-20_000:]   # measure on text the tokenizer never saw

# --- Part 1: text is already numbers -------------------------------------------
print('Part 1: what a computer actually stores')
word = 'Hi ń'
print(f'  string            {word!r}')
print(f'  Unicode points    {[ord(c) for c in word]}')
print(f'  UTF-8 bytes       {list(word.encode("utf-8"))}   <- "ń" needs 2 bytes')

# --- Part 2: the simplest tokenizer, one token per character -------------------
print('\nPart 2: character tokenizer')
char_tok = CharTokenizer(TEXT)
ids = char_tok.encode('To be, or not')
print(f'  vocabulary size   {char_tok.vocab_size}  (every distinct character in Shakespeare)')
print(f'  "To be, or not" -> {ids}')
print(f'  decoded back    -> {char_tok.decode(ids)!r}')
try:
    char_tok.encode('Dzień')
except ValueError as e:
    print(f'  "Dzień" fails     {e}')

# --- Part 3: train byte-pair encoding ------------------------------------------
print('\nPart 3: training BPE (first 12 merges shown)')
bpe = BPETokenizer()
history = bpe.train(TRAIN, vocab_size=1024)
for (a, b), new_id, n in history[:12]:
    print(f'  merge {new_id - 255:3d}: {bpe.vocab[a]!r:>8} + {bpe.vocab[b]!r:<8} -> {bpe.vocab[new_id]!r:<12} seen {n:,}x')
print(f'  ... {len(history)} merges, final vocabulary {bpe.vocab_size}')
print(f'  a few late tokens: {[bpe.vocab[i].decode("utf-8", "replace") for i in range(1000, 1012)]}')

# --- Part 4: encode, decode, and compression -----------------------------------
print('\nPart 4: encode and decode')
line = 'To be, or not to be, that is the question.'
print(f'  pieces {bpe.tokens(line)}')
print(f'  ids    {bpe.encode(line)}')
n_bytes = len(HELD_OUT.encode('utf-8'))
n_tokens = len(bpe.encode(HELD_OUT))
print(f'  held-out text: {n_bytes:,} bytes -> {n_tokens:,} tokens = {n_bytes / n_tokens:.2f} bytes per token')
print(f'  "Dzień dobry" -> {bpe.tokens("Dzień dobry")}  (no crash: any text is just bytes)')

print('\n  vocabulary size vs compression (same training text, held-out text):')
sweep = {}
for v in (256, 384, 512, 768, 1024):
    t = bpe if v == 1024 else BPETokenizer()
    if v != 1024:
        t.train(TRAIN, vocab_size=v)
    sweep[v] = n_bytes / len(t.encode(HELD_OUT))
    print(f'    vocab {v:5d}  {sweep[v]:.2f} bytes per token  {"#" * round(sweep[v] * 10)}')

# --- Part 5: the real thing, OpenAI's tokenizers -------------------------------
print('\nPart 5: production tokenizers (tiktoken)')
encs = {name: tiktoken.get_encoding(name) for name in ('gpt2', 'cl100k_base', 'o200k_base')}
prod_ratio = {}
for name, enc in encs.items():
    prod_ratio[name] = n_bytes / len(enc.encode(HELD_OUT))
    print(f'  {name:12} vocabulary {enc.n_vocab:7,}   held-out Shakespeare {prod_ratio[name]:.2f} bytes per token')
samples = ['strawberry', ' strawberry', 'Dzień dobry, jak się masz?', '127 + 4 = 131', '        return y']
for s in samples:
    print(f'  {s!r}')
    for name, enc in encs.items():
        pieces = [enc.decode_single_token_bytes(t).decode('utf-8', 'replace') for t in enc.encode(s)]
        print(f'      {name:12} {len(pieces):2d} tokens {pieces}')

print('\nPart 6: same meaning, different price')
en, pl = 'Good morning, how are you?', 'Dzień dobry, jak się masz?'
for name, enc in encs.items():
    print(f'  {name:12} English {len(enc.encode(en))} tokens, Polish {len(enc.encode(pl))} tokens')


# TRY THIS (predict first, then run):
#   1. Add 2048 to the sweep. Will the gain from 1024 to 2048 be bigger or smaller than 512 to 1024? Why?
#   2. Train on TEXT.upper() instead. What happens to "To be, or not" afterwards?
#   3. Add your own sentence in Polish or German to `samples`. Which tokenizer is cheapest?


def check():
    # Round trip must be exact for any text, including unseen Unicode.
    for s in ['To be, or not', 'Dzień dobry 🙂', HELD_OUT[:3000]]:
        assert bpe.decode(bpe.encode(s)) == s, f'round trip failed for {s[:20]!r}'
    # BPE must compress better than one byte per token on unseen English.
    assert n_bytes / n_tokens > 2.0, 'a 1024 vocabulary should beat 2 bytes per token on held-out English'
    # 256 = raw bytes, exactly 1 byte per token; every larger vocabulary compresses more.
    assert abs(sweep[256] - 1.0) < 1e-9
    assert all(sweep[a] < sweep[b] for a, b in zip(sorted(sweep), sorted(sweep)[1:]))
    # The first merge is the most common pair in the training text.
    assert history[0][2] == max(c for _, _, c in history), 'first merge must be the most frequent'
    # Regex pre-split means no learned token straddles a word and punctuation.
    assert all(not (p.strip() and ' ' in p.strip()) for p in (v.decode('utf-8', 'replace') for v in bpe.vocab.values()))
    # Production facts used in the chapter.
    assert encs['gpt2'].n_vocab == 50257
    assert all(r > n_bytes / n_tokens for r in prod_ratio.values()), 'big vocabularies should beat our 1024'
    assert len(encs['o200k_base'].encode(' strawberry')) == 1
    assert len(encs['o200k_base'].encode('strawberry')) == 3
    assert len(encs['gpt2'].encode(pl)) > len(encs['o200k_base'].encode(pl))


check()
print('\nCHECK PASSED')
