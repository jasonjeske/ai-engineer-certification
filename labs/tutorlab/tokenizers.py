"""Tokenizers, from the simplest to the kind GPT-4 uses.

CharTokenizer: one token per character. Easy to read, wasteful in practice.
BPETokenizer:  byte-level byte-pair encoding with a regex pre-split, the same
               design as GPT-2/GPT-4 tokenizers (after Karpathy's minbpe, MIT).

Why the pre-split matters: without it, BPE happily merges across word and
punctuation boundaries ("dog." or "the c"), wasting vocabulary on accidents of
the training text. Real tokenizers first chop text into word-like chunks and
only merge inside a chunk.
"""
import json

import regex as re

# GPT-4 (cl100k_base) split pattern, as published in OpenAI's tiktoken.
GPT4_SPLIT = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")


class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f'character {e.args[0]!r} is not in the vocabulary') from None

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)


def _pair_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE. Base vocabulary = the 256 possible bytes, so ANY text encodes."""

    def __init__(self, pattern=GPT4_SPLIT):
        self.pattern = pattern
        self._re = re.compile(pattern)
        self.merges = {}                                   # (id, id) -> new id, in learned order
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, 'vocabulary must at least cover the 256 bytes'
        # Count each distinct chunk once ("the" appears thousands of times but is
        # processed once, weighted by its count). Same result, far faster.
        freq = {}
        for c in self._re.findall(text):
            freq[c] = freq.get(c, 0) + 1
        chunks = [list(c.encode('utf-8')) for c in freq]
        weights = list(freq.values())
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        history = []
        for k in range(vocab_size - 256):
            counts = {}
            for ids, w in zip(chunks, weights):
                for pair in zip(ids, ids[1:]):
                    counts[pair] = counts.get(pair, 0) + w
            if not counts:
                break
            # Most frequent pair; ties go to the smallest pair so training is deterministic.
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            new_id = 256 + k
            chunks = [_merge(ids, pair, new_id) if len(ids) > 1 else ids for ids in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            history.append((pair, new_id, counts[pair]))
            if verbose:
                print(f'merge {k + 1:4d}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r}'
                      f' -> {self.vocab[new_id]!r}  ({counts[pair]} times)')
        return history

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            counts = _pair_counts(ids)
            # Apply the merge that was LEARNED EARLIEST, exactly as training did.
            pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        out = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk.encode('utf-8')))
        return out

    def decode(self, ids):
        return b''.join(self.vocab[int(i)] for i in ids).decode('utf-8', errors='replace')

    def tokens(self, text):
        """Human-readable pieces, for inspection."""
        return [self.vocab[i].decode('utf-8', errors='replace') for i in self.encode(text)]

    def save(self, path):
        data = {'pattern': self.pattern, 'merges': [[a, b, n] for (a, b), n in self.merges.items()]}
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls(data['pattern'])
        for a, b, n in data['merges']:
            tok.merges[(a, b)] = n
            tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        return tok
