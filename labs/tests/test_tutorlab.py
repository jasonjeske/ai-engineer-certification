"""Correctness tests for the shared lab library. Run: ../.venv/bin/python -m pytest -q tests"""
import math
from pathlib import Path
import random

import pytest
import tiktoken
import torch

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.micrograd import MLP, Value
from tutorlab.tokenizers import GPT4_SPLIT, BPETokenizer, CharTokenizer

SAMPLE = "Hello world! It's 2026, and naïve café owners say zdrowie. 123456 tokens?\n\n  def f(x): return x**2"


def test_char_roundtrip_and_unknown():
    tok = CharTokenizer('hello world')
    assert tok.decode(tok.encode('hello')) == 'hello'
    assert tok.vocab == sorted(set('hello world'))
    with pytest.raises(ValueError):
        tok.encode('z')


def test_split_pattern_matches_tiktoken():
    # Our regex pre-split must chunk text exactly like OpenAI's cl100k_base.
    enc = tiktoken.get_encoding('cl100k_base')
    import regex
    ours = regex.findall(GPT4_SPLIT, SAMPLE)
    theirs = regex.findall(enc._pat_str, SAMPLE)
    assert ours == theirs


def test_bpe_roundtrip_any_unicode():
    tok = BPETokenizer()
    tok.train(SAMPLE * 20, 300)
    rng = random.Random(0)
    for _ in range(50):
        s = ''.join(chr(rng.choice([rng.randint(32, 126), rng.randint(0x80, 0x2FFF), 0x1F600])) for _ in range(40))
        assert tok.decode(tok.encode(s)) == s


def test_bpe_merges_never_cross_chunks():
    tok = BPETokenizer()
    tok.train('the cat. the cat. the cat sat.', 280)
    for piece in tok.vocab.values():
        s = piece.decode('utf-8', 'replace')
        assert not (s.strip() and ' ' in s.strip()), f'merge crossed a word boundary: {s!r}'


def test_bpe_encode_matches_training_and_save_load(tmp_path):
    text = (Path(__file__).resolve().parents[1] / 'data' / 'tinyshakespeare.txt').read_text()[:50_000]
    tok = BPETokenizer()
    tok.train(text, 400)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode()) / 1.6, 'BPE should compress training text well'
    p = tmp_path / 't.json'
    tok.save(p)
    assert BPETokenizer.load(p).encode(text[:2000]) == tok.encode(text[:2000])


def test_micrograd_matches_torch():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    e = (c * d).tanh() + (d / 10).exp() - (b.relu() * 3)
    e.backward()
    ta = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    tc = ta + tb
    td = ta * tb + tb ** 3
    te = torch.tanh(tc * td) + torch.exp(td / 10) - torch.relu(tb) * 3
    te.backward()
    assert abs(e.data - te.item()) < 1e-9
    assert abs(a.grad - ta.grad.item()) < 1e-9 and abs(b.grad - tb.grad.item()) < 1e-9


def test_micrograd_mlp_learns():
    random.seed(1)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    net = MLP(3, [4, 4, 1])
    for _ in range(200):
        loss = sum((net(x) - y) ** 2 for x, y in zip(xs, ys))
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data -= 0.05 * p.grad
    assert loss.data < 0.05


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_gpt_initial_loss_is_uniform_guess():
    m = _tiny()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 50, (8, 16), generator=g)
    y = torch.randint(0, 50, (8, 16), generator=g)   # targets unrelated to inputs
    # Note: using y = x would score lower than ln(50), because the tied output
    # head favors the token whose embedding is already in the residual stream.
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(50)) < 0.1


def test_gpt_is_causal():
    m = _tiny().eval()
    x = torch.randint(0, 50, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 7) % 50       # change only the FUTURE tokens
    lx, _ = m(x)
    ly, _ = m(y)
    assert torch.allclose(lx[0, :10], ly[0, :10], atol=1e-6)
    assert not torch.allclose(lx[0, 10:], ly[0, 10:])


def test_gpt_can_memorize_a_batch():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 50, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
