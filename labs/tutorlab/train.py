"""Train a small character-level GPT on Tiny Shakespeare and save a checkpoint.

Run from labs/:   ../.venv/bin/python -m tutorlab.train --steps 3000
Defaults train in a few minutes on an Apple-silicon GPU (MPS) and produce
labs/models/shakespeare-char.pt, which later chapters load.
"""
import argparse
import json
import math
import pathlib
import time

import torch

from .gpt import GPT, GPTConfig
from .tokenizers import CharTokenizer

LABS = pathlib.Path(__file__).resolve().parents[1]
DATA = LABS / 'data' / 'tinyshakespeare.txt'


def device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data():
    text = DATA.read_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tok, data[:n], data[n:]


def get_batch(data, block_size, batch_size, gen, dev):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # targets = inputs shifted by one
    return x.to(dev), y.to(dev)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, gen, dev, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, batch_size, gen, dev))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--block', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--embd', type=int, default=192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--out', default=str(LABS / 'models' / 'shakespeare-char.pt'))
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = device()
    tok, train, val = load_data()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=a.block, n_layer=a.layers,
                    n_head=a.heads, n_embd=a.embd, dropout=0.1)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    warmup = 100

    def lr_at(step):   # linear warmup, then cosine decay to 10%
        if step < warmup:
            return a.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, a.steps - warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    print(f'device={dev} params={model.num_params():,} vocab={tok.vocab_size} '
          f'train_tokens={len(train):,} val_tokens={len(val):,}')
    print(f'loss at init should be about ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}')
    t0, log = time.time(), []
    for step in range(a.steps + 1):
        if step % 500 == 0 or step == a.steps:
            ev = estimate_loss(model, {'train': train, 'val': val}, a.block, a.batch, gen, dev)
            log.append({'step': step, **ev})
            print(f'step {step:5d}  train {ev["train"]:.4f}  val {ev["val"]:.4f}  ({time.time() - t0:.0f}s)')
        if step == a.steps:
            break
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        _, loss = model(*get_batch(train, a.block, a.batch, gen, dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'cfg': cfg.__dict__, 'state': model.state_dict(), 'vocab': tok.vocab, 'log': log}, out)
    (out.with_suffix('.json')).write_text(json.dumps({'cfg': cfg.__dict__, 'log': log, 'seconds': round(time.time() - t0)}, indent=1))
    ctx = torch.tensor([tok.encode('\n')], dtype=torch.long, device=dev)
    sample = model.eval().generate(ctx, 400, temperature=0.8, top_k=40,
                                   generator=torch.Generator(device=dev).manual_seed(a.seed) if dev != 'mps' else None)
    print('\n--- sample ---\n' + tok.decode(sample[0].tolist()))
    print(f'\nsaved {out}')


def load(path=None, dev=None):
    path = pathlib.Path(path or LABS / 'models' / 'shakespeare-char.pt')
    ck = torch.load(path, map_location=dev or 'cpu', weights_only=False)
    tok = CharTokenizer(''.join(ck['vocab']))
    model = GPT(GPTConfig(**ck['cfg']))
    model.load_state_dict(ck['state'])
    return model.to(dev or 'cpu').eval(), tok


if __name__ == '__main__':
    main()
