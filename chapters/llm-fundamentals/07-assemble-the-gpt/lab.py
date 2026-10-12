"""Assemble and inspect an untrained GPT, then test the complete computation.

Run from the project root:
  env -u PYTHONPATH .venv/bin/python chapters/llm-fundamentals/07-assemble-the-gpt/lab.py
Or: cd labs && ../.venv/bin/python ../chapters/llm-fundamentals/07-assemble-the-gpt/lab.py
Resolves labs/ from this file, so an absolute invocation works from anywhere.
Expected runtime: under 10 seconds on CPU. Offline, no checkpoint or training.
"""
import io
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'labs'))
import torch
import torch.nn.functional as F

from tutorlab.gpt import GPT, GPTConfig
from tutorlab.gpt_assembly import parameter_budget, trace_forward


def main():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    torch.use_deterministic_algorithms(True)

    # --- Part 1: a batch of IDs, not a bag of words ---
    vocab = 'abcdefg .'
    stoi = {char: i for i, char in enumerate(vocab)}
    texts = ['bad ca.', 'cab ad.']
    data = torch.tensor([[stoi[c] for c in text] for text in texts])
    idx, targets = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    cfg = GPTConfig(vocab_size=len(vocab), block_size=6, n_layer=3,
                    n_head=2, n_embd=8, dropout=0.0)
    model = GPT(cfg).eval()
    print('Part 1: inputs and next-token targets')
    print(f'  vocabulary={vocab!r}; B=2 T=6 C=8 V=9 layers=3 heads=2')
    print(f'  inputs={idx.tolist()}')
    print(f'  targets={targets.tolist()}')

    # --- Part 2: assemble the real components, retain every stage ---
    with torch.no_grad():
        trace = trace_forward(model, idx)
        logits, loss = model(idx, targets)
    print('\nPart 2: shapes through the assembled GPT')
    for name, tensor in trace.items():
        print(f'  {name:14s} {tuple(tensor.shape)}')
    forward_error = (trace['logits'] - logits).abs().max().item()
    repeat_token_error = (trace['tokens'][0, 1] - trace['tokens'][0, 5]).abs().max().item()
    repeat_pos_gap = (trace['residual_in'][0, 1] - trace['residual_in'][0, 5]).abs().max().item()
    print(f'  explicit assembly vs GPT.forward max error={forward_error:.9f}')
    print(f'  repeated a: token-vector difference={repeat_token_error:.9f}')
    print(f'  repeated a: after positions difference={repeat_pos_gap:.9f}')

    # --- Part 3: logits, probabilities and a hand-sized unembedding ---
    probs = logits.softmax(dim=-1)
    print('\nPart 3: output distributions (untrained)')
    print('  last-row probabilities: ' + ' '.join(
        f'{c if c != " " else "_"}:{p:.6f}' for c, p in zip(vocab, probs[0, -1].tolist())))
    print(f'  row sums min={probs.sum(-1).min():.6f} max={probs.sum(-1).max():.6f}')
    print(f'  next-token loss={loss.item():.6f}; uniform baseline ln(9)={math.log(9):.6f}')
    h = torch.tensor([1., 2.], dtype=torch.float64)
    table = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], dtype=torch.float64)
    hand_logits = h @ table.T
    hand_probs = hand_logits.softmax(-1)
    print(f'  hand example: h={h.tolist()} table={table.tolist()}')
    print(f'  hand logits={hand_logits.tolist()}')
    print('  hand probabilities=' + str([round(p, 6) for p in hand_probs.tolist()]))
    print(f'  hand target index=1 loss={-hand_probs[1].log():.6f}')

    # --- Part 4: parameter accounting and a real checkpoint round trip ---
    budget = parameter_budget(cfg)
    print('\nPart 4: unique learned numbers')
    for name, count in budget.items():
        print(f'  {name}={count}')
    print(f'  per block={sum(p.numel() for p in model.blocks[0].parameters())}')
    print(f'  total={model.num_params()}; formula={sum(budget.values())}')
    print(f'  tied head is same Parameter={model.head.weight is model.tok_emb.weight}')
    buffer_count = sum(b.numel() for b in model.buffers())
    print(f'  mask buffer elements (not parameters)={buffer_count}')
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = GPT(cfg).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        restored_logits, _ = restored(idx)
    roundtrip_error = (restored_logits - logits).abs().max().item()
    print(f'  checkpoint round-trip max error={roundtrip_error:.9f}')

    # --- Part 5: the model contract, not just a shape check ---
    changed = idx.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % cfg.vocab_size
    with torch.no_grad():
        changed_logits, _ = model(changed)
        prefix_logits, _ = model(idx[:, :3])
        single_logits, _ = model(idx[:1])
    past_error = (changed_logits[:, :3] - logits[:, :3]).abs().max().item()
    future_gap = (changed_logits[:, 3:] - logits[:, 3:]).abs().max().item()
    prefix_error = (prefix_logits - logits[:, :3]).abs().max().item()
    batch_error = (single_logits - logits[:1]).abs().max().item()
    model.zero_grad(set_to_none=True)
    _, grad_loss = model(idx, targets)
    grad_loss.backward()
    grad_norms = [block.attn.qkv.weight.grad.norm().item() for block in model.blocks]
    all_grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                           for p in model.parameters())
    rejected = False
    try:
        model(torch.zeros((1, cfg.block_size + 1), dtype=torch.long))
    except AssertionError as exc:
        rejected = True
        print(f'\nPart 5: overlong input rejected: {exc}')
    print(f'  suffix edit: past max error={past_error:.9f}; changed suffix max gap={future_gap:.9f}')
    print(f'  prefix-only max error={prefix_error:.9f}; separate batch-row max error={batch_error:.9f}')
    print('  attention gradient norms=' + str([round(g, 6) for g in grad_norms]))
    print(f'  all parameter gradients present and finite={all_grads_finite}')
    print('  optimizer steps=0; no language competence claimed')

    # TRY THIS: set n_layer=1; predict parameter count and output shape before running.
    # TRY THIS: use n_head=4 at width 8; predict whether the parameter count changes.
    # TRY THIS: change only the final input ID; predict which output rows may change.
    def check():
        assert logits.shape == (2, 6, 9)
        assert forward_error < 1e-7
        assert repeat_token_error == 0 and repeat_pos_gap > 1e-6
        assert torch.isfinite(probs).all() and (probs >= 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones((2, 6)), atol=1e-6)
        assert torch.allclose(hand_logits, torch.tensor([1., 2., -1.], dtype=torch.float64))
        assert model.num_params() == sum(budget.values())
        assert model.head.weight is model.tok_emb.weight
        assert restored.head.weight is restored.tok_emb.weight and roundtrip_error == 0
        assert past_error < 1e-6 and future_gap > 1e-6
        assert prefix_error < 1e-6 and batch_error < 1e-6
        assert all_grads_finite and all(g > 0 for g in grad_norms)
        assert rejected
        manual_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        assert torch.allclose(loss, manual_loss)
    check()
    print('CHECK PASSED')


if __name__ == '__main__':
    main()
