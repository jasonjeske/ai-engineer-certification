## Slide: Assemble the GPT
- LLM Fundamentals, chapter 7
- A complete, untrained next-token predictor
- Inspect the computation before training it
Narration:
A complete GPT is a chain of components you already understand. Today we join them and inspect the entire path from token IDs to next-token scores. We will not train the model yet, and that distinction matters. A machine can produce a valid probability distribution without knowing anything useful about language. Our job is to establish that the architecture is connected correctly, that it respects the causal boundary, and that a learning signal can reach its parameters. Then the next chapter can address how those parameters actually learn.

## Slide: Start with the smallest complete input
```text
Vocabulary: abcdefg .
Original:   bad ca.
Input:      bad ca
Target:     ad ca.
```
- Targets are shifted successors, not copies
Narration:
Our vocabulary contains nine characters, including a space and a period. Consider the original string shown on screen. The input omits its final character. The target omits its first character. That offset makes every target the successor of the corresponding input token. The model must predict what comes next using what it can already see. If targets were identical to inputs, we would instead reward copying the current visible token. Our shared model does not shift the labels automatically. The lab performs that preparation explicitly, before calling the model.

## Slide: Give every axis a meaning
```text
B = 2 independent sequences
T = 6 input positions
C = 8 internal features
V = 9 vocabulary candidates
IDs: (B,T) -> vectors: (B,T,C)
```
Narration:
The batch contains two sequences, each with six input positions. Each position will have eight feature coordinates. The vocabulary has nine possible candidates. Say what each axis means before manipulating the tensor. The batch axis separates examples. The time axis separates positions within an example. The feature axis describes a position. Later, the output head replaces that feature axis with a vocabulary axis. Adding more blocks does not create additional input tokens. It increases the depth of the computation performed on the same collection of positions.

## Slide: Add identity and location
```python
pos = torch.arange(idx.shape[1], device=idx.device)
tokens = model.tok_emb(idx)
positions = model.pos_emb(pos)
x = model.drop(tokens + positions)
```
- Addition preserves width
Narration:
A token ID selects a row from the token embedding table. A position number selects a row from a second learned table. We add those vectors coordinate by coordinate. We do not concatenate them, so their width stays unchanged. The position table is shared across batch rows by broadcasting. In the lab, two occurrences of the same character have identical token vectors before the addition, but different representations afterward. Position addition supplies an explicit location signal. The causal mask also introduces positional structure, so removing this table would not make every possible order distinction disappear.

## Slide: Repeat a shape-preserving block
```text
u = x + attention(norm1(x))
y = u + mlp(norm2(u))

Each block: (2,6,8) -> (2,6,8)
```
Narration:
A block first normalizes its input for attention, computes a contextual update and adds that update to the original representation. It then normalizes the updated representation for the feedforward network and adds that network's result. Both residual additions require matching shapes. The feedforward network expands internally, but projects back to the original width before the addition. Attention also recombines its heads back to that width. This common input and output contract is what makes blocks stackable. The next block receives an updated representation, not a longer sequence or a wider feature vector.

## Slide: Independent blocks, registered parameters
```python
self.blocks = nn.ModuleList(
    Block(cfg) for _ in range(cfg.n_layer)
)
for block in self.blocks:
    x = block(x)
```
Narration:
Each call to the block constructor creates a separate set of parameters. That is different from repeating a reference to the same block. The latter would reuse weights across depth. ModuleList tells PyTorch that these blocks belong to the model, so optimizers and checkpoint operations can discover their parameters. A plain Python list alone does not provide that registration. The loop then feeds each output into the next input. These details are ordinary programming, but getting them wrong can undermine a training run even when every individual matrix multiplication has a legal shape.

## Slide: Read out scores, not a token
```python
h = model.ln_f(x)
logits = model.head(h)
probs = logits.softmax(dim=-1)
```
- Hidden vectors: (2,6,8)
- Vocabulary scores: (2,6,9)
Narration:
After the block stack, final layer normalization operates within each position's feature vector. The output head then computes a score for every vocabulary candidate. These scores are logits. They need not be positive, and they do not need to sum to one. Softmax converts each score row into a probability distribution over the vocabulary. Notice that the model has not selected a token yet. Choosing and appending a token belongs to the generation procedure. The final position's distribution predicts what follows the whole input; earlier positions predict successors to shorter prefixes.

## Slide: A head calculation you can do by hand
```text
h = [1, 2]
Candidate rows:
[1, 0]  ->  1
[0, 1]  ->  2
[1,-1]  -> -1
Probabilities: [0.259496, 0.705385, 0.035119]
```
Narration:
Here is a tiny scoring example executed separately in the lab. Take a representation with coordinates one and two. Compare it with each candidate row using a dot product. The first score is one, the second is two, and the third is negative one. Softmax gives the middle candidate the largest probability. Nothing symbolic decides that this candidate is correct. It simply aligns best with the current representation under these chosen weights. If the middle candidate is the target, the loss is the negative logarithm of its probability. That is the link between scoring and the training objective.

## Slide: One table, two jobs
```python
self.head.weight = self.tok_emb.weight
```
- Lookup rows at the input
- Score candidates at the output
- Count the shared Parameter once
Narration:
In our implementation, the token embedding table and the output head use the very same parameter. This is weight tying. At the input, an ID selects a row. At the output, the final representation is compared with all rows to produce candidate scores. The shared table has two jobs, but it is counted only once as learned storage. Tying does not invert the transformer and recover the original token. The output task still predicts a successor. Also, this particular sharing does not mean that different transformer blocks share their weights. They remain separate instances.

## Slide: Audit the learned numbers
```text
Token table and tied head:   72
Position table:              48
Three blocks:             2616
Final normalization:        16
Unique parameters:        2752
```
Narration:
The lab derives a parameter budget from the actual implementation and compares it with PyTorch's parameter iterator. The total is two thousand seven hundred fifty two learned numbers. That total includes linear biases and the learned scale and shift of layer normalization. It does not include the causal masks, because those masks are buffers rather than trainable parameters. This distinction matters when reading checkpoints or estimating capacity. Two names can refer to a shared parameter, and stored buffers can consume memory without adding learned degrees of freedom. Parameter count is not a complete measure of runtime memory.

## Slide: Valid probabilities are not learned language
```text
Measured loss:       2.213514
Uniform ln(9):       2.197225
Optimizer steps:     0
```
- Structural correctness is not predictive quality
Narration:
Our random model produces finite, nonnegative probabilities whose rows sum to one. Its measured loss is near the uniform baseline for this small vocabulary. But it has taken no optimizer steps. We have no evidence of language competence, and we do not claim any. Random initialization is not necessarily exactly uniform either. Tied weights and residual pathways can create preferences before learning. This is why the lab checks validity without demanding that every candidate receive exactly the same probability. Accuracy and generalization will require learning and evaluation on suitable data, not just successful assembly.

## Slide: Test the entire causal boundary
```text
Change only positions 3 onward
Earlier logits max difference: 0.000000000
Changed suffix max difference: 0.151288509
```
- Use deterministic evaluation for comparisons
Narration:
Now perform a test that output shapes cannot replace. Keep the early prefix fixed and change only later input tokens. Earlier predictions must not change. Our recorded run shows no difference in earlier logits and a clear difference in the edited suffix. The causal guarantee must hold through every block, not only in the first attention matrix. Normalization and the feedforward network act within positions, so they do not introduce future information. Use evaluation mode when comparing separate calls, because active dropout could otherwise create random differences unrelated to future leakage.

## Slide: Check independence and restoration
- A prefix alone matches its full-sequence rows
- A sequence alone matches its batch row
- Save weights, restore a fresh model, compare logits
Narration:
Two additional comparisons test the computation's boundaries. Running only a prefix should agree with the corresponding prefix rows from the full input. Running one example alone should agree with its row from the batch. The lab also performs a real save and load through an in-memory checkpoint. A fresh model with the same configuration restores identical logits and retains the tied parameter relationship. This checks preservation of the computation, not training quality. For a practical checkpoint outside this exercise, keep the configuration and vocabulary ordering too, because weight tensors alone do not explain the meaning of each token ID.

## Slide: Make sure learning can reach every block
```text
Attention gradient norms:
[0.049288, 0.042088, 0.041107]
All parameter gradients present and finite: True
```
- Backward is not an optimizer step
Narration:
Finally, compute the next-token loss and run backward. Every parameter must have a present, finite gradient, and each block's attention projection receives a nonzero gradient norm in this experiment. This demonstrates a connected learning path. It does not prove that optimization will succeed on a large dataset. No weights are updated here. Also notice the distinction between evaluation mode and disabling gradients. Evaluation mode changes behavior such as dropout, while gradient recording is controlled separately. We intentionally use evaluation mode for a deterministic gradient connectivity test and still perform backpropagation.

## Slide: Your prediction comes before the edit
- Reduce depth: which shapes stay fixed?
- Change head count at fixed width: does storage grow?
- Change the final token: which rows may respond?
Narration:
Before editing the lab, write down a prediction. Reducing depth should remove transformations without changing the output vocabulary shape. Changing the head count at fixed total width changes how projected features are partitioned, not the size of those projections in this implementation. Editing the final input token must leave earlier predictions unchanged. Try one modification at a time and compare with the original output. Then explain the full path in your own words. Your study partner will look for one gap, ask one follow-up, and give a new application case rather than treating execution alone as understanding.

## Slide: Summary: connected, tested, still untrained
- IDs plus positions become a residual stream
- Independent blocks preserve the shape
- A tied head scores the vocabulary
- Causality, restoration and gradients are tested
Narration:
You now have a complete GPT forward computation with visible intermediate tensors. Token identity and position initialize the representation. Independent blocks repeatedly update it without changing the external shape. Final normalization and a tied head turn it into vocabulary scores. The lab checks normalized probabilities, causal isolation, batch independence, checkpoint restoration and gradient connectivity. None of those tests claims that the model has learned language. That separation is the main lesson: first establish the computation and its contracts, then train it, and finally evaluate what it actually learned. Keep those claims separate as the model becomes larger.
