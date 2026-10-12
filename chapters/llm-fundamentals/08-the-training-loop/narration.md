## Slide: The training loop
- LLM Fundamentals, chapter 8 of 10
- Prediction uses weights
- Training changes weights using examples
Narration:
A model learns when an update changes its adjustable numbers to better fit its training examples. The previous chapter assembled a complete transformer and checked the forward computation. That did not make its predictions useful. Today we cross the boundary from a working computation to an actual learning experiment. We begin with three scores you can inspect by hand, then train a table of next-character predictions. Finally, the same update function trains our shared GPT. The point is not to build a useful chatbot in this tiny experiment. It is to understand exactly what changes, why it changes, and what the result proves.

## Slide: Give every prediction the right target
```python
x = tokens[:, :-1].contiguous()
y = tokens[:, 1:].contiguous()
```
- Inputs omit the final token
- Targets omit the first token
- Each target is the next token
Narration:
Training starts with a clearly defined task. For next-token prediction, the target at a position is the token immediately following it. We slice one sequence in two overlapping ways. Inputs leave off the last token, while targets leave off the first. Both arrays then have matching shapes. The contiguous call arranges storage for the existing model's flattening operation; it does not change which tokens are present. If you accidentally use the input itself as the target, you reward copying a token the model can already see. A small loss on that task would not establish that your next-token predictor works.

## Slide: Scores become probabilities, then a penalty
```text
logits -> softmax probabilities
one target loss = -log(target probability)
batch loss = mean of target losses
```
- PyTorch cross entropy takes raw logits
Narration:
A forward pass produces logits, which are unrestricted real scores. Softmax turns one score row into a probability distribution over vocabulary candidates. The loss looks up the probability of the actual target and takes its negative natural logarithm. A larger target probability earns a smaller penalty. We average those penalties over the scored positions. This measures confidence, not just whether the highest-scoring token is correct. PyTorch cross entropy expects the raw logits and performs logarithmic normalization internally. Do not apply softmax first and then supply those probabilities as logits. The shapes may still work while the intended calculation does not.

## Slide: Start with three equal candidates
```text
logits:       [0, 0, 0]
target ID:    1
probabilities: one third each
loss:         1.098612
```
Narration:
Here is the smallest complete example from the lab. Three logits all start at zero. The model has no preference, so every candidate receives one third of the probability. The target is the middle candidate, with index one. The measured loss is one point zero nine eight six one two. In this first experiment, the logits themselves are directly adjustable values. That is a simplification for teaching, not how a transformer's output scores are stored. In the transformer, scores are computed from many earlier parameters. First we learn how to improve these simple scores, then let differentiation trace back to the earlier weights.

## Slide: The gradient tells us a local direction
```text
gradient = probability - one_hot_target
         = [1/3, -2/3, 1/3]
mean over N targets: divide each row by N
```
Narration:
A derivative tells us how the loss responds to a small change. For one hard-target cross-entropy example, the derivative with respect to a logit is its probability minus the corresponding one-hot target entry. Our target vector is zero, one, zero. Subtracting it gives one third, negative two thirds, one third. Raising either competing score locally raises the loss. Raising the correct score locally lowers it. For a mean across multiple target positions, divide each row's contribution by the number of scored positions. This simple formula is a logit gradient, not the gradient of every weight in the transformer. Earlier weights require the chain rule.

## Slide: One real parameter update
```text
new = old - learning_rate * gradient
learning_rate = 0.3
new logits = [-0.1, 0.2, -0.1]
target probability = 0.402960
loss = 0.908918
```
Narration:
Plain gradient descent subtracts the learning rate times the gradient. With learning rate zero point three, the middle score rises to zero point two and the other scores fall to negative zero point one. Subtracting a negative derivative is why the correct score rises. The lab recomputes the probabilities and loss using those updated values. The target probability increases and the loss decreases, exactly as the slide shows. It also checks the derivative by making tiny positive and negative perturbations and measuring the loss slope. That finite-difference check is an independent diagnostic, not the algorithm PyTorch uses to train a large model.

## Slide: Backward is not the update
```text
backward: compute and accumulate gradients
step:     change parameters
reset:    clear gradient buffers
```
- Part 2 checks weights before and after backward
Narration:
Keep these operations separate. Backward calculates derivative contributions and accumulates them in gradient buffers. It does not itself update the weights. The optimizer step changes the parameters in this loop. The reset clears temporary gradient buffers, not learned weights. Part two of the lab stores a copy of the original weights and confirms that backward leaves them unchanged. Then it recomputes the same forward calculation at those unchanged weights and calls backward again without clearing. The stored gradients double. A fresh forward graph makes this a clean accumulation experiment rather than an attempt to reuse an already-consumed graph.

## Slide: Accumulation can be intentional
- Reset once per intended update
- Weight microbatch means by target counts
- Accumulate backward contributions
- Step after the complete effective batch
Narration:
Usually we clear gradients before the next independent update so old contributions do not leak into it. But accumulation is also useful. When a full batch does not fit in memory, process smaller pieces and collect their gradient contributions before taking one optimizer step. The scaling must preserve the intended mean over targets. Equal-sized pieces can receive equal weights. Unequal pieces need weights proportional to their contributing target counts. Our test compares that weighted accumulation with a full-batch gradient. Stepping after each piece is different, because later derivatives would be calculated at changed weights. The reset schedule is therefore part of the algorithm, not cosmetic housekeeping.

## Slide: Rebuild the bigram as learned scores
```python
self.table = nn.Embedding(vocab_size, vocab_size)
logits = self.table(idx)
```
- One row per current character
- Eight vocabulary entries
- Forty-two training targets
Narration:
Now scale from one score row to a table. Every current character selects a row containing scores for all possible next characters. The model cannot use earlier context because its lookup never reads that context. Our small corpus contains eight distinct characters and forty-two training targets. We initialize all scores to zero, producing an exactly uniform baseline. Counting transitions gives us an independent reference distribution for each row. Gradient-based learning should approach those frequencies under this objective. That is a stronger check than simply demanding a smaller loss, because we can inspect what probabilities the learned table actually represents.

## Slide: The complete update helper
```python
model.train()
optimizer.zero_grad(set_to_none=True)
_, loss = model(x, y)
loss.backward()
optimizer.step()
```
Narration:
This is the actual loop body shared by our bigram and transformer experiments. Training mode selects layer behavior such as dropout. Reset prepares the gradient buffers. Forward returns a loss connected to the parameters. Backward calculates derivatives, and step applies the optimizer's update. We construct the optimizer once outside the loop, so its internal state persists. These repeated fitting experiments use AdamW, which carries adaptive state and is not identical to the plain gradient-descent equation used earlier. Weight decay is explicitly disabled here. Also notice that this loss was computed before the update. Reading its scalar value afterward does not turn it into a new post-update measurement.

## Slide: Compare learning with the count-based limit
```text
updates    train loss
0          2.079442
50         0.748651
200        0.735081
count-based infimum: 0.731409
```
Narration:
The recorded curve is evaluated after the stated number of updates. It approaches the count-based loss limit rather than zero. Why is the limit positive? The same current character has several possible successors, and the bigram must use one distribution for all its occurrences. It cannot distinguish contexts it never reads. Exact zero probabilities for unobserved transitions also sit at a boundary that finite softmax scores only approach. We therefore call the reference an infimum. The learned probabilities become close to the empirical counts, giving us evidence that the optimizer fitted the intended statistical object, not merely that some number happened to fall.

## Slide: A falling loss is not a universal victory
```text
alternating-pattern model
train loss:  0.693147 -> 0.025389
stress loss: 0.693147 -> 3.686117
```
- Deliberately conflicting evaluation pattern
Narration:
Now challenge the conclusion. A separate model learns an alternating pattern of token IDs. Its training loss improves, but we also evaluate it on a string that repeatedly uses zero. That stress string rewards a conflicting transition. The stress loss gets worse as the model becomes more confident in its training pattern. This is intentional distribution mismatch, not a natural-language benchmark or an ordinary measured overfitting curve. It shows why lower training loss cannot guarantee better performance everywhere. Even our separate toy validation string shares phrases with training. A separate file is not enough to establish independent evidence when the content itself overlaps.

## Slide: The same loop trains the shared GPT
```text
inputs:  (2, 8)
logits:  (2, 8, 3)
params:  3488
updates: 80
loss:    1.156632 -> 0.002131
```
Narration:
Replace the table with the shared GPT from the earlier chapters. Both models accept inputs and targets and return logits with an optional loss, so the same update function works unchanged. The tiny transformer fits short cyclic token sequences, and its measured training loss falls sharply. The lab also checks that attention projection weights changed. Learning reached inside the model, not just an isolated output table. But this pattern is simple enough for a bigram too. We have demonstrated successful optimization through the complete transformer computation, not a benefit from attention or broad language competence. This is a memorization and integration test with an explicitly limited claim.

## Slide: Evaluate without changing the experiment
- eval controls module behavior
- no_grad disables graph recording
- Restore the caller's mode
- Preserve weights and existing gradients
Narration:
Evaluation must measure the model without silently changing the training experiment. Evaluation mode controls behavior such as dropout. A no-grad context separately prevents recording the differentiation graph. Neither operation is an optimizer update. Our helper uses both controls and restores the caller's root mode even if evaluation raises an error. The tests verify that weights and existing gradient buffers remain unchanged. This helper is for models with uniform training or evaluation mode, not a general utility for restoring deliberately mixed per-layer modes. Public training scripts add scheduling, clipping, checkpoints and other machinery, but those additions do not repair incorrect targets or contaminated evaluation data.

## Slide: Summary, explain and apply
- Align targets and compute raw logits
- Measure loss and differentiate
- Update parameters with a deliberate reset schedule
- Evaluate the claim on appropriate data
Narration:
The complete story is now concrete. Correctly aligned targets define the task. Forward computes scores, the loss measures fit, backward accumulates derivatives, and the optimizer changes parameters. The gradient reset schedule determines which contributions belong to an update. Evaluation tells you something only about the examples and objective you actually measured. Run the lab, then explain the three-candidate update without looking at the code. Your study partner will find one gap and ask one focused follow-up. After that, diagnose unequal microbatches whose means were averaged equally. Accurate explanation plus correct application is the learning goal. Receiving this chapter or seeing a passing lab is not itself evidence of mastery.
