## Slide: The transformer block
- LLM Fundamentals, chapter 6 of 10
- Preserve the state, then add useful corrections
- Residuals, LayerNorm, and a position-wise MLP
Narration:
Attention lets token positions exchange information, but attention alone is not the whole transformer block. Today we wrap it in a repeatable unit. We keep a token's current representation, add a contextual correction, then add a nonlinear feature correction. We will build this on the CPU with random weights and inspect every intermediate. This is a test of how the machine is wired, not a claim that it has learned language. By the end, you should be able to trace the state through the block and predict which operations can mix information between positions.

## Slide: Start with addition
```text
state   = [1.0,  3.0]
update  = [0.5, -0.5]
result  = [1.5,  2.5]
```
- A residual update preserves a direct path
Narration:
Start with the smallest operation. The current state has two numbers, one and three. The branch proposes a correction of zero point five and negative zero point five. Add the corresponding entries and you get one point five and two point five. We call this a residual connection. The branch does not have to reconstruct the entire state. It can supply a correction, including no correction at all. But nothing here forces that correction to be small. This is ordinary addition, and a large or opposing correction is still possible.

## Slide: A direct derivative path, not a guarantee
```text
y = x + F(x)
derivative = 1 + F'(x)
8 layers, branch scale 0.5:
plain gradient     0.00390625
residual gradient 25.62890625
```
Narration:
The direct path matters during backpropagation too. The derivative of the output is one plus the derivative of the branch. In our controlled scalar experiment, the plain chain keeps multiplying the gradient by zero point five. The residual chain instead multiplies by one point five. Look at the resulting numbers: the residual derivative grows. So the correct lesson is not that residual connections prevent all gradient problems. They add a direct route. Our cancellation experiment makes that even clearer: a branch that negates its input cancels the direct path and produces a zero gradient.

## Slide: Normalize one row by hand
```text
row       [1, 3, 5, 7]
mean      4
centered  [-3, -1, 1, 3]
variance  5
```
- Average squared deviations over the feature count
Narration:
Now prepare a branch's input. Take the row one, three, five, seven. Average its values and the mean is four. Subtract four from each entry and the centered row is negative three, negative one, one, three. Square those deviations and average them to obtain a population variance of five. The word population matters here: we divide by the number of features, not by one fewer. This is the statistic defining LayerNorm. A sample variance calculation answers a different question and would not match the operation we are building.

## Slide: Epsilon and learned affine parameters
```text
z = (x - mean) / sqrt(variance + epsilon)
output = gamma * z + beta
normalized variance: 0.999998
gamma=2, beta=3: output mean 3.000000
```
Narration:
Divide each centered value by the square root of the variance plus a small positive epsilon. Epsilon prevents division by zero when a row is constant. It also means the normalized variance is only approximately one. Then the module applies a learned scale and offset to each feature. These are called gamma and beta. Our example sets the scale to two and the offset to three, so the output mean becomes three. Remember the distinction: the standardized values and the final affine output are not the same thing. LayerNorm does not promise that every final output has zero mean.

## Slide: Which axis is normalized?
```text
input: (batch, tokens, features)
lab:   (2, 4, 8)
LayerNorm(8): features only, separately per row
```
- Shared affine parameters, separate row statistics
Narration:
Our tensor has a batch axis, a token-position axis, and a feature axis. LayerNorm over the feature width computes separate statistics for each token row. It does not average over other positions or other sequences. The learned scale and offset are shared across positions, but the statistics are not. That difference is easy to miss. If you accidentally normalize over tokens as well as features, an earlier position can read future information through the mean and variance. The output shape will still look correct. That is why a shape check alone cannot validate a causal model.

## Slide: The complete pre-norm block
```python
a = attention(norm1(x))
y = x + a
m = mlp(norm2(y))
out = y + m
```
Narration:
Here is the whole block. First normalize a view of the incoming state, run attention, and add its update to the original state. Then normalize a view of that updated state, run the MLP, and add its update. Notice two details. The additions use the persistent state, not the normalized view. And the second normalization receives the post-attention state, not the original input. We call this pre-norm because normalization happens before each learned branch. Normalizing after the residual addition is a different architecture, with different forward and backward behavior.

## Slide: Reuse attention, inspect the wiring
- 2 heads, width 4 per head
- Every exposed intermediate: (2, 4, 8)
- Existing course Block parity error: 0.000000
Narration:
We reuse the course's existing attention implementation rather than rebuilding it here. It projects the full input into queries, keys and values, applies the causal attention calculation, rejoins the heads, and returns an update of the original width. The new inspectable block names each intermediate so you can follow the data. For a meaningful comparison, we copy its exact parameters into the existing course block and compare outputs. The reported error is zero. Comparing independently initialized blocks would not test the wiring, because their random weights would already be different.

## Slide: The MLP changes features within a row
```text
(2, 4, 8) -> (2, 4, 32) -> (2, 4, 8)
Linear -> GELU -> Linear
Same weights at every position
```
Narration:
The MLP expands the feature width, applies a nonlinear activation, and projects back to the original width. In this lab the features expand from eight to thirty two and return to eight. Expansion supplies intermediate features to transform. Returning to the original width makes the residual addition possible. The same weights operate at every token position. We are not allocating a separate little network to every token. Local computation means that a call processes each input row independently. It does not mean that the row lacks context, because attention has already contributed contextual information.

## Slide: Why the nonlinearity matters
```text
GELU([-1, 0, 1])
[-0.158655, 0.0, 0.841345]
```
- Without activation, two affine maps collapse into one
- Negative inputs need not become zero
Narration:
The activation here is GELU, a smooth function that scales each hidden value according to the standard normal cumulative distribution. It is deterministic, not a random gate. Notice the negative input in the lab: its output is negative rather than exactly zero. More importantly, the nonlinearity prevents the pair of affine maps from collapsing into a single affine map. Simply making an intermediate layer wider would not achieve that on its own. This statement is about the MLP's internal maps. Attention already has input-dependent weights, so we are not claiming attention itself is linear in the original input.

## Slide: A causal test and a broken control
```text
Change one feature at the final position.
Correct stack, earlier difference: 0.000000
Wrong cross-time norm:            1.571958
```
Narration:
Now test a dependency rather than a shape. Change one feature at the final position and run the same stack again. Earlier outputs should remain unchanged, because the block is causal. The final position does change, confirming that our perturbation actually did something. Then run the same idea through deliberately wrong normalization that includes the time axis. Earlier values change. That broken control is useful: it shows the test can detect future information leaking through statistics. A correct attention mask cannot undo a leak that occurred earlier in the computation. Causality is a property of the entire path.

## Slide: Zero corrections recover the input
```text
Zero both branch output weights and biases.
block(x) = x
input gradient of output.sum() = ones
```
Narration:
Make a copy of the block and zero the final projection weights and biases in both branches. The internal attention and MLP calculations still exist, but neither branch contributes an update. The full block should now return its input exactly. Sum that output and backpropagate: every input coordinate should receive a gradient of one. The lab checks both properties. This is a precise way to test the residual wiring. Remember to zero the biases too, because a nonzero output bias would still add a correction even when the output weights were zero.

## Slide: Backward is not training
- Diagnostic mean-square loss: 1.282190
- Every parameter gradient is present and finite
- No optimizer step, no language capability claim
Narration:
For the ordinary block, we compute a mean-square diagnostic scalar and run a real backward pass. Every parameter has a present and finite gradient. Both the attention and MLP input projections also receive nonzero gradients. That checks that the computation is connected and numerically finite in this example. It does not establish language ability or convergence. Training needs an appropriate objective, data, optimizer updates, and evaluation. The lab does not quietly turn a gradient check into a model benchmark. Its parameter counts are also structural facts, not measurements of speed or importance.

## Slide: Read real implementations carefully
- nanoGPT: pre-norm attention and MLP residuals
- PyTorch: normalization axes follow normalized_shape
- Original Transformer and pre-norm are not identical
Narration:
The public nanoGPT source shows this same pre-norm pattern, with a final normalization after the block stack. PyTorch documents the exact normalization axes, variance convention and learned affine parameters. The original Transformer paper uses a different normalization placement. Read the actual implementation before assuming that all transformer blocks are identical. For the guided lecture, Karpathy's published chapter markers identify feedforward layers, residual connections and LayerNorm. The chapter provides verified links and explains that transcript access was blocked. Those are topic markers, not fabricated quotations or evidence about private model architectures.

## Slide: Summary and explain-back
- Preserve state; attention adds context
- Normalize a branch's view over features
- Shared MLP transforms each contextualized row
- Predict first, then test one change
Narration:
The block keeps a state and adds corrections. Attention communicates across permitted positions. Normalization prepares each branch's view using the feature axis. The shared MLP transforms each contextualized row, and residual additions preserve a direct path. Run the lab, then make one prediction before changing a copy. For example, predict whether the identity check survives after removing one residual addition. Finally explain how an MLP can act independently at every position while still using earlier context. An accurate explanation followed by a correct application is evidence of understanding. Receiving this lesson alone is not.
