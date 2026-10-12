## Slide: Multi head attention: many perspectives at once
- LLM Fundamentals, chapter 5 of 10
- Several routes through the same input
- Concatenate first, then mix features
Narration:
One attention head produces a weighted mixture of information from other positions. Today we keep that operation intact and run several independently parameterized versions. Each version can choose a different mixture. We preserve their results side by side, then learn how to combine their features. That is multi-head attention. We will calculate a tiny example, inspect a working implementation, and test it against PyTorch. Our tidy numbers come from hand-chosen weights, not from a model that learned grammar. The point is to make the mechanism visible before attaching stories about language to it.

## Slide: Why one mixture can be restrictive
- One head uses one source distribution per receiver
- All value coordinates share that distribution
- Different heads can form different distributions
Narration:
Suppose a code completion could benefit from a variable definition and the structure of the current expression. One head gives each receiving position a single distribution over source positions. All coordinates of that head's values use those same source weights. Multiple heads allow different feature groups to use different mixtures. That is an architectural opportunity, not a guarantee of specialization. We do not assign a grammar expert and a coding expert. We provide separate parameters. Whether training makes them useful, redundant, or interpretable requires evidence from a trained model, which this small lab does not claim to provide.

## Slide: Every head can read the whole input
```text
Input:        (B, T, C)
Each head:    C -> d
Head count:   H
Equal widths: H*d = C
```
- Split projected features, not raw input slices
Narration:
Here is the most important correction to the usual picture. A head does not ordinarily receive only one isolated slice of the raw embedding. Its learned projections can read the entire input feature vector. The projected outputs are narrower. In our hand example, the input width is four and each of two heads has width two. Each query projection therefore maps four input features to two query features. When packed code appears to split the embedding into heads, look just before the split. A full-input linear projection has already happened. The split organizes those projected features.

## Slide: The familiar head, repeated independently
```text
query = input @ query_projection
key   = input @ key_projection
value = input @ value_projection
scores = query @ key.T / sqrt(head_width)
weights = softmax(mask(scores))
output = weights @ value
```
Narration:
Each head does the same job as last chapter. Queries and keys determine source compatibility. Divide each dot product by the square root of the head width, because that is the number of coordinates used in the dot product. Replace future scores with negative infinity, then normalize across source positions. Finally, use those weights to mix value vectors. Every head has its own projections, but all obey the same causal visibility rule. The heads do not consult one another while calculating scores. They operate independently until their outputs are joined and projected.

## Slide: Head zero, a hand calculation
```text
Values: [2,0], [0,4], [6,2]
Scores: [0, ln(2), ln(4)]
Weights: [1/7, 2/7, 4/7]
Output: [26/7, 16/7]
```
Narration:
Our three source values are deliberately simple. The first is two and zero, the second zero and four, and the third six and two. At the final position, head zero's scores are zero, the natural logarithm of two, and the natural logarithm of four. Exponentiation turns those into one, two, and four. Their sum is seven, so the weights are one seventh, two sevenths, and four sevenths. Multiplying and adding the value coordinates gives twenty six sevenths and sixteen sevenths. The lab checks those exact fractions before printing their rounded decimal forms.

## Slide: Head one, a different route
```text
Scores: [0, -ln(2), -ln(4)]
Weights: [4/7, 2/7, 1/7]
Output: [14/7, 10/7]
```
- Same values, different routing
Narration:
Head one uses the same queries and values, but the sign of its key projection is reversed. Its exponentiated scores are one, one half, and one quarter. Normalize those and the weights become four sevenths, two sevenths, and one seventh. Now the first source receives the strongest share instead of the last source. Applying the new weights gives fourteen sevenths and ten sevenths. Keeping the values identical isolates the effect of routing. A general trainable module also has separate value projections. Both patterns here were constructed by us, so neither is evidence of learned semantic specialization.

## Slide: Concatenation preserves separate routes
```text
Head 0: [26/7, 16/7]
Head 1: [14/7, 10/7]
Joined: [26/7, 16/7, 14/7, 10/7]
```
- No averaging, no voting, no winner
Narration:
Concatenation places the head results next to one another along the feature axis. It does not average them, and it does not select a winning head. The separate routes are still distinguishable. In this equal-width design, the joined representation is already the same width as the input. That is worth saying carefully: the output projection is not needed just to repair the width. Its purpose is to learn useful combinations of the collected features. If we summed the heads immediately, their separate differences would be lost before the next transformation could use them.

## Slide: The output projection can add and subtract
```text
Joined coordinates: [a, b, c, d]
Projected: [a+c, b-d, a-c, b+d]
Result:
[5.714286, 0.857143, 1.714286, 3.714286]
```
Narration:
The output projection is an ordinary linear transformation applied separately at each position. Our chosen matrix adds the first coordinates, subtracts the second coordinates, subtracts the first coordinates, and adds the second coordinates. This produces the displayed result, copied from the lab. Notice the negative signs. A head output is a weighted mixture of its values, but the final output need not be another positive average. The projection can subtract or amplify features. It also does not mix token positions. If every head output respects causality, this within-position projection cannot introduce a new dependency on a future token.

## Slide: Build genuinely independent heads
```python
self.heads = nn.ModuleList([
    SingleHead(width, head_width)
    for _ in range(num_heads)
])
cat = torch.cat(head_outputs, dim=-1)
output = self.proj(cat)
```
Narration:
The loop implementation creates a fresh head on every iteration and stores the heads in a registered module list. Registration matters because PyTorch must find their parameters for training, saving, and device movement. Repeating a reference to the same head object would share parameters rather than create independent ones. An ordinary Python list would not register the heads in this way. The forward calculation collects each head's result, joins the outputs along the feature dimension, and applies the projection. Our reusable module also returns intermediate weights and concatenated features so you can inspect the calculation directly.

## Slide: Packed heads are the same mathematics
```text
Projected Q: (B,T,C)
Reshape:     (B,T,H,d)
Transpose:   (B,H,T,d)
Weights:     (B,H,T,T)
Head output: (B,H,T,d)
Rejoin:      (B,T,C)
```
Narration:
The packed version groups projection weights into larger matrices. After projection, it introduces a head axis and moves that axis ahead of the token axis. Attention then runs across all batches and heads using the same matrix operations. After value mixing, we transpose back before joining head features. Skipping that transpose can preserve the final shape while scrambling which token owns which features. Our packed path uses exactly the loop version's parameters, and the tests compare intermediate values and gradients. Packing is a reorganization of the same computation, not a different kind of attention or a demonstrated speed improvement in this lab.

## Slide: Test whether the mask really works
```text
causal prefix max change 0.000000
unmasked prefix changes True
future nonzero weights 0
normalized rows True
```
Narration:
A plausible triangle is not enough. The lab changes the last input position and checks the complete projected output. Earlier rows remain unchanged with the causal mask. Then we remove the mask as a negative control, and those earlier rows change. That proves the chosen perturbation can expose leakage. We also check zero future weights and row normalization in every head. These are complementary tests, because a wrong implementation can preserve shape or normalization while violating the intended information flow. The regression suite separately checks that one batch example cannot influence another example.

## Slide: Compare against an independent implementation
```text
loop matches packed True
loop matches PyTorch True
per-head weights match PyTorch True
```
- Bias and dropout disabled for this comparison
- Boolean True means blocked in MultiheadAttention
Narration:
We copy our random projections into PyTorch's multi-head attention layer, with bias and dropout disabled. We request separate attention weights for each head rather than their average. We then compare outputs and weights within floating-point tolerance. The tests also compare input and parameter gradients. Pay attention to the mask contract: in this module's Boolean attention mask, true means blocked. Do not assume every attention API uses that convention. These comparisons establish numerical correctness for the tested cases. Finite nonzero gradients establish trainability. Neither result proves that a language task has been learned, because this experiment performs no such training.

## Slide: Head count changes different costs differently
```text
C=8, B=2, T=4, no bias
Heads   Width/head   Parameters   Scores
1       8            256          32
2       4            256          64
4       2            256          128
```
Narration:
Hold total feature width fixed and increase the number of heads. Each head becomes narrower. Across all heads, the query, key, value, and output projections still contain two hundred fifty six parameters in this example. But the explicit score tensors get larger, because each head has its own token-by-token matrix. The displayed score counts grow from thirty two to sixty four to one hundred twenty eight. Parameter count, intermediate storage, and execution speed are different quantities. Our lab counts parameters and score entries; it does not benchmark a production attention kernel or claim that more heads are always better.

## Slide: Predict an intervention before running it
- Set the fixture's query feature to zero
- Both final mixtures become equal
- Predict the projection's difference coordinates
Narration:
Now make a prediction. Set the hand fixture's constant query feature to zero at every position. All dot products become zero, so both heads give equal shares to allowed sources. At the final position they average the same three values and produce identical mixtures. Therefore the output coordinates that subtract one head from the other become zero. The lab runs this intervention and checks the result. This is a better understanding check than reciting tensor shapes: you have to connect queries, routing, values, concatenation, and output mixing to explain which quantities change and which stay fixed.

## Slide: Summary: routes first, feature mixing second
- Full-input projections, independent attention patterns
- Concatenate, then project within each position
- Test values, causality, gradients and costs separately
- Next: assemble a full Transformer block
Narration:
The whole mechanism is now visible. Each head projects the full input into its own query, key, and value space, forms a source distribution, and mixes values. Concatenation preserves those results side by side. The output projection learns how their features contribute at each position. A head number is not a semantic job description, and multiple heads are not multiple agents. To check your understanding, explain why the equal-query intervention makes the difference coordinates vanish. Then compare the loop and packed code with Karpathy's verified lecture moments in the chapter. The next step is placing this operation inside a complete Transformer block.
