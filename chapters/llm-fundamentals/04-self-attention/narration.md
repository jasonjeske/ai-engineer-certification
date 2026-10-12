## Slide: Self attention, from the inside
- LLM Fundamentals, chapter 4 of 10
- Build one head, then try to break its invariants
- Constructed numbers, not a trained language model
Narration:
A token needs a way to use its context. The bigram model used the current token to predict what comes next. Today we build an operation that mixes information from the available prefix. We will calculate a tiny example by hand, run the same arithmetic in code, and test whether future inputs can leak into earlier outputs. The example is deliberately constructed. We are proving how attention works, not claiming that a random head understands language. Keep that distinction visible throughout the walkthrough.

## Slide: Begin with an average
```text
Values: [2, 0]   [0, 4]   [6, 2]
Final equal weights: [1/3, 1/3, 1/3]
Final average: [2.666667, 2.000000]
```
- Earlier positions average only their allowed prefix
Narration:
Start with a familiar operation: an average. The slide shows three value vectors, each containing two coordinates. At the final position, equal shares produce the average printed by the lab. At an earlier position, only the available prefix participates. This already lets information travel between positions. But a fixed average treats all available sources alike. A code assistant might need a variable definition more than an unrelated comment. The next step is to calculate the shares from the content rather than fixing them in advance.

## Slide: Routing and payload
- Query: what this receiving position seeks
- Key: what a source can match
- Value: the information it contributes
- Self-attention: all come from the same sequence
Narration:
Make three views of each input vector. The query describes what a receiving position seeks. A key describes what a source can match. A value contains the information that source contributes. Those descriptions are analogies for numeric vectors, not literal English questions or labels. Queries and keys determine the routing weights. Values are the payload. In self attention all three views come from the same sequence. Self does not mean each token can only read itself. It means the source of these projected representations is shared.

## Slide: Project the input
```text
X: (3, 6)
Q = X W_Q: (3, 2)
K = X W_K: (3, 2)
V = X W_V: (3, 2)
```
- The same projection acts on every position
Narration:
Our input has three rows and six features per row. Each projection turns those six features into two coordinates. The same projection parameters apply at every position. In the worked example, the matrices simply select different pairs of input columns. That choice makes the arithmetic easy to inspect. In a trained model, those matrices are learned and generally mix features. The separate query, key, and value transformations let matching behavior differ from the information delivered. A later part of the lab checks randomly initialized trainable projections as well.

## Slide: Scores compare queries with keys
```text
S = Q K^T / sqrt(d_k)
Final scaled scores:
[0.000000, 0.693147, 1.386294]
```
- Row = receiver; column = source
- Scores are not probabilities
Narration:
Compare a query with a key by multiplying matching coordinates and adding them. That is a dot product. Matrix multiplication computes every such comparison at once. The transpose makes the source keys into columns. Each score row belongs to one receiver. We divide by the square root of the query and key width to control how score magnitudes grow with width under the usual initialization assumptions. This is not cosine similarity. The resulting numbers are compatibility scores, and they have not yet been turned into probabilities or shares.

## Slide: Hide the future first
```text
Allowed sources:
[yes, no,  no ]
[yes, yes, no ]
[yes, yes, yes]
```
- Forbidden score becomes negative infinity
- The diagonal is allowed
Narration:
During next token prediction, the current input is known, but the following target is not. A receiver may therefore use itself and earlier positions, not later positions. The allowed pattern is the lower triangle including the diagonal. Replace forbidden scores with negative infinity before normalization. Keeping the diagonal is important because it leaves every row at least one valid source. This mask controls information flow by position. It does not decide whether a document is trustworthy or whether an agent should be allowed to execute a tool.

## Slide: Softmax makes shares
```text
Final exponentials: [1, 2, 4]
Sum: 7
Final weights: [1/7, 2/7, 4/7]
```
- Normalize across source positions, not the batch
Narration:
Softmax exponentiates each score and divides by the sum of exponentials in the same row. Our scores were chosen so the final exponentials are one, two, and four. Their sum is seven, so the shares are one seventh, two sevenths, and four sevenths. At the middle position, the final source is forbidden, and only the first two exponentials remain. Each receiver has its own distribution over source positions. Different examples in the batch must not mix. Use the final tensor axis for this normalization in our layout.

## Slide: Mix the values
```text
First coordinate:
(1/7)*2 + (2/7)*0 + (4/7)*6 = 26/7
Second coordinate:
(1/7)*0 + (2/7)*4 + (4/7)*2 = 16/7
Output: [3.714286, 2.285714]
```
Narration:
Now multiply each value vector by its share and add the results. Work coordinate by coordinate, as shown on the slide. The final vector is the measured output from our lab. Notice that we did not select only the value with the largest weight. We blended several sources. Also notice that this vector is not a token identifier or a probability over the vocabulary. Attention updates a representation. A complete language model needs later transformations and a vocabulary output layer to turn representations into next token predictions.

## Slide: The core in code
```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(q.shape[-1])
scores = scores.masked_fill(~allowed, -float('inf'))
weights = torch.softmax(scores, dim=-1)
out = weights @ v
```
Narration:
Here is the entire mixing operation. Transpose only the last two key dimensions so batch dimensions remain intact. Divide by the square root of the feature width. Fill the forbidden positions, which are the logical opposite of the allowed mask. Normalize across sources. Finally multiply by values. The reusable helper also checks shapes, because accidental broadcasting can produce plausible output with incorrect batch semantics. This implementation supports square self attention. It does not implement padding policies, a key value cache, or rectangular attention with shifted positions.

## Slide: Test causality by intervention
```text
Change only the final input.
Causal prefix max change:   0.000000
Unmasked prefix max change: 9.921041
```
- Unmasked run is a negative control
Narration:
Do not trust the mask just because it looks triangular. Change a future input and recompute the outputs. Earlier outputs should stay unchanged. The lab changes the final input's key and value features, and its causal prefix remains identical. Remove the mask and the same intervention changes the earlier outputs substantially. That second run is a negative control. It proves the chosen modification can reveal leakage. A test that never changes anything, even in a broken implementation, would not give us this evidence.

## Slide: Two mistakes that look plausible
```text
Zero-score mask, first weight row:
[0.333333, 0.333333, 0.333333]

Mask after softmax, row sums:
[0.142857, 0.428571, 1.000000]
```
Narration:
A zero score does not mean zero weight, because the exponential of zero is one. The first broken version accidentally lets the first receiver use all sources. Its weights still sum to one, so normalization alone cannot prove causality. The second broken version applies the mask after softmax. Forbidden positions already consumed part of the denominator. Removing them afterward shrinks the remaining row sum, and the scale may still depend on future scores. Mask before normalization, then test both the distribution and actual information flow.

## Slide: Weights are not the whole output
- Hold queries and keys fixed
- Increase the first value's first coordinate by 7
- All attention weights stay identical
- Final output's first coordinate rises by 1
Narration:
Here is a useful prediction exercise. Hold queries and keys fixed, but increase the first value's first coordinate by seven. The weights cannot change because values do not enter the score calculation. The final receiver gives that source one seventh of the mixture, so its first output coordinate increases by one. This is why an attention heatmap is not a complete explanation of the answer. It shows routing, but the routed vectors matter too. Later layers and residual connections create additional paths that this isolated head does not represent.

## Slide: Trainable is not trained
```text
Random batch input:  (2, 4, 6)
Output:              (2, 4, 2)
Weights:             (2, 4, 4)
Projection parameters: 36
All projections get finite nonzero gradients
```
Narration:
The next experiment uses real linear layers with randomly initialized parameters. It compares our explicit calculation with the library attention operation and finds agreement within floating point tolerance. A simple objective based on the output sends finite, nonzero gradients back to every projection. That proves the learning machinery is connected. It does not prove language learning, because we did not optimize a language objective. There is no trained semantic relationship to interpret here. Separate evidence that code is differentiable from evidence that a trained model solves a useful task.

## Slide: The matrix has a cost
```text
One dense float32 score matrix:
T=128: 16,384 entries,  65,536 bytes
T=256: 65,536 entries, 262,144 bytes
```
- These are not whole-model memory totals
- Optimized attention need not store the entire matrix
Narration:
Every receiver is compared with every source, so the explicit score matrix grows quadratically with sequence length. The lab calculates the storage shown here for one matrix. Doubling the sequence length quadruples that storage. Batches, heads, layers, and other activations add further costs. Optimized algorithms such as FlashAttention reduce memory traffic with tiled computation while preserving the attention operation. Our teaching implementation stores the matrix because we want to inspect it. The numerical comparison with the library is not a speed benchmark or proof that a particular accelerated kernel ran.

## Slide: Summary and your next test
- Project, score, scale, mask, normalize, mix
- Attention updates features, not token IDs
- Predict a value change, then run the lab
- Explain why the causal test needs a negative control
Narration:
The whole head now has a concrete meaning. Project the input into queries, keys, and values. Compare queries with keys, scale the scores, hide forbidden positions, normalize each row, and mix the values. Run the lab and read the intermediate results before changing anything. Then predict the value intervention in your own words and test it. Explain why the masked prefix stays unchanged and why the unmasked negative control matters. Your study partner will look for an accurate explanation and a correct new application, not count watching this walkthrough as mastery.
