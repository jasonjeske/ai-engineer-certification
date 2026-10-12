## Slide: From token IDs to vectors
- LLM Fundamentals, chapter 2 of 10
- Look up a row, then learn its numbers
- Geometry is useful, but not a meaning oracle
Narration:
A token identifier tells the model which entry you meant. It does not tell the model what that entry means. In this chapter we take the smallest possible next step. We keep a table of adjustable numbers, then use each identifier to select a row. That row is an embedding. We will inspect the lookup, measure the geometry, and actually change the table through training. Our training example is deliberately synthetic. It proves a mechanism, not that a tiny table suddenly understands English. That distinction will matter throughout the walkthrough.

## Slide: An address is not a measurement
- Vocabulary: cat, dog, run, walk, the, a
- Token IDs select rows; their numerical distances do not matter
- Six rows, four coordinates, 24 parameters
Narration:
Think of each token as having a labeled drawer. The drawer contains a short list of numbers. Asking for the token retrieves that list. Our vocabulary contains cat, dog, run, walk, the, and a. Each gets four coordinates, so the table has twenty four adjustable entries. The numerical distance between the identifiers tells us nothing about the relationship between the words. We could reorder the dictionary, provided we moved the rows consistently. The numbers inside each drawer are what the rest of the model can calculate with and what training can change.

## Slide: Start random, not meaningful
```text
cat -> [1.927, 1.487, 0.901, -2.106]
untrained cosine(cat, dog) = 0.393828
untrained cosine(the, a)   = -0.545457
```
- Real lab output, rounded for display
Narration:
Here is a real row from the lab before training. The coordinates look precise, but precision is not evidence of meaning. They came from a seeded random initialization. The comparisons underneath are random geometry too. A plausible neighbor can happen by chance, and an implausible neighbor is not a bug in the cosine calculation. We have not asked the table to learn anything yet. Reproducibility lets us repeat an experiment; it does not make an experiment meaningful. We need an objective and evidence about performance on that objective before describing these numbers as learned representations.

## Slide: Follow the tensor shapes
```text
IDs:             [4, 0, 2]
sequence output: (3, 4)
batched output:  (2, 3, 4)
table shape:     (6, 4)
```
Narration:
The phrase on our toy input is the cat run. It selects the rows addressed by four, zero, and two. A sequence of three identifiers becomes three rows, each containing four numbers. Put two sequences into a batch and the result has batch, token position, and coordinate dimensions. Notice that the table itself does not become bigger when the batch grows. Occurrences share parameters. If cat appears again, it selects the same row again. The lookup returns identical static vectors for identical identifiers at fixed weights, regardless of which neighboring words supplied the context.

## Slide: Lookup equals row selection
```python
vectors = table(ids)
selectors = F.one_hot(ids, num_classes=6).float()
assert torch.equal(vectors, selectors @ table.weight)
```
- One-hot multiplication explains lookup; indexing avoids the zeros
Narration:
We can prove the lookup explanation with another representation. Make a selector containing a one at the requested index and zeros everywhere else. Multiplying that selector by the table keeps exactly the selected row. Stack selectors for a sequence and the same matrix multiplication selects several rows. The assertion on screen passes in the lab. This is a mathematical equivalence, not a recommendation to allocate enormous arrays of zeros. Direct indexing is the simple implementation. Keep the objects separate: the identifier is an integer address, while the selected row contains floating point parameters.

## Slide: Calculate cosine by hand
```text
u = [3, 4]     v = [4, 3]
dot(u, v) = 24
norm(u) = 5   norm(v) = 5
cosine = dot / (norm(u) * norm(v)) = 0.960
```
Narration:
Now we need a way to compare directions. A dot product multiplies matching coordinates and adds the results. A vector's norm is its length, calculated from the square root of the sum of squared coordinates. For the vectors on screen, the dot product is twenty four and each length is five. Divide the dot product by both lengths to get cosine similarity, zero point nine six. The lab calculates this directly, then checks the result against PyTorch. Cosine is the cosine of an angle, not the angle itself and not a probability of shared meaning.

## Slide: Direction is not length
```text
cosine(u, 2*u) =  1.000
dot(u, 2*u)    = 50.0
cosine(u, -u)  = -1.000
```
- Positive scaling leaves cosine unchanged
- Zero vectors have no mathematical direction
Narration:
Stretching a vector by a positive amount does not change its direction. Cosine removes that stretch by dividing by the lengths. A raw dot product does not, so its ranking can change when vector lengths differ. Reversing a vector points it in the opposite direction and produces negative one. Those are geometric facts, not rules about dictionary antonyms. There is also a boundary case: a zero vector has no direction. Our helper rejects it. PyTorch uses an epsilon convention to avoid division by zero, which is useful numerically but does not create a meaningful angle.

## Slide: Self-similarity is not learning
- Nonzero random rows already have self-cosine 1
- A nearest-neighbor query must exclude itself
- A geometric identity is not a language evaluation
Narration:
Here is the easiest way to fool yourself. Compare every token with itself, get a score of one, and announce that the model understands the vocabulary. The calculation would pass before training too. A nonzero vector points in the same direction as itself whether its coordinates were learned or randomly drawn. That makes self similarity an arithmetic check, not a learning test. When we search for a nearest other token, we explicitly exclude the query row. Otherwise the search can return itself and look impressively accurate while answering the wrong question.

## Slide: Repeated tokens share gradients
```text
loss = sum(table([0, 0, 1]))
grad cat: [2, 2, 2, 2]
grad dog: [1, 1, 1, 1]
other rows: all zero from this loss
```
Narration:
To see how a lookup learns, first use an intentionally simple loss. Select cat twice and dog once, then sum every selected coordinate. Each cat parameter contributes twice to the sum, so its gradient is two. Each dog parameter contributes once, so its gradient is one. Unselected rows contribute nothing to this particular calculation. Backpropagation routes feedback into the floating point entries, not into the integer addresses. Repeated occurrences accumulate because they share parameters. This probe is not a language objective. It isolates the mechanism so we can see exactly where the feedback goes.

## Slide: A transparent training fixture
```text
columns: purrs, barks, quickly, slowly
cat counts: [8, 2, 0, 0] -> [0.8, 0.2, 0.0, 0.0]
dog counts: [7, 3, 0, 0] -> [0.7, 0.3, 0.0, 0.0]
```
- Hand-authored targets, not measured English
Narration:
Now train something with a visible target. We supply artificial context counts, then divide each row by its total to make a target pattern. The cat and dog patterns are deliberately similar. The motion words use the other coordinates, and the articles get identical targets. These are hand authored teaching data, not observations about real English. In particular, the labels do not claim a measured frequency of cats barking. We chose transparent patterns so you can trace the cause of the final geometry. In a real language model, the objective usually reaches the embedding through other learned computations instead.

## Slide: Write the update, no magic box
```python
loss = 0.5 * (table(ids) - targets).square().sum()
table.zero_grad(set_to_none=True)
loss.backward()
with torch.no_grad():
    table.weight -= lr * table.weight.grad
```
Narration:
Our loss is half the sum of squared differences between current entries and target entries. Squaring makes errors contribute positively. The factor of one half makes the derivative especially simple: current value minus target value. Clear old gradients, differentiate the loss, then subtract learning rate times gradient. If a weight is above its target, that correction pushes it down. If it is below, the correction pushes it up. The no grad context prevents the update itself from building another differentiation graph. Every line has a job. Nothing here changes the tokenizer or the token identifiers.

## Slide: Measure the result honestly
```text
initial loss:              14.9425401688
loss after 10 updates:     0.1722757369
maximum final target error: 0.000000119
trained cosine(cat, dog):   0.987241
trained cosine(the, a):     1.000000
```
Narration:
After the updates, the loss has fallen and the table closely matches the supplied targets. The lab prints a separate maximum error so a loss rounded to zero is not mistaken for exact equality. The new similarities reflect the geometry we supplied. Cat and dog are close in direction; the article rows match because their targets match. That is successful training on this task. It is not evidence that we discovered all these relationships from a natural language corpus. A convincing experiment states both what changed and what its design does not allow us to conclude.

## Slide: Test the explanation by intervention
- Change a target: the learned geometry should follow it
- Stop updates: random rows stay random
- Relabel IDs consistently: retrieved vectors stay the same
- Rotate every row: cosines stay the same
Narration:
A good explanation survives changes to the experiment. Replace the dog's target with the motion pattern and predict where its vector will move. Stop training entirely and ask which checks should fail. Relabel the token identifiers and move the rows consistently; the retrieved vectors stay unchanged. Rotate every row with the same orthogonal transformation and the pairwise cosines stay unchanged too. This last test does not authorize changing one layer of a complete model in isolation. It shows why coordinate labels are not universal meanings, even when the relationships between vectors remain useful.

## Slide: Where the simple table stops
- Static lookup: same token, same row
- Transformer blocks add context-dependent computation
- Retrieval encoders represent whole texts
- Similarity does not establish truth or authority
Narration:
Our shared GPT code starts with token and position embeddings, then applies transformer blocks. Those later computations create representations that depend on surrounding text. That is different from this static lookup. A model used for document retrieval also needs a suitable representation of the whole query or passage, not an arbitrary token row. Matching vector widths does not make different encoders compatible. And a high similarity score cannot tell an agent that a source is trustworthy or that instructions inside a passage are authorized. Representation, relevance, and authority are separate questions an engineer must not collapse into one number.

## Slide: Summary and your next experiment
- IDs are addresses; embeddings are adjustable rows
- Cosine compares directions, not truth or probability
- Gradients change the rows through an objective
- Explain why fixed IDs can still support learning
Narration:
The complete idea is small enough to explain without jargon. Use a token's address to retrieve adjustable numbers. Compare directions when that metric suits the task. Let a loss send feedback into the selected parameters, accumulating contributions when tokens repeat. Then evaluate what the training actually achieved. Run the lab and read its intermediate output before changing anything. Predict the effect of a changed context target, then test your prediction in a separate copy. Finally, explain why a lookup table can learn even though its identifiers never change. Your explanation and a correct new application, not delivery of this video, establish understanding.
