## Slide: The bigram model
- LLM Fundamentals, chapter 3 of 10
- Count neighbors, predict, sample, test
- A real model without a neural network
Narration:
A table of counts can predict text. Today we will build that table, turn its rows into probabilities, and use those probabilities to generate characters. Then we will test whether the model learned useful structure on text it did not fit. This is chapter three of our language model course. You already know how text becomes token identifiers and how an identifier selects a row. Now the row gets a specific meaning. It tells us which token is likely to come next. We will keep the mechanism small enough to inspect completely.

## Slide: Count one tiny string
```text
text:   abacaba
pairs:  ab ba ac ca ab ba

current     next a   next b   next c
   a           0        2        1
   b           2        0        0
   c           1        0        0
```
Narration:
Start with the little string on the slide. Read every pair of neighboring letters. After the letter a, the letter b occurs twice and the letter c occurs once. After b, a occurs twice. After c, a occurs once. The final letter has no successor, so it contributes no pair. We have seven tokens but only six transitions. The table records those events. Its rows name current characters, and its columns name next characters. That orientation matters. Every prediction will begin by selecting a row, not a column.

## Slide: A row answers a conditional question
```text
After a: counts [0, 2, 1]
Observed successors: 3
Raw P(b | a) = 2 / 3 = 0.666667
```
- Divide by this row's total
- Not the total of the whole table
Narration:
Suppose the current character is a. Which character should follow? The simplest estimate repeats the observed proportions. The letter b followed a twice among three observed successors, so its estimated probability is two thirds. We call this a conditional probability because the current character is given. Divide by the total of that row. Dividing by the entire table would answer a different question about how common a pair is overall. Now we can name our model. A bigram model predicts the next token using only the immediately preceding token.

## Slide: Unseen does not mean impossible
```text
Add one to every count:
[0, 2, 1] -> [1, 3, 2]

New denominator: 3 + 3 = 6
Probabilities: [1/6, 3/6, 2/6]
```
Narration:
Our raw table has a problem. Some pairs never occurred, so their estimated probabilities are zero. That is stronger than saying we have not seen them. It says they are impossible. Add one to every count before dividing, and the row now leaves room for every candidate. The new denominator includes all the added counts. For our toy row, the total goes from three to six. The probability of b becomes one half. This is add one smoothing. The added counts express a modeling choice, not evidence that those missing pairs actually occurred.

## Slide: Keep the denominator honest
```python
weights = counts + alpha
totals = weights.sum(dim=1, keepdim=True)
probs = weights / totals
```
- alpha is added to every cell
- Every probability row sums to 1
- Empty rows need positive smoothing
Narration:
Here is the entire normalization step. Alpha is the amount added to every cell. Sum across the columns to get each row's new total, keeping the extra dimension so division broadcasts in the intended direction. Each row must sum to one. That check catches many implementation mistakes, but it does not prove the predictions are useful. A uniform distribution also sums to one. A context that never appeared needs special care. With positive smoothing it becomes uniform. Without smoothing, an empty row has a zero denominator, so our implementation raises an error rather than pretending the result is valid.

## Slide: Score the observed continuation
```text
Given initial a, score the suffix ba:

P(b | a) = 0.5
P(a | b) = 0.6
P(ba | initial a) = 0.3

Mean negative log likelihood = 0.601986
```
Narration:
How do we evaluate the predictor without choosing a flattering sample? Give it actual target characters and measure their probabilities. For the sequence on the slide, the first character is supplied. The model assigns one half to the next transition and six tenths to the transition after that. Multiply them to get three tenths for the suffix. We use negative logarithms so long products become sums, then average over predicted transitions. This is mean negative log likelihood. Lower is better because the model gave more probability to what actually happened. We are not scoring the supplied first character.

## Slide: Loss is not an error percentage
- Natural logs give nats per predicted token
- Perplexity = exp(mean loss)
- Zero probability for a target gives infinite loss
- Compare the same tokenization and scoring convention
Narration:
The score is not an error percentage. It measures the probabilities assigned to observed targets. A model can choose the same favorite character as another model and still have a different loss because its confidence differs. Perplexity is the exponential of the average loss, another way to express the same score. If a target has zero probability, its negative logarithm is infinite. Our lab demonstrates that failure on an unseen pair and then makes the loss finite with smoothing. Do not compare these character scores directly with scores from a different tokenizer and call one model better.

## Slide: Generate by feeding the choice back
```python
out = [start_id]
for _ in range(new_tokens):
    row = probs[out[-1]]
    nxt = torch.multinomial(row, 1, generator=rng)
    out.append(nxt.item())
```
Narration:
Generation is a small feedback loop. Begin with a supplied character. Retrieve its probability row. Draw the next character, append it, and let that new character choose the next row. Repeat until you reach the requested length. The output list grows, but the model still looks only at its final token. No counts change during this process. Sampling uses the fitted model; it does not train it. The explicit random generator makes the experiment reproducible in this environment. The same seed is not a promise of identical strings on every device or library version.

## Slide: Sampling and greedy choice differ
```text
Greedy output:
ababababababa

Independent draws after a:
[0.164, 0.5044, 0.3316]
```
- Greedy always takes the largest probability
- Sampling sometimes chooses alternatives
Narration:
Greedy decoding always selects the most probable next token. Our toy table sends it into the repeating pattern shown here. Sampling can take less probable transitions and escape that particular cycle. To check the sampling operation itself, the lab draws repeatedly from the same row. The measured frequencies are close to the row's probabilities, but not exactly equal. Finite samples fluctuate. During real sequence generation, we do not keep drawing from that fixed row. Each chosen character changes the context for the next draw. The distribution and the decoding policy are different parts of the system.

## Slide: Separate fitting, tuning, testing
```text
Tiny Shakespeare slices:
training      200000 characters
validation     20000 characters
test           20000 characters

Training vocabulary: 62 characters
```
- Fit counts on training only
- Choose smoothing on validation only
Narration:
Now use real text from the local Shakespeare dataset. The training slice supplies both the vocabulary and the counts. Validation chooses the smoothing strength, and a separate test slice measures the selected model. We do not count pairs across those boundaries. The vocabulary contains sixty two characters in this training slice. Unknown characters would cause an explicit error; smoothing cannot invent a missing token identifier. This distinction is important. An unseen pair among known tokens is a probability problem. A character outside the vocabulary is a tokenization problem with a separate policy.

## Slide: Compare useful baselines
```text
Held-out test loss, nats per token:

uniform     4.127134
unigram     3.389097
bigram      2.629084
```
- Uniform ignores frequency and context
- Unigram learns frequency, ignores context
- Bigram uses the previous character
Narration:
The uniform baseline gives every character equal probability. The unigram baseline learns which characters are common, but gives the same prediction after every context. The bigram learns a separate distribution for each current character. The measured test losses are on the slide. Lower is better, and the bigram wins this comparison. That is evidence that neighboring characters contain useful predictive information in these text slices. It is not proof of language understanding. The generated sample still wanders through recognizable fragments without maintaining meaning. The score and the sample tell us different things about the same limited model.

## Slide: The limitation more data cannot fix
```text
context ab -> row b
context cb -> row b

Different histories, identical predictions
```
- Only the last token selects the row
- More counts refine the same row
- Longer context needs a different representation
Narration:
Consider two histories with different prefixes but the same final character. Our model must give them the same next token distribution, because both select the same row. Even if the complete histories are stored elsewhere in the program, the prediction function does not use them. More data can improve the estimated probabilities within that row. It cannot split the row according to information the model never receives. This is an architectural limitation, not a lack of training effort. The next chapters change how context is represented so earlier tokens can influence the prediction.

## Slide: The bridge to an embedding table
```python
logit_table = torch.nn.Embedding.from_pretrained(
    smoothed.log(), freeze=True
)
logits = logit_table(x)
loss = F.cross_entropy(logits, y)
```
- softmax(log P) recovers P
- Cross entropy matches direct negative log likelihood
Narration:
An embedding retrieves a row, so we can store next token scores in an embedding table. Here we fill it with the logarithms of our smoothed probabilities and freeze it. Softmax exponentiates those values and normalizes the row, recovering the same probabilities. Cross entropy gives the same loss as our direct calculation. This is a representation equivalence, not a training experiment. No optimizer runs. A neural bigram could learn the scores with gradients, but it would still use only the final token. Changing optimization does not automatically add context or memory.

## Slide: Boundaries and checks are part of the model
- Count within independent sequences
- Reject invalid IDs and probability rows
- Verify hand counts and hand loss
- Keep held-out data out of fitting
Narration:
The less glamorous details determine whether the experiment is trustworthy. Joining independent sequences can invent a transition that never occurred. Normalizing the wrong dimension can produce a plausible looking table with the wrong meaning. Scoring training data as if it were held out can exaggerate performance. Our lab and tests check those failure modes directly. They also check sampling reproducibility and agreement between the simple loop and the vectorized counter. Run the lab rather than only reading the output. Then predict one change before you make it, so the experiment tests your explanation instead of merely displaying more numbers.

## Slide: Summary: counts to a real predictor
- Count pairs, smooth, normalize rows
- Score actual targets on held-out text
- Sampling does not refit the model
- Last-token-only context is the hard limit
Narration:
We built a real predictor from counts. Each current token selects a distribution over the next token. Smoothing leaves room for unseen pairs, and held out loss lets us compare the fitted model with simpler baselines. Generation repeatedly draws a token and feeds it back in. None of that gives the model access to earlier history. Your next step is to explain the toy row in your own words, then apply the same rule to a new count row. Accurate explanation plus correct application demonstrates understanding. Receiving the chapter or producing a sample does not.
