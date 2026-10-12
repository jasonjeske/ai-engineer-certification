## Slide: Generation, sampling text from your model
- LLM Fundamentals, chapter 9
- Frozen weights, changing context
- Predict, choose, append, repeat
Narration:
A trained language model does not hand you a finished paragraph. It gives you scores for possible next tokens. Another piece of code chooses a token, appends it to the prompt, and asks the model again. That is generation. Today we will inspect this loop, first with a distribution small enough to calculate by hand, then with the real Shakespeare model saved in the course. We will not train anything. The important separation is between what the model predicts and how our program chooses from those predictions.

## Slide: Take the final position
```python
all_logits, _ = model(prompt)
last = all_logits[:, -1, :]
```
- Prompt: ROMEO followed by a colon and newline
- Full shape: (1, 7, 65)
- Last-position shape: (1, 65)
Narration:
The prompt is the name Romeo, a colon, and a newline. Our character tokenizer turns it into seven token IDs. The model returns a score for every vocabulary candidate at every input position. Here the dimensions are one sequence, seven positions, and sixty five possible characters. To continue the whole prompt, we need only the last position. Earlier positions predict after earlier prefixes. Selecting the wrong position would still give a valid looking row of scores, but it would answer the wrong next token question.

## Slide: Weights become probabilities
```text
weights = [4, 2, 1, 1]
sum = 8
probabilities = [0.5, 0.25, 0.125, 0.125]
```
- Labels: A, B, C, D
- Sampling is weighted, not uniform
Narration:
Start with four tokens, called A, B, C, and D. Give them weights four, two, one, and one. The total is eight. Divide each weight by that total and you get the probabilities on the slide. A is the favorite, but a weighted draw can still choose B, C, or D. These probabilities are not instructions to generate a particular count of each token in a short sequence. They describe the chances of the next draw. Our model normally gives logits instead of these positive weights.

## Slide: Softmax and temperature
```text
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```
- z: next-token score
- T: positive temperature
- Normalize across vocabulary candidates
Narration:
A logit is a real valued score, not a probability. Softmax exponentiates the scores and divides by the sum of those exponentials. Temperature enters before this normalization: divide each score by the same positive temperature. In our hand example, the logits are logarithms of four, two, one, and one. At temperature one, exponentiation recovers those original weights. Nothing about the learned model has changed. We are manipulating the scores returned by its forward pass. The denominator makes the final probabilities sum to one.

## Slide: Work the cold case by hand
```text
T = 0.5
weights = [16, 4, 1, 1]
sum = 22
p = [0.727273, 0.181818, 0.045455, 0.045455]
```
Narration:
At temperature one half, dividing the logarithms by temperature doubles them. Exponentiating a doubled logarithm squares the original weight. Four becomes sixteen, two becomes four, and the ones stay one. The total is twenty two. Dividing by that total raises A's probability to about zero point seven three. A was already the favorite, and now its advantage is stronger. The lower ranked tokens are still possible. Low positive temperature is not the same thing as explicitly choosing the maximum every time.

## Slide: Hotter means flatter, not wiser
```text
T = 2
p = [0.369398, 0.261204, 0.184699, 0.184699]
```
- Equal logits stay equal
- Positive temperature preserves ranking
- Probability is not a truth score
Narration:
At temperature two, the positive weights become square roots instead. That reduces their relative differences, so the distribution becomes flatter. A remains the favorite, but the other options gain probability. We also measure distributional entropy in the lab to make this flattening visible. Do not translate entropy into intelligence or creativity. A distribution can spread its probability across many bad choices. Similarly, a sharp distribution can be confidently wrong. Temperature changes how we sample the model's beliefs; it does not check those beliefs against the world.

## Slide: Greedy is a separate choice
- Explicit argmax, no random draw
- Zero temperature is our helper's convention
- Ties choose the lower vocabulary ID
```text
greedy path:      0.600 * 0.510 = 0.306
alternative:     0.400 * 0.990 = 0.396
```
Narration:
Our helper treats zero temperature as an explicit greedy branch, not a division by zero. Greedy takes the largest score and resolves ties by choosing the lower vocabulary ID. But the best next token does not necessarily start the most probable complete sequence. The small tree on this slide is constructed, not a model benchmark. Greedy picks the first branch, whose complete path has probability zero point three zero six. The other branch reaches zero point three nine six. Local maximization missed the better complete path.

## Slide: Top-k cuts the candidate list
```text
top_k = 2
p = [0.666667, 0.333333, 0, 0]
```
- Keep highest-ranked IDs, then renormalize
- Stable tie breaking keeps an exact count
- k = 1 matches our greedy rule
Narration:
Top k keeps a fixed number of high ranked candidates. In the hand example, keeping two removes C and D, then renormalizes A and B. We still choose only one next token. Top k does not mean generating multiple tokens or always picking the token at rank k. Our implementation uses stable sorting, so ties are resolved by vocabulary ID and the count is exact. A score cutoff can retain extra tied candidates, which is a different convention. With only one surviving candidate, sampling matches greedy decoding.

## Slide: Top-p adapts to the distribution
```text
probabilities: [0.5, 0.25, 0.125, 0.125]
cumulative:   [0.5, 0.75, 0.875, 1.0]
threshold:    0.7
keep:         A and B
```
- Include the token that crosses the threshold
Narration:
Nucleus sampling keeps enough high probability candidates to reach a chosen amount of probability mass. Here the threshold is zero point seven. A alone reaches only one half, but A plus B reaches three quarters, so both survive. The crossing token must be included. Removing it is a common implementation bug. With a flat distribution over the same vocabulary, reaching this threshold requires three candidates instead of two. That changing candidate count is the point of nucleus sampling. After selecting the set, we renormalize and draw as before.

## Slide: Order changes the policy
```text
base = [0.60, 0.25, 0.15]
top_p=0.7 alone: keep first two
top_k=2, then top_p=0.7: keep first only
```
- Our order: temperature, top-k, top-p
Narration:
Combining knobs introduces another question: in what order are they applied? Our helper applies temperature, then top k, then nucleus filtering on the renormalized survivors. In this example, nucleus filtering alone keeps the first two tokens. But top k first removes the third. That renormalization raises the first token's probability above the nucleus threshold, so the combined policy keeps only it. Configuration values are not a complete description unless the processing order is also known. This is why our code documents and tests its order instead of leaving it implicit.

## Slide: The loop is small
```python
ctx = ids[:, -model.cfg.block_size:]
logits, _ = model(ctx)
probs = probabilities(logits[:, -1, :])
nxt = torch.multinomial(probs, 1, generator=rng)
ids = torch.cat([ids, nxt], dim=1)
```
- Simplified unfiltered sampled branch
- New ID becomes input on the next iteration
Narration:
This simplified sampled branch shows the whole feedback loop. Crop the visible context, run the model, select the final position, convert scores to probabilities, draw an ID, and append it along the time dimension. The next pass now sees that chosen token. The draw uses a local random generator so unrelated random calls need not disturb it. In the lab's independent draw experiment, observed frequencies approach the specified probabilities. During text generation, however, the distribution changes with the context, so whole text frequencies need not match the initial row.

## Slide: Real checkpoint, real limitations
- Same trained weights and prompt in every run
- Greedy sample repeats “season”
- Warm sample contains malformed-looking words
- Combined nucleus run changes several controls
Narration:
Now the lab loads the real saved character GPT. The greedy continuation repeats the word season. The warmer sample contains malformed looking words, while the cooler and filtered samples have different phrasing. These are real program outputs, not text we wrote to demonstrate a point. They are also only individual samples. Do not conclude that one policy always wins. The filtered run changes multiple settings, so it does not isolate the effect of nucleus filtering. A fair comparison needs controlled settings, multiple prompts, and a task-relevant definition of success.

## Slide: Stop and crop are separate
```text
prompt length:      130
visible lengths:    [128, 128]
returned length:    132
```
- Token budget bounds the loop
- EOS fixture tests stopping, not language skill
Narration:
The stored output and the visible model context are different things. Our long prompt has one hundred thirty tokens, but the GPT sees only the most recent one hundred twenty eight. After two new tokens, the returned sequence has one hundred thirty two. This does not give the model unlimited memory. The actual Shakespeare checkpoint has no special end of sequence token, so its runs stop at the requested budget. A separate labelled fixture emits a designated stopping token to test that branch. It is not evidence of a learned conversation ending skill.

## Slide: Reproducibility needs more than a seed
- eval mode controls dropout behavior
- Inference mode avoids autograd recording
- Record checkpoint, tokenizer, prompt, policy and environment
- Frozen parameters checked explicitly
Narration:
Evaluation mode and inference mode solve different problems. Evaluation mode disables training behavior such as dropout. Inference mode avoids recording work for gradient computation; it does not automatically switch the model into evaluation mode. Our helper requires both conditions and checks that generation leaves parameters unchanged. Repeating a seeded call gives identical IDs here, but PyTorch does not promise exact equality across platforms or releases. Record the full experiment, not just the seed. Deterministic replay is valuable for debugging, but it still says nothing by itself about factual correctness.

## Slide: Summary and explain-back
- Model: scores possible next tokens
- Decoder: reshape, restrict, choose
- Loop: append and ask again
- Validation and permissions remain separate
Narration:
The complete picture is now small enough to explain without jargon. The model supplies scores. The decoder reshapes or restricts the distribution and chooses a token. The loop appends that token and asks again, until a stopping rule fires. The weights stay fixed while the context changes. Before editing the lab, predict what happens when top k is one at both temperatures. Then explain why changing temperature can change the output without teaching the model anything new. Accurate explanation plus correct application is the learning goal, not merely reaching the lab's final success message.
