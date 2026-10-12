## Slide: Scaling, and what comes next
- LLM Fundamentals, chapter 10 of 10
- Count the machine before making it larger
- Size, memory, compute and quality are different questions
Narration:
You have reached the final chapter of LLM Fundamentals. The model you assembled already has the central pieces of a larger language model. But making its matrices larger is not the same thing as making a useful assistant better. Today we will count the learned numbers, examine what each size change costs, and separate exact measurements from estimates. The lab runs a real tiny model. It does not train a large model or produce an empirical scaling law. That boundary is part of the lesson, not a footnote.

## Slide: Start with learned numbers
- Parameter: a learned scalar
- Matrix count: rows times columns
- Temporary activations are not parameters
Narration:
A parameter is a learned number. An embedding table contains one row per token, with one column per feature. Multiplying rows by columns tells us how many learned values it contains. The same reasoning applies to the projection matrices inside attention and the feedforward network. Now separate those learned values from the working values created for a request. Attention scores and intermediate activations occupy memory, but they are not extra parameters. Training needs additional arrays too. We will keep those categories separate instead of calling every byte part of model size.

## Slide: The block's main matrices
```text
Attention: 4*C^2
MLP:       8*C^2
Together: 12*C^2 per block
```
Narration:
Look inside one transformer block. Queries, keys and values each use a projection from the embedding width back to that width. The output projection adds another matrix of the same size. That gives attention four times the width squared. The feedforward network expands to four times the width and then contracts again. Each of its two matrices contributes four times the width squared. Together, attention and the feedforward network contribute twelve times the width squared. The coefficient comes from these actual shapes. Different architectures can require a different formula.

## Slide: Do not omit the small terms
```text
P = V*C + S*C + L*(12*C^2 + 13*C) + 2*C
V: vocabulary    C: width
S: positions     L: blocks
```
Narration:
The main matrices are not the whole model. Linear biases and the learned scales and offsets in layer normalization add more values. Our exact block adds thirteen times the width beyond its matrix weights. The final normalization adds twice the width. Outside the blocks, count the token table and the learned position table. This expression describes our particular implementation, with affine layer norms, learned positions and a tied output head. It is not a generic transformer calculator. The lab checks exact equality against the tensors PyTorch actually constructs, rather than accepting a loose tolerance.

## Slide: The toy count is real
```text
Token table       72
Position table    48
All blocks     2,616
Final norm        16
Total          2,752
```
Narration:
Here is the smallest concrete example. The toy has nine tokens, width eight, six learned positions and three blocks. Its token table contains seventy two numbers, its position table forty eight, its blocks two thousand six hundred sixteen, and its final normalization sixteen. The exact total is two thousand seven hundred fifty two. The leading matrix estimate misses three hundred twenty eight learned values. The lab also runs a forward pass and a backward pass, checking finite gradients on every parameter. That proves the pieces are connected, not that the random model has learned useful language.

## Slide: One table, two uses
- Token embedding and output head share one Parameter
- Tied toy: 2,752
- Untied toy: 2,824
- Vocabulary scoring still happens
Narration:
The input embedding and output head share the same parameter object. Count that object once even though two parts of the model use it. To untie it, the lab clones the head into a new parameter. The count then rises to two thousand eight hundred twenty four, adding one independent vocabulary matrix. Sharing is about object identity, not just starting with equal numbers. Weight tying saves stored learned values. It does not eliminate the work of producing vocabulary scores at the end of the forward pass.

## Slide: Turn one knob at a time
```text
Base             2,752
Double layers    5,368
Double width    10,112
Double heads     2,752
```
Narration:
Now change one configuration field at a time. Doubling layers nearly doubles the total, because the tables and final normalization stay fixed. Doubling width grows the large block matrices much faster, because width appears in both matrix dimensions. The total does not quite quadruple because other terms grow only linearly. Doubling attention heads at fixed width leaves the parameter count unchanged. Those heads divide the same total projection width into smaller pieces. If you instead kept head width fixed, adding heads would increase total width. Always say what you held constant.

## Slide: GPT-2 shape without a download
- Exact count: 124,439,808
- Meta tensors store shape metadata, not weight values
- A counting check is not a trained model
Narration:
We can inspect a much larger configuration without allocating all its weights. The lab constructs the shared model with the documented small GPT two dimensions on PyTorch's meta device. Meta tensors record shapes and types but contain no numeric values. The exact count is one hundred twenty four million four hundred thirty nine thousand eight hundred eight. The token table alone is thirty one point zero two percent of that total. This verifies the shape accounting. It does not download a checkpoint, reproduce trained behavior or prove that our implementation is numerically identical to the original model.

## Slide: Weights are only part of memory
- Weight bytes = parameters times bytes per value
- FP32 Adam arrays: weights, gradients, two moments
- Activations, buffers and temporary storage are extra
Narration:
To estimate weight storage, multiply the parameter count by the bytes per value. But training has more arrays than weights. In our explicitly chosen full precision layout, gradients and both Adam moment arrays have the same number of entries as the weights. All four arrays use four bytes per entry. The test checks that accounting after a real optimizer step on the toy. It still excludes activations, buffers and temporary allocations. Mixed precision or sharded optimizers can use other layouts. Treat this as a named estimate, not a promise that a training run fits in a particular memory budget.

## Slide: A longer request, identical weights
```text
Dense scores: B*H*T*T
Full multi-head KV cache: 2*L*B*T*C values
Double T: scores x4, cache x2
```
Narration:
A longer request can need more memory without changing the model's learned weights. Our explicit attention implementation constructs a square score matrix for every head and batch item. Doubling actual context therefore quadruples that score storage. A key value cache is different: it retains keys and values for previous positions, so its size grows linearly with context in this full multi-head design. Doubling context doubles that hypothetical cache. Our generation function does not implement caching; it recomputes the context. The printed cache estimate describes an alternative implementation, not a hidden capability of this lab.

## Slide: Better kernels change storage
- FlashAttention: tiled exact attention
- Avoid full score-matrix materialization in GPU memory
- Mathematical attention and implementation storage differ
Narration:
The square score-storage estimate describes our educational code. It is not a universal requirement for every backend. FlashAttention uses tiled computation to avoid storing the full attention matrix in high bandwidth memory. That is an implementation improvement, not a claim that attention stopped comparing token pairs. Similarly, grouped query attention changes the number of key and value heads and therefore changes cache accounting. Before using a memory formula, inspect both the architecture and the execution method. A correct formula applied to the wrong implementation can still give a misleading result.

## Slide: Tokenization changes the tradeoff
```text
Same sentence:
Character tokens: 42
GPT-2 BPE tokens: 13
Smaller sequence, larger vocabulary table
```
Narration:
The same sentence in the lab becomes forty two character tokens or thirteen GPT two byte pair tokens. Shorter sequences can reduce attention work for a fixed amount of text. But the byte pair vocabulary needs a much larger embedding table and more vocabulary scores. This is a tradeoff between sequence length and vocabulary size, not free compression. The lab uses a local tokenizer cache and verifies exact decoding in both cases. It does not replace the tokenizer inside a trained model, because that model's embedding rows already have specific token meanings.

## Slide: Compute couples model and data
```text
Rough dense-training FLOPs: 6*N*D
N: model-size count
D: processed training tokens, including repeats
Fixed budget: bigger N means smaller D
```
Narration:
For a rough dense training budget, multiply model size by processed tokens and by six. This approximation captures dominant matrix work, not every operation. Long context attention, embedding details and implementation choices can matter. The lab shows that doubling model size while halving processed tokens preserves the rough budget. Doubling both instead multiplies it by four. Data and parameters cannot both grow for free. Also, processed tokens include repeats. A larger token count is not automatically the same thing as more independent useful information. Quality and validation remain necessary.

## Slide: Illustration is not evidence
```text
Chosen function: 1.5 + 1/sqrt(n) + 1/sqrt(d)
Constraint: n*d = 16
Best tested allocation: n=4, d=4
NOT a fitted or measured scaling law
```
Narration:
The final table evaluates a deliberately chosen teaching function. Its model and data units are dimensionless. They are not real parameter and token counts. With a fixed product, this symmetric function favors a balanced allocation. The floor, coefficients and exponents were chosen by us. No training losses were measured to produce this table. Real scaling papers fit controlled experiments and test their predictions. Kaplan and Chinchilla are evidence about the regimes they studied; this table is only a way to understand their allocation question. Changing our chosen coefficients can change the preferred balance.

## Slide: What comes next depends on the failure
- Missing evidence: inspect retrieval and context
- Behavior mismatch: study post-training
- Unknown quality: define and run evaluations
- Tool actions: permissions and software checks
Narration:
A bigger model is one possible intervention, not a diagnosis. If a document assistant never receives the relevant passage, inspect retrieval and context assembly before assuming more weights will solve the problem. If behavior is the issue, study post-training. If success is unclear, define evaluation cases. If the system takes actions, enforce permissions and validate them in software. Modern architectures also change normalization, positions, key value sharing and expert routing. The small model gives you a foundation for understanding those changes, not permission to assume every production system is just this file with larger constants.

## Slide: Summary and explain-back
- Count exact tensors and state assumptions
- Separate size, runtime memory and quality
- Distinguish measured evidence from constructed examples
- Explain first, then apply to a new case
Narration:
The useful habit is to ask what changed and what stayed fixed. Width, depth, head count and request length have different effects. A parameter count is not a memory profile, and a lower language modeling loss is not a complete assistant evaluation. Run the lab, then explain why doubling width, doubling heads at fixed width, and doubling request length are different interventions. Apply that explanation to a new case before calling the idea mastered. Completing the course materials is not the same as demonstrating understanding. Your explanation and application are the evidence that matters.
