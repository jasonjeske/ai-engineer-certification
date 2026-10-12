## Slide: Tokens, teaching a computer to read
- LLM Fundamentals, chapter 1
- Build a GPT-4 style tokenizer and measure it against the real ones
Narration:
Welcome to chapter one of LLM Fundamentals. In this walkthrough we are going to answer a question that sounds simple: what does a language model actually read? The answer is not words and not letters. It is a list of integers called tokens. By the end of this video you will know exactly where those integers come from, you will have seen a tokenizer learn its vocabulary from Shakespeare, and you will understand why tokens explain some of the strangest things language models do.

## Slide: The model only sees integers
- Text goes in, a list of IDs comes out
- The model predicts the next ID, not the next letter
- Prices, context windows and limits are all counted in tokens
Narration:
Here is the core idea. A tokenizer is a translator between text and numbers. It has a fixed vocabulary, which is just a big table where every entry is a piece of text with an ID. To encode, you cut the text into pieces from that table and write down their IDs. To decode, you look the IDs up and glue the pieces back together. The model in the middle never sees anything but the IDs. That matters for you in a very practical way, because API prices are per token, context windows are measured in tokens, and when an agent forgets its instructions, the reason is almost always that it ran out of tokens.

## Slide: Option one, one token per character
```text
"To be, or not" -> [32, 53, 1, 40, 43, 6, 1, 53, 56, 1, 52, 53, 58]
vocabulary size 65
"Dzień" fails: 'ń' is not in the vocabulary
```
Narration:
The simplest tokenizer gives every character its own ID. Shakespeare only uses sixty five different characters, so the vocabulary is tiny. "To be, or not" becomes thirteen tokens. That is easy to understand, but it has two problems. Sequences get long, which wastes the model's limited context window. And it breaks on anything it has not seen. Try to encode the Polish word "dzień" and it fails, because the letter n with an accent never appears in Shakespeare.

## Slide: Option two, whole words
- "To be, or not" is about 5 tokens, nice and short
- Typos, new names, other languages and code: no ID exists
- run, runs, running, runner all need separate entries
Narration:
The opposite extreme is one token per word. Now sequences are short, which is great, until someone types a word that is not in the table. A typo, a new product name, a variable in some code, a word in another language. The model simply has no ID for it. Word vocabularies also explode in size, because every form of every word needs its own entry. So neither extreme works. What we want is something in between, where common things are single tokens and rare things still work.

## Slide: Start from bytes
```text
'Hi ń'
Unicode points  [72, 105, 32, 324]
UTF-8 bytes     [72, 105, 32, 197, 132]
```
- 256 possible bytes, so nothing is ever unknown
Narration:
The trick that solves the unknown word problem is to start from bytes. Computers store text as UTF-8 bytes. Plain English letters take one byte. The Polish n with an accent takes two, here the pair one ninety seven, one thirty two. Emoji take four. There are exactly two hundred fifty six possible bytes, so if your starting vocabulary is those two hundred fifty six values, any text in any language can always be encoded. The worst case is just one token per byte. This is called byte level encoding, and GPT-2 introduced it for OpenAI's models.

## Slide: Byte-pair encoding in five steps
1. Split text into chunks with a regex
2. Turn each chunk into bytes, IDs 0 to 255
3. Count every neighboring pair
4. Merge the most frequent pair into a new ID
5. Repeat until the vocabulary is big enough
Narration:
Now we grow the vocabulary. This is byte pair encoding. First, chop the training text into word like chunks with a regular expression. Then turn every chunk into bytes. Count every pair of neighbors across all the text. Take the most frequent pair, give it the next free ID, starting at two fifty six, and replace every occurrence of that pair. Then count again and repeat. Each merge makes the text a little shorter. After a few thousand merges, common words become single tokens. The list of merges, in the order they were learned, is the entire tokenizer.

## Slide: Watching it learn from Shakespeare
```text
merge 1:  ' ' + 't'   -> ' t'    seen 4,336x
merge 2:  'h' + 'e'   -> 'he'    seen 3,484x
merge 3:  'o' + 'u'   -> 'ou'    seen 2,529x
merge 9:  ' t' + 'he' -> ' the'  seen 1,783x
... 768 merges, vocabulary 1024
```
Narration:
Here is real output from the lab, training on the first two hundred thousand characters of Shakespeare. The very first merge is a space followed by the letter t, which appears four thousand three hundred thirty six times. The second is h plus e. Then o plus u. And look at merge nine: it glues two earlier tokens together, space t plus h e, to make space t h e. That is how BPE builds bigger tokens out of smaller ones. Notice the leading space. In these tokenizers, a word usually carries the space in front of it.

## Slide: Why split with a regex first?
- Without it, BPE learns junk like "dog." and "e, t"
- Merges stay inside a word, a number, punctuation or whitespace
- We use GPT-4's exact pattern, tested against tiktoken
Narration:
Why the regular expression? Without it, BPE would happily merge across boundaries. If "dog" is often followed by a period, it might learn a token for "dog period", and then the same word gets different tokens depending on what comes after it. Real tokenizers prevent that by splitting text into chunks first, words with their leading space, short runs of digits, punctuation, and whitespace, and they only merge inside a chunk. Our library uses the exact pattern from GPT-4's tokenizer, and its tests check that it splits text identically to OpenAI's tiktoken.

## Slide: Encoding new text
```python
while len(ids) >= 2:
    counts = _pair_counts(ids)
    pair = min(counts, key=lambda p: merges.get(p, inf))
    if pair not in merges:
        break
    ids = _merge(ids, pair, merges[pair])
```
Narration:
To encode new text, we replay the merges. Start from the bytes of each chunk. Look at every neighboring pair, and among the pairs we actually learned, apply the one that was learned earliest. Repeat until no learned pair is left. Using the training order matters, because it reproduces exactly the decisions training made. OpenAI's tiktoken calls that order the rank. Decoding is the easy direction: every token maps to a fixed string of bytes, so you concatenate them and read them as UTF-8.

## Slide: The result
```text
['To', ' be', ',', ' or', ' not', ' to', ' be', ',',
 ' that', ' is', ' the', ' qu', 'est', 'ion', '.']
```
- 15 tokens; common words whole, "question" in 3 pieces
Narration:
And here is the result. "To be, or not to be, that is the question" becomes fifteen tokens. Common words like be, not, that and the are single tokens. The less common word question is assembled from three pieces, q u, e s t, and i o n. And because we started from bytes, nothing can fail. The Polish greeting "dzień dobry" encodes fine, it is just split into smaller pieces, including the two raw bytes of that accented n.

## Slide: More vocabulary, more compression
```text
vocab   256   1.00 bytes per token
vocab   512   1.92
vocab  1024   2.31
GPT-2   50K   3.07
GPT-4  100K   3.49
GPT-4o 200K   3.57
```
Narration:
How do you measure a tokenizer? By compression: how many bytes of unseen text each token covers. Raw bytes are exactly one byte per token. Our tokenizer with five hundred twelve entries reaches one point nine two, and with one thousand twenty four, two point three one. OpenAI's real tokenizers do better. GPT-2 reaches three point zero seven on the same text, GPT-4 three point four nine, GPT-4o three point five seven. Look at the shape of those numbers. Doubling from two fifty six to five twelve almost doubled compression. Doubling from a hundred thousand to two hundred thousand gained about two percent on this English text. The returns shrink fast.

## Slide: Why not a million tokens?
- Every token needs an embedding row and an output score
- GPT-2 small: 50,257 x 768 = about 38.6M of 124M parameters
- Rare tokens barely get trained
Narration:
So why not use a vocabulary of a million tokens? Because vocabulary is not free inside the model. Every token needs its own learned vector at the input, called an embedding, and the output layer has to score every token at every step. In GPT-2 small, the embedding table alone is fifty thousand rows of seven hundred sixty eight numbers, about thirty eight point six million parameters, close to a third of the model. Rare tokens are a second problem. If a token almost never shows up in training data, its embedding barely learns anything. That is the root of the famous glitch tokens like SolidGoldMagikarp.

## Slide: Surprise one, spelling
```text
' strawberry'  -> 1 token   [' strawberry']
'strawberry'   -> 3 tokens  ['st', 'raw', 'berry']
```
Narration:
Now the payoff. Let's run real strings through OpenAI's tokenizers. In all three, the word strawberry with a leading space is a single token. Without the space, it splits into three tokens: the letters s and t, then the letters r, a and w, then the word berry. So when you ask a model how many r's are in strawberry, it is looking at one or three opaque IDs, not ten letters. Whatever it knows about the letters inside a token, it had to learn indirectly from data. That is why spelling and letter counting are harder than they look, and why asking the model to spell the word out first often helps.

## Slide: Surprise two, languages cost different amounts
```text
"Good morning, how are you?"   7 tokens in all three
"Dzień dobry, jak się masz?"   GPT-2 16, GPT-4 11, GPT-4o 10
```
Narration:
Same meaning, different price. The English greeting is seven tokens in every tokenizer. The Polish version is sixteen tokens in GPT-2, eleven in GPT-4 and ten in GPT-4o. In GPT-2, the accented letters are not even whole tokens, each one is two raw bytes. Since you pay per token and the context window is counted in tokens, the Polish request costs more and uses more context. Newer vocabularies spend many of their extra entries on other languages, and the gap shrinks, but it does not close.

## Slide: Surprises three and four, numbers and code
```text
'127 + 4 = 131'   GPT-2  ['127', ' +', ' 4', ' =', ' 131']
                  GPT-4  ['127', ' +', ' ', '4', ' =', ' ', '131']
8 spaces + 'return y'   GPT-2 9 tokens   GPT-4 3 tokens
```
Narration:
Two more. Numbers: GPT-4 era tokenizers split digits into runs of at most three and keep the space separate, so a number gets the same tokens whether or not a space came before it. Even so, the model does arithmetic on chunks like one three one, not on digits, which is part of why long arithmetic without tools is unreliable. And code: eight spaces of Python indentation are nine tokens in GPT-2, one per space, but only three in GPT-4, which has tokens for runs of spaces. That made code denser, so the same context window holds more of it.

## Slide: Mistakes to avoid
- A token is not a word: measure, don't guess
- The model reads IDs, not letters
- The tokenizer is trained separately, then frozen
- Token counts differ between models
Narration:
Let's lock in the misconceptions. A token is not a word. Sometimes it is, often it is not, so when it matters, count instead of guessing. The model reads IDs, not letters. The tokenizer is trained separately from the model, before it, and then frozen, and every embedding is tied to a specific token ID, so you can't just swap tokenizers. And token counts differ between models, so budget with each model's own counter. For Claude, Anthropic provides a token counting endpoint in the API rather than a downloadable tokenizer.

## Slide: Your turn
- Run lab.py and read the output top to bottom
- Predict first: does vocab 2048 gain more or less than 512 to 1024?
- Watch Karpathy, "Let's build the GPT Tokenizer", from 22:30
- Explain it back: why does a model miscount r's in strawberry?
Narration:
Here is what to do next. Run the lab and read its output from top to bottom. Then try the experiments at the bottom of the file, and predict the answer before you run each one. For example, will growing the vocabulary from one thousand twenty four to two thousand forty eight gain more or less than the step from five twelve to one thousand twenty four? Then watch Karpathy's tokenizer lecture starting around twenty two minutes, where he builds this same algorithm live. Finally, explain it back to Your study partner in your own words: why can a model know what a strawberry is but still miscount its r's? If you can explain that clearly, you understand tokens.
