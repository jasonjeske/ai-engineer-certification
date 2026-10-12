# LLM Fundamentals: Build a Tiny LLM From Scratch

The rebuilt KrypteiaSec Academy: ten complete chapters, tested Python labs, worked calculations, quizzes, Markdown and PDFs. Read and watch at https://krypteiasec.com/academy.

The former multi-course catalog and certification are retired. The Git tag `academy-before-redo-20261012` preserves the prior version; it is not the active course.

## Setup

Use Python 3.12 in a virtual environment. From this repository's root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python chapters/llm-fundamentals/01-tokens/lab.py
.venv/bin/python -m pytest labs/tests
```

Installing dependencies needs internet access. Afterwards the examples use the supplied Shakespeare data, tokenizer caches and trained checkpoint offline. The small teaching models run on CPU; CUDA or Apple MPS is optional. Each chapter ends with assertions and `CHECK PASSED`. A notebook beside each lab runs that exact lab, rather than a different example.

## Chapters

| Chapter | Subject | Files |
|---|---|---|
| 1 | Tokens: teaching a computer to read | [Read and run](chapters/llm-fundamentals/01-tokens) |
| 2 | Embeddings: giving each token a vector of meaning | [Read and run](chapters/llm-fundamentals/02-embeddings) |
| 3 | The bigram model: your first real predictor | [Read and run](chapters/llm-fundamentals/03-the-bigram-model) |
| 4 | Self attention: the idea that changed everything | [Read and run](chapters/llm-fundamentals/04-self-attention) |
| 5 | Multi head attention: many perspectives at once | [Read and run](chapters/llm-fundamentals/05-multi-head-attention) |
| 6 | The transformer block: residuals, layer norm, and an MLP | [Read and run](chapters/llm-fundamentals/06-the-transformer-block) |
| 7 | Assemble the GPT: stacking blocks into a model | [Read and run](chapters/llm-fundamentals/07-assemble-the-gpt) |
| 8 | The training loop: how a model actually learns | [Read and run](chapters/llm-fundamentals/08-the-training-loop) |
| 9 | Generation: sampling text from your model | [Read and run](chapters/llm-fundamentals/09-generation) |
| 10 | Scaling, and what comes next | [Read and run](chapters/llm-fundamentals/10-scaling-and-what-comes-next) |

Each chapter directory includes `chapter.md`, `lab.py`, `lab.ipynb`, `quiz.json`, `lectures.json`, `narration.md` and `pdf-build/chapter.pdf`. Videos use James Agent narration and the same Signal theme as the website, and are hosted on the chapter pages. The PDFs use that palette and typography too.

## What the checks establish

The code tests tensor shapes, causal masking, gradients, training and sampling with fixed seeds. Some examples deliberately fit synthetic fixtures; their explanations distinguish those fixtures from real language-model evaluation. Passing a quiz or running a script is not proof of professional mastery.

## Attribution

The implementations build on Andrej Karpathy's MIT-licensed micrograd, minbpe and nanoGPT teaching projects. Source and lecture links are listed in every chapter. See `THIRD-PARTY-NOTICES.md` for their license notices. Shakespeare is public-domain text from the Tiny Shakespeare teaching dataset. The included trained model is a tiny educational character GPT, not a production assistant.
