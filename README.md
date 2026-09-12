https://github.com/user-attachments/assets/791396cc-3b07-4149-8c98-10a9ef632974

<h1 align="center">AI Engineer Certification</h1>

<p align="center">
  <b>Zero to hireable. Completely free.</b><br>
  15 courses &nbsp;·&nbsp; 126 chapters &nbsp;·&nbsp; 125 runnable labs &nbsp;·&nbsp; build a real LLM from scratch, then a real agent
</p>

<p align="center">
  <img src="https://img.shields.io/badge/price-free-1b6fb8?style=flat-square" alt="Free">
  <img src="https://img.shields.io/badge/license-MIT-1b6fb8?style=flat-square" alt="MIT">
  <img src="https://img.shields.io/badge/courses-15-2e9bf5?style=flat-square" alt="15 courses">
  <img src="https://img.shields.io/badge/labs-125-2e9bf5?style=flat-square" alt="125 labs">
  <img src="https://img.shields.io/badge/setup-none-1b6fb8?style=flat-square" alt="No setup">
</p>

<p align="center">
  <a href="https://krypteiasec.com/academy"><b>Start the certification →</b></a>
</p>

---

Everyone says AI will take your job. This certification is built on the opposite bet: that the people who can *build* with AI become unstoppable. It takes you from Python and the math you need, through building a language model by hand, to designing and shipping real autonomous agents, with a runnable lab for every single chapter. No paywall, no login, no catch.

The full interactive certification, prose teaching, quizzes, in-browser labs, and a free certificate, lives at **[krypteiasec.com/academy](https://krypteiasec.com/academy)**. This repository is the code layer: everything you clone, read, and run yourself.

## What's inside

**15 courses · 126 chapters · a runnable lab and a Jupyter notebook for every chapter that has one · two really-trained model checkpoints.** Free, no login, no paywall. Run 91 of the 125 labs right in your browser at **[krypteiasec.com/academy](https://krypteiasec.com/academy)**.

The path has five tracks. The first two are what a model is made of and how you work with one; the middle is the certification's spine, agentic engineering, taught vendor-agnostic and then applied with Claude; the last two go deep and get you hired.

### What models are made of (Courses 0–1)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 0 | **Foundations: Python & Math for AI** | Python from zero plus the math you actually need: vectors, matrices, probability, softmax, gradients | 8 |
| 1 | **Build a Tiny LLM From Scratch** | A real language model by hand: tokenizer, embeddings, attention, transformer, training loop, generation | 10 |

### Working with models (Courses 2–5)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 2 | **Prompt Engineering** | Few-shot, chain-of-thought, structured output, prompt caching, eval-driven optimization | 8 |
| 3 | **LLM Application Engineering** | APIs and SDKs, streaming, structured output, retries, the daily job | 8 |
| 4 | **RAG & Embeddings** | Embeddings, chunking, vector search, re-ranking, end-to-end RAG, and the failure modes that bite | 8 |
| 5 | **Evaluation & Testing** | Evals as the new unit tests: graders, datasets, regression | 8 |

### Agentic engineering (Courses 6–9) — the spine

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 6 | **Agentic Engineering: How Autonomous Agents Are Built** | The agent loop by hand, tool contracts, environments, planning, memory, permissions and human-in-the-loop, protocols beyond MCP, multi-agent systems, evaluating agents, observability and cost, running agents in production — vendor-agnostic, plus a TypeScript appendix proving the loop isn't a Python idea | 13 |
| 7 | **Building Agents with Claude: API, Agent SDK and Claude Code** | The applied, version-stamped unit: the Messages API, the Claude Agent SDK loop, driving Claude Code, subagents/hooks/MCP, skills and context files, orchestrating a fleet | 8 |
| 8 | **AI Security: Models and Agents** | OWASP LLM Top 10, prompt-injection offense and defense, guardrails, then the agent attack surface (the lethal trifecta, tool-result injection, MCP tool poisoning, confused deputy) and red-teaming your own agent | 10 |
| 9 | **AI Engineering in Production** | Serving, quantization (AWQ/GGUF), vLLM and KV-cache, LLMOps | 8 |

### Depth (Courses 10–12)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 10 | **Training & Fine-tuning** | Datasets, loss curves, SFT and LoRA/QLoRA, DPO | 8 |
| 11 | **Transformers Deep Dive** | Positional encodings (RoPE), attention variants, the internals cold | 8 |
| 12 | **Multimodal AI** | Vision-language, image generation, speech (STT/TTS), multimodal RAG | 6 |

### Getting hired (Courses 13–14)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 13 | **Interview Prep & System Design** | The real interview loop, from-scratch coding, LLM system design, and the agentic system-design round | 9 |
| 14 | **Capstone Projects** | Deployed, evaluated builds that actually get you hired, including one agent with its own eval harness, permission model, and cost report | 6 |

Full chapter-by-chapter detail in **[`COURSES.md`](COURSES.md)**.

<details>
<summary><b>Repository layout</b></summary>

```
labs/
  academy_llm.py          a tiny deterministic offline stand-in "LLM" the teaching labs import
  _models/                two REAL trained checkpoints + the training code
  foundations/            Course 0 labs
  <course>/               one folder of labs per course (prompt-engineering, rag, agentic-engineering, ...)
notebooks/                125 Jupyter notebooks, one per lab
courses.export.json       the machine-readable catalog the site consumes
COURSES.md                the human catalog: every course, chapter, and its lab
requirements.txt          torch, numpy, jupyter (only for the PyTorch + notebook labs)
```

</details>

## Run it

Two ways to work through any lab, both first-class.

**Terminal.** Every lab is a standalone script that prints an invariant proving the concept and exits 0.

```bash
python3 labs/prompt-engineering/pe2-few-shot.py
python3 labs/agentic-engineering/ae2-loop-by-hand.py
```

Most labs are pure standard library and need no install. The PyTorch labs and the trained models need `pip install -r requirements.txt`.

**Jupyter.** Each lab has a matching notebook, split into cells with the explanation above each step.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Or run **91 of the 125 labs with zero setup, right in your browser** at [krypteiasec.com/academy](https://krypteiasec.com/academy) (Python compiled to WebAssembly).

## The trained models (`labs/_models/`)

Two genuinely trained checkpoints you can load and generate from, not mocks:

- `tinygpt.pt`: a from-scratch character-level GPT (the same tiny architecture you build by hand in Course 1).
- `lora_adapter.pt`: a real LoRA fine-tune of that base model, with the base weights frozen.

```bash
python3 labs/_models/verify.py          # loads both, generates, prints MODELS OK
python3 labs/_models/train_tinygpt.py   # retrain the base from scratch (~15s on Apple Silicon)
```

## License

MIT, see [`LICENSE`](LICENSE). The curriculum and code are free to read, run, fork, and teach from.

<p align="center"><sub>Built by <a href="https://krypteiasec.com">Krypteia Sec</a> · an independent educational resource on AI and agentic engineering.</sub></p>
