https://github.com/krypteiasec/ai-engineer-certification/raw/main/assets/intro.mp4

<h1 align="center">AI Engineer Certification</h1>

<p align="center">
  <b>Zero to hireable. Completely free.</b><br>
  19 courses &nbsp;·&nbsp; 151 chapters &nbsp;·&nbsp; 151 runnable labs &nbsp;·&nbsp; build a real LLM from scratch
</p>

<p align="center">
  <img src="https://img.shields.io/badge/price-free-7a1f1f?style=flat-square" alt="Free">
  <img src="https://img.shields.io/badge/license-MIT-5c1717?style=flat-square" alt="MIT">
  <img src="https://img.shields.io/badge/courses-19-b53d3d?style=flat-square" alt="19 courses">
  <img src="https://img.shields.io/badge/labs-151-b53d3d?style=flat-square" alt="151 labs">
  <img src="https://img.shields.io/badge/setup-none-5c1717?style=flat-square" alt="No setup">
</p>

<p align="center">
  <a href="https://krypteiasec.com/academy"><b>Start the certification →</b></a>
</p>

---

Everyone says AI will take your job. This certification is built on the opposite bet: that the people who can *build* with AI become unstoppable. It takes you from Python and the math you need all the way to shipping AI agents and building a language model by hand, with a runnable lab for every single chapter. No paywall, no login, no catch.

The full interactive certification, prose teaching, narrated videos, quizzes, in-browser labs, and a free certificate, lives at **[krypteiasec.com/academy](https://krypteiasec.com/academy)**. This repository is the code layer: everything you clone, read, and run yourself.

## What's inside

**19 courses · 151 chapters · a runnable lab, a Jupyter notebook, and a narrated video for every single chapter · two really-trained model checkpoints.** Free, no login, no paywall. Run 117 of the 151 labs right in your browser at **[krypteiasec.com/academy](https://krypteiasec.com/academy)**.

### Core track · become the engineer (Courses 0 to 13)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 0 | **Foundations: Python & Math for AI** | Python from zero plus the math you actually need: vectors, matrices, probability, softmax, gradients | 8 |
| 1 | **Build a Tiny LLM From Scratch** | A real language model by hand: tokenizer, embeddings, attention, transformer, training loop, generation | 10 |
| 2 | **Prompt Engineering** | Few-shot, chain-of-thought, structured output, prompt caching, eval-driven optimization | 8 |
| 3 | **RAG & Embeddings** | Embeddings, chunking, vector search, re-ranking, end-to-end RAG, and the failure modes that bite | 8 |
| 4 | **Agents, Tools & MCP** | The ReAct loop, tool and function calling, agent memory, and MCP | 8 |
| 5 | **Evaluation & Testing** | Evals as the new unit tests: graders, datasets, regression | 8 |
| 6 | **LLM Application Engineering** | APIs and SDKs, streaming, structured output, retries, the daily job | 8 |
| 7 | **Training & Fine-tuning** | Datasets, loss curves, SFT and LoRA/QLoRA, DPO | 8 |
| 8 | **Transformers Deep Dive** | Positional encodings (RoPE), attention variants, the internals cold | 8 |
| 9 | **AI Engineering in Production** | Serving, quantization (AWQ/GGUF), vLLM and KV-cache, LLMOps | 8 |
| 10 | **AI Safety & Security** | OWASP LLM Top 10, prompt-injection offense and defense, guardrails | 8 |
| 11 | **Multimodal AI** | Vision-language, image generation, speech (STT/TTS), multimodal RAG | 6 |
| 12 | **Interview Prep & System Design** | The real interview loop, from-scratch coding, LLM system design | 8 |
| 13 | **Capstone Projects** | 3 to 5 deployed, evaluated builds that actually get you hired | 7 |

### Applied track · Claude, agents & your own AI OS (Courses 14 to 18)

| # | Course | What you build | Ch. |
|:-:|--------|----------------|:-:|
| 14 | **Claude Code & Agentic Builds** | Build agents the way Claude Code does: the agentic loop and tool contract | 8 |
| 15 | **Claude Code SDK & API** | Drive Claude from code: the Messages API and Agent SDK end to end | 8 |
| 16 | **CCA-F Certification Prep** | The weighted Claude exam domains, scenarios, and practice | 8 |
| 17 | **LifeOS: Setup to Advanced** | A personal AI operating system end to end: current state to ideal state | 8 |
| 18 | **LifeOS Mastery: The Complete Setup** | The flagship LifeOS course, taught the way its creator teaches it | 8 |

Full chapter-by-chapter detail in **[`COURSES.md`](COURSES.md)**.

<details>
<summary><b>Repository layout</b></summary>

```
labs/
  academy_llm.py        a tiny deterministic offline stand-in "LLM" the teaching labs import
  _models/              two REAL trained checkpoints + the training code
  foundations/          Course 0 labs
  <course>/             one folder of labs per course (prompt-engineering, rag, ...)
notebooks/              151 Jupyter notebooks, one per lab
courses.export.json     the machine-readable catalog the site consumes
COURSES.md              the human catalog: every course, chapter, and its lab
requirements.txt        torch, numpy, jupyter (only for the PyTorch + notebook labs)
```

</details>

## Run it

Two ways to work through any lab, both first-class.

**Terminal.** Every lab is a standalone script that prints an invariant proving the concept and exits 0.

```bash
python3 labs/prompt-engineering/pe2-few-shot.py
python3 labs/rag/rag6-rag-pipeline.py
```

Most labs are pure standard library and need no install. The PyTorch labs and the trained models need `pip install -r requirements.txt`.

**Jupyter.** Each lab has a matching notebook, split into cells with the explanation above each step.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Or run **117 of the 151 labs with zero setup, right in your browser** at [krypteiasec.com/academy](https://krypteiasec.com/academy) (Python compiled to WebAssembly).

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
