# Course Catalog

> 15 courses, 126 chapters, 125 runnable labs, 125 notebooks.

## 0. Foundations: Python and Math for AI

From zero. Real Python you run yourself and only the math you need (vectors, matrices, probability, softmax, gradients). 8 chapters, live.

- **Ch 1: Your first Python: variables and types**  lab `f1-python-basics.py`, notebook `f1-python-basics.ipynb`
- **Ch 2: Lists and loops: many values at once**  lab `f2-lists-loops.py`, notebook `f2-lists-loops.ipynb`
- **Ch 3: Functions and dictionaries: reusable logic and lookups**  lab `f3-functions-dicts.py`, notebook `f3-functions-dicts.ipynb`
- **Ch 4: Vectors: the math object at the center of AI**  lab `f4-vectors.py`, notebook `f4-vectors.ipynb`
- **Ch 5: Matrices and the multiply that powers models**  lab `f5-matrices.py`, notebook `f5-matrices.ipynb`
- **Ch 6: Probability and softmax: turning scores into choices**  lab `f6-probability-softmax.py`, notebook `f6-probability-softmax.ipynb`
- **Ch 7: Derivatives and gradients: which way is downhill**  lab `f7-gradients.py`, notebook `f7-gradients.ipynb`
- **Ch 8: Gradient descent: how models learn**  lab `f8-gradient-descent.py`, notebook `f8-gradient-descent.ipynb`

## 1. LLM Fundamentals: Build a Tiny LLM From Scratch

Build a real language model by hand: tokenizer, embeddings, attention, transformer, training loop, generation. The depth vertical. 10 chapters, live.

- **Ch 1: Tokens: teaching a computer to read**  lab `lab-01-tokenizer.py`, notebook `lab-01-tokenizer.ipynb`
- **Ch 2: Embeddings: giving each token a vector of meaning**  lab `lab-02-embeddings.py`, notebook `lab-02-embeddings.ipynb`
- **Ch 3: The bigram model: your first real predictor**  lab `lab-03-bigram.py`, notebook `lab-03-bigram.ipynb`
- **Ch 4: Self attention: the idea that changed everything**  lab `lab-04-attention.py`, notebook `lab-04-attention.ipynb`
- **Ch 5: Multi head attention: many perspectives at once**  lab `lab-05-multihead.py`, notebook `lab-05-multihead.ipynb`
- **Ch 6: The transformer block: residuals, layer norm, and an MLP**  lab `lab-06-block.py`, notebook `lab-06-block.ipynb`
- **Ch 7: Assemble the GPT: stacking blocks into a model**  lab `lab-07-gpt.py`, notebook `lab-07-gpt.ipynb`
- **Ch 8: The training loop: how a model actually learns**  lab `lab-08-training.py`, notebook `lab-08-training.ipynb`
- **Ch 9: Generation: sampling text from your model**  lab `lab-09-generation.py`, notebook `lab-09-generation.ipynb`
- **Ch 10: Scaling, and what comes next**  lab `lab-10-scaling.py`, notebook `lab-10-scaling.ipynb`

## 2. Prompt Engineering

The first lever and highest-frequency daily skill: few-shot, chain-of-thought, structured output, caching, and eval-driven optimization.

- **Ch 1: Anatomy of a prompt: system, instructions, context**  lab `pe1-anatomy.py`, notebook `pe1-anatomy.ipynb`
- **Ch 2: Few-shot prompting: teaching by example**  lab `pe2-few-shot.py`, notebook `pe2-few-shot.ipynb`
- **Ch 3: Chain-of-thought and reasoning modes**  lab `pe3-chain-of-thought.py`, notebook `pe3-chain-of-thought.ipynb`
- **Ch 4: Structured and JSON output**  lab `pe4-structured-output.py`, notebook `pe4-structured-output.ipynb`
- **Ch 5: XML tagging and output shaping**  lab `pe5-xml-tagging.py`, notebook `pe5-xml-tagging.ipynb`
- **Ch 6: Prompt caching and cost**  lab `pe6-prompt-caching.py`, notebook `pe6-prompt-caching.ipynb`
- **Ch 7: Context engineering**  lab `pe7-context-engineering.py`, notebook `pe7-context-engineering.ipynb`
- **Ch 8: Eval-driven prompt optimization**  lab `pe8-eval-optimization.py`, notebook `pe8-eval-optimization.ipynb`

## 3. LLM Application Engineering

The literal daily job: APIs and SDKs, streaming, structured output, retries and rate limits, cost control, multi-provider routing, observability.

- **Ch 1: Calling model APIs and SDKs**  lab `ae1-api-client.py`, notebook `ae1-api-client.ipynb`
- **Ch 2: Streaming responses (SSE) and TTFT**  lab `ae2-streaming.py`, notebook `ae2-streaming.ipynb`
- **Ch 3: Structured output in production**  lab `ae3-structured-output.py`, notebook `ae3-structured-output.ipynb`
- **Ch 4: Resilience: retries, backoff, rate limits**  lab `ae4-retries-backoff.py`, notebook `ae4-retries-backoff.ipynb`
- **Ch 5: Cost control: routing, caching, budgets**  lab `ae5-cost-budget.py`, notebook `ae5-cost-budget.ipynb`
- **Ch 6: Multi-provider architecture**  lab `ae6-multi-provider-router.py`, notebook `ae6-multi-provider-router.ipynb`
- **Ch 7: Observability for LLM apps**  lab `ae7-observability.py`, notebook `ae7-observability.ipynb`
- **Ch 8: Putting it together: a robust LLM app**  lab `ae8-robust-app.py`, notebook `ae8-robust-app.ipynb`

## 4. RAG and Embeddings

The #1 production pattern (70% of teams). Embeddings, chunking, vector search, re-ranking, end-to-end RAG, and the failure modes that bite.

- **Ch 1: Why models need external memory**  lab `rag1-external-memory.py`, notebook `rag1-external-memory.ipynb`
- **Ch 2: Embeddings for retrieval**  lab `rag2-embeddings.py`, notebook `rag2-embeddings.ipynb`
- **Ch 3: Vector search: nearest neighbors**  lab `rag3-vector-search.py`, notebook `rag3-vector-search.ipynb`
- **Ch 4: Chunking documents well**  lab `rag4-chunking.py`, notebook `rag4-chunking.ipynb`
- **Ch 5: Building the retrieval pipeline**  lab `rag5-pipeline.py`, notebook `rag5-pipeline.ipynb`
- **Ch 6: Retrieval augmented generation, end to end**  lab `rag6-rag-pipeline.py`, notebook `rag6-rag-pipeline.ipynb`
- **Ch 7: Failure modes: hallucination, stale context, bad chunks**  lab `rag7-failure-modes.py`, notebook `rag7-failure-modes.ipynb`
- **Ch 8: Evaluating a RAG system**  lab `rag8-evaluation.py`, notebook `rag8-evaluation.ipynb`

## 5. Evaluation and Testing

Evals replace unit tests. Golden datasets, code graders vs LLM-as-judge, regression gates, hallucination scoring. Table-stakes.

- **Ch 1: Why evals replace unit tests**  lab `ev1-why-evals.py`, notebook `ev1-why-evals.ipynb`
- **Ch 2: Building a golden dataset**  lab `ev2-golden-dataset.py`, notebook `ev2-golden-dataset.ipynb`
- **Ch 3: Code-based graders**  lab `ev3-code-graders.py`, notebook `ev3-code-graders.ipynb`
- **Ch 4: LLM-as-judge and its calibration**  lab `ev4-llm-as-judge.py`, notebook `ev4-llm-as-judge.ipynb`
- **Ch 5: Groundedness and hallucination scoring**  lab `ev5-groundedness.py`, notebook `ev5-groundedness.ipynb`
- **Ch 6: pass@k and statistical literacy**  lab `ev6-pass-at-k.py`, notebook `ev6-pass-at-k.ipynb`
- **Ch 7: Regression gates in CI**  lab `ev7-regression-gate.py`, notebook `ev7-regression-gate.ipynb`
- **Ch 8: Tracing and observability**  lab `ev8-tracing.py`, notebook `ev8-tracing.ipynb`

## 6. Agentic Engineering: How Autonomous Agents Are Built

The center of the certification. An agent is a model in a loop with tools, state and a stopping rule. You build that loop by hand, then add planning, memory, permissions, multi-agent structure, evaluation and a cost model, and learn the protocols that let it plug into anything. Nothing here depends on one vendor product; the applied course after it uses Claude as the worked example.

- **Ch 1: What an agent is, and is not**  lab `ae1-workflow-or-agent.py`, notebook `ae1-workflow-or-agent.ipynb`
- **Ch 2: The loop, by hand**  lab `ae2-loop-by-hand.py`, notebook `ae2-loop-by-hand.ipynb`
- **Ch 3: Tools as contracts**  lab `ae3-tool-contracts.py`, notebook `ae3-tool-contracts.ipynb`
- **Ch 4: Environments and side effects**  lab `ae4-sandbox-environment.py`, notebook `ae4-sandbox-environment.ipynb`
- **Ch 5: Planning and control**  lab `ae5-plan-then-execute.py`, notebook `ae5-plan-then-execute.ipynb`
- **Ch 6: State, memory and context management**  lab `ae6-context-budget.py`, notebook `ae6-context-budget.ipynb`
- **Ch 7: Permissions, human-in-the-loop and reversibility**  lab `ae7-approval-gate.py`, notebook `ae7-approval-gate.ipynb`
- **Ch 8: Protocols: MCP and what comes after it**  lab `ae8-protocol-pattern.py`, notebook `ae8-protocol-pattern.ipynb`
- **Ch 9: Multi-agent systems**  lab `ae9-orchestrator-workers.py`, notebook `ae9-orchestrator-workers.ipynb`
- **Ch 10: Evaluating agents**  lab `ae10-trajectory-vs-outcome.py`, notebook `ae10-trajectory-vs-outcome.ipynb`
- **Ch 11: Observability, cost and latency of loops**  lab `ae11-trace-cost-killswitch.py`, notebook `ae11-trace-cost-killswitch.ipynb`
- **Ch 12: Running agents in production, and the capstone**  lab `ae12-durable-resume.py`, notebook `ae12-durable-resume.ipynb`
- **Ch 13: Appendix: the same loop, in TypeScript**  no runnable lab (appendix chapter)

## 7. Building Agents with Claude: API, Agent SDK and Claude Code

The applied unit that follows the vendor-agnostic theory: everything Agentic Engineering taught in the abstract, built for real on one stack. The Messages API tool handshake, the Agent SDK loop, then Claude Code itself as the worked example, subagents, hooks, MCP, skills and context files, a guarded fleet, and a complete workflow. Version-stamped to current model ids and API shapes, so check the docs when a parameter moves.

- **Ch 1: The Messages API and tool use**  lab `cla1-messages-and-tools.py`, notebook `cla1-messages-and-tools.ipynb`
- **Ch 2: Streaming, structured output, and caching**  lab `cla2-streaming-schema-cache.py`, notebook `cla2-streaming-schema-cache.ipynb`
- **Ch 3: The Agent SDK loop**  lab `cla3-agent-sdk-loop.py`, notebook `cla3-agent-sdk-loop.ipynb`
- **Ch 4: Claude Code as an agent you drive**  lab `cla4-drive-claude-code.py`, notebook `cla4-drive-claude-code.ipynb`
- **Ch 5: Subagents, hooks and MCP in Claude Code**  lab `cla5-subagents-hooks-mcp.py`, notebook `cla5-subagents-hooks-mcp.ipynb`
- **Ch 6: Skills and context files**  lab `cla6-skills-and-context.py`, notebook `cla6-skills-and-context.ipynb`
- **Ch 7: Orchestrating a fleet**  lab `cla7-fleet.py`, notebook `cla7-fleet.ipynb`
- **Ch 8: A complete agentic workflow**  lab `cla8-workflow.py`, notebook `cla8-workflow.ipynb`

## 8. AI Security: Models and Agents

The OWASP LLM Top 10 and prompt-injection offense and defense, then the agent attack surface: the lethal trifecta, injection through tool results, MCP tool poisoning, the confused deputy, and red-teaming your own agent. Ada's differentiator.

- **Ch 1: The LLM attack surface and the OWASP LLM Top 10**  lab `se1-attack-surface.py`, notebook `se1-attack-surface.ipynb`
- **Ch 2: Prompt injection: direct (LLM01)**  lab `se2-direct-injection.py`, notebook `se2-direct-injection.ipynb`
- **Ch 3: Indirect injection and RAG poisoning (LLM08)**  lab `se3-indirect-injection.py`, notebook `se3-indirect-injection.ipynb`
- **Ch 4: System-prompt extraction and leakage (LLM07)**  lab `se4-system-prompt-extraction.py`, notebook `se4-system-prompt-extraction.ipynb`
- **Ch 5: Guardrails and input/output filtering**  lab `se5-guardrails.py`, notebook `se5-guardrails.ipynb`
- **Ch 6: Tool and agent abuse prevention (LLM06)**  lab `se6-tool-abuse.py`, notebook `se6-tool-abuse.ipynb`
- **Ch 7: Sensitive-data disclosure and exfiltration (LLM02)**  lab `se7-data-exfiltration.py`, notebook `se7-data-exfiltration.ipynb`
- **Ch 8: Red-team your own app: the attack harness**  lab `se8-red-team-harness.py`, notebook `se8-red-team-harness.ipynb`
- **Ch 9: The agent attack surface**  lab `se9-agent-attack-surface.py`, notebook `se9-agent-attack-surface.ipynb`
- **Ch 10: Red-team your own agent**  lab `se10-red-team-agent.py`, notebook `se10-red-team-agent.ipynb`

## 9. AI Engineering in Production

Ship and operate: serving, quantization (AWQ/GGUF), vLLM/KV cache, eval-gated CI/CD, cost and drift monitoring, LLMOps.

- **Ch 1: Serving models: latency and throughput**  lab `pr1-serving-batching.py`, notebook `pr1-serving-batching.ipynb`
- **Ch 2: Quantization: running models cheaply**  lab `pr2-quantization.py`, notebook `pr2-quantization.ipynb`
- **Ch 3: The KV cache and the memory cost of serving**  lab `pr3-kv-cache-cost.py`, notebook `pr3-kv-cache-cost.ipynb`
- **Ch 4: Evals as a production gate**  lab `pr4-eval-gate.py`, notebook `pr4-eval-gate.ipynb`
- **Ch 5: Cost monitoring and budgets**  lab `pr5-cost-monitor.py`, notebook `pr5-cost-monitor.ipynb`
- **Ch 6: Drift and quality monitoring**  lab `pr6-drift-detector.py`, notebook `pr6-drift-detector.ipynb`
- **Ch 7: The LLMOps lifecycle: versioning and rollback**  lab `pr7-llmops-lifecycle.py`, notebook `pr7-llmops-lifecycle.ipynb`
- **Ch 8: Shipping: serving, gating, and monitoring as one system**  lab `pr8-capstone.py`, notebook `pr8-capstone.ipynb`

## 10. Training and Fine-tuning

Shape a base model: datasets, loss curves, SFT and LoRA/QLoRA, DPO, and the judgment of when to fine-tune vs prompt vs RAG.

- **Ch 1: Datasets: the raw material a model learns from**  lab `tr1-datasets.py`, notebook `tr1-datasets.ipynb`
- **Ch 2: Reading loss curves: overfitting and validation**  lab `tr2-loss-curves.py`, notebook `tr2-loss-curves.ipynb`
- **Ch 3: Supervised fine-tuning (SFT)**  lab `tr3-sft.py`, notebook `tr3-sft.ipynb`
- **Ch 4: LoRA from scratch: freeze the base, learn a little**  lab `tr4-lora-from-scratch.py`, notebook `tr4-lora-from-scratch.ipynb`
- **Ch 5: LoRA and QLoRA in practice: base vs adapter**  lab `tr5-lora-base-vs-adapter.py`, notebook `tr5-lora-base-vs-adapter.ipynb`
- **Ch 6: Preference tuning with DPO**  lab `tr6-dpo-preference.py`, notebook `tr6-dpo-preference.ipynb`
- **Ch 7: Evaluating a fine-tune without fooling yourself**  lab `tr7-evaluate-finetune.py`, notebook `tr7-evaluate-finetune.ipynb`
- **Ch 8: Putting it together: fine-tune vs prompt vs RAG**  lab `tr8-finetune-vs-prompt-vs-rag.py`, notebook `tr8-finetune-vs-prompt-vs-rag.ipynb`

## 11. Transformers Deep Dive

Explain the internals cold: positional encodings (RoPE), attention variants, BPE, KV cache, scaling laws. The T-shape depth vertical.

- **Ch 1: Positional encodings: from sinusoidal to RoPE**  lab `tf1-positional-encoding.py`, notebook `tf1-positional-encoding.ipynb`
- **Ch 2: Attention variants: MHA, MQA, and GQA**  lab `tf2-attention-variants.py`, notebook `tf2-attention-variants.ipynb`
- **Ch 3: Tokenization at scale: byte-pair encoding**  lab `tf3-bpe.py`, notebook `tf3-bpe.ipynb`
- **Ch 4: The KV cache and efficient inference**  lab `tf4-kv-cache.py`, notebook `tf4-kv-cache.ipynb`
- **Ch 5: Normalization and residuals in depth**  lab `tf5-layernorm-residual.py`, notebook `tf5-layernorm-residual.ipynb`
- **Ch 6: The full block, and a hard proof of causality**  lab `tf6-block-causality.py`, notebook `tf6-block-causality.ipynb`
- **Ch 7: Scaling laws**  lab `tf7-scaling-laws.py`, notebook `tf7-scaling-laws.ipynb`
- **Ch 8: Why transformers work**  lab `tf8-why-transformers-work.py`, notebook `tf8-why-transformers-work.ipynb`

## 12. Multimodal AI

Vision-language, image generation, speech (STT/TTS), and multimodal RAG. The converging edge.

- **Ch 1: Image representation: pixels to patches**  lab `mm1-image-patches.py`, notebook `mm1-image-patches.ipynb`
- **Ch 2: Vision-language models: one shared space**  lab `mm2-clip-matching.py`, notebook `mm2-clip-matching.ipynb`
- **Ch 3: Image generation: denoising step by step**  lab `mm3-image-generation.py`, notebook `mm3-image-generation.ipynb`
- **Ch 4: Speech: STT and TTS concepts**  lab `mm4-speech.py`, notebook `mm4-speech.ipynb`
- **Ch 5: Multimodal RAG: retrieving over mixed media**  lab `mm5-multimodal-rag.py`, notebook `mm5-multimodal-rag.ipynb`
- **Ch 6: Putting multimodal together: a tiny assistant**  lab `mm6-assistant.py`, notebook `mm6-assistant.ipynb`

## 13. Interview Prep and System Design

Ace the interview: the real loop, from-scratch coding, LLM system design, take-home patterns, behavioral, and question banks.

- **Ch 1: The real interview loop**  lab `ip1-cosine-retrieval.py`, notebook `ip1-cosine-retrieval.ipynb`
- **Ch 2: Practical coding rounds**  lab `ip2-tokenizer.py`, notebook `ip2-tokenizer.ipynb`
- **Ch 3: From-scratch ML coding**  lab `ip3-attention.py`, notebook `ip3-attention.ipynb`
- **Ch 4: ML and LLM theory with judgment**  lab `ip4-rag-vs-finetune-decision.py`, notebook `ip4-rag-vs-finetune-decision.ipynb`
- **Ch 5: LLM system design**  lab `ip5-sizing-calculator.py`, notebook `ip5-sizing-calculator.ipynb`
- **Ch 6: Take-home projects that pass**  lab `ip6-take-home-grader.py`, notebook `ip6-take-home-grader.ipynb`
- **Ch 7: Behavioral and safety mindset**  lab `ip7-sampling.py`, notebook `ip7-sampling.ipynb`
- **Ch 8: Question banks and mock rounds**  lab `ip8-softmax.py`, notebook `ip8-softmax.ipynb`
- **Ch 9: The agentic system design round**  lab `ip9-agentic-design-rubric.py`, notebook `ip9-agentic-design-rubric.ipynb`

## 14. Capstone Projects

What gets you hired: 4 deployed, evaluated builds. RAG assistant, an autonomous agent with evals and permissions and a cost report, reusable eval pipeline, self-red-teamed app.

- **Ch 1: The portfolio that gets hired**  lab `cap1-portfolio-rubric.py`, notebook `cap1-portfolio-rubric.ipynb`
- **Ch 2: Capstone 1: a deployed RAG assistant that cites its sources**  lab `cap2-rag-assistant.py`, notebook `cap2-rag-assistant.ipynb`
- **Ch 3: Capstone 2: an autonomous agent with evals, permissions, and a cost report**  lab `cap-agentic.py`, notebook `cap-agentic.ipynb`
- **Ch 4: Capstone 3: a reusable eval pipeline that gates a build**  lab `cap4-eval-pipeline.py`, notebook `cap4-eval-pipeline.ipynb`
- **Ch 5: Capstone 4: a self-red-teamed, secured LLM app**  lab `cap6-red-team-app.py`, notebook `cap6-red-team-app.ipynb`
- **Ch 6: Packaging: manifest, READMEs, demo links, and the write-up**  lab `cap7-packaging.py`, notebook `cap7-packaging.ipynb`
