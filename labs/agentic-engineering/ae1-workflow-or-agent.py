#!/usr/bin/env python3
"""
LAB AE1: What an agent is, and is not.

An agent has four parts: a model, tools, state, and a stopping rule. Take away
the freedom to choose the path and you no longer have an agent, you have a
workflow: a fixed pipeline that happens to call a model. Most production systems
called "agents" should be workflows, because when the path is known in advance,
letting the model rediscover it every run costs more and fails in more ways.

This lab runs the SAME fixed task two ways over the same four reviews: a
workflow with a hardcoded path, and an agent that decides its own route per
item. It counts decision steps and checks answers against ground truth. The
workflow wins on both, which is the whole point.

Run: python3 modules/academy-content/labs/agentic-engineering/ae1-workflow-or-agent.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, tool_route

# The task is FIXED and known: label each review. Ground truth is known too, so
# accuracy is measurable rather than vibes.
REVIEWS = [
    ("I love this, the best purchase all year", "positive"),
    ("Terrible, it broke on day one", "negative"),
    ("The checkout math is broken and I hate the slow refund", "negative"),
    ("Excellent build and a wonderful price", "positive"),
]
TOOLS = ["sentiment", "calculator", "search", "weather"]

# ── The WORKFLOW. One hardcoded path, no routing decision at all. ─────────────
print("WORKFLOW (fixed path: classify each review, no route decision)")
wf_steps, wf_right = 0, 0
for text, truth in REVIEWS:
    wf_steps += 1                                    # one model call per item
    got = complete("Classify the sentiment: " + text)
    wf_right += (got == truth)
    print(f"  {got:<9} (truth {truth:<9}) {text[:38]!r}")

# ── The AGENT. Same task, but it picks its own tool per item first. ───────────
# That extra freedom is the only difference, and it is what breaks.
print("")
print("AGENT (free to route each item itself, then act)")
ag_steps, ag_right = 0, 0
for text, truth in REVIEWS:
    ag_steps += 1                                    # step 1: decide
    picked = tool_route(text, TOOLS) or "sentiment"
    ag_steps += 1                                    # step 2: act
    got = complete("Classify the sentiment: " + text) if picked == "sentiment" else "unknown"
    ag_right += (got == truth)
    print(f"  routed -> {picked:<11} answer {got:<9} (truth {truth})")

print("")
print(f"workflow : {wf_steps} steps, {wf_right}/{len(REVIEWS)} correct")
print(f"agent    : {ag_steps} steps, {ag_right}/{len(REVIEWS)} correct")

# Determinism: the workflow's path cannot vary, so a second run is identical.
second = [complete("Classify the sentiment: " + t) for t, _ in REVIEWS]
first = [complete("Classify the sentiment: " + t) for t, _ in REVIEWS]
print(f"workflow repeatable across runs: {second == first}")

ok = (wf_steps < ag_steps) and (wf_right > ag_right) and (second == first)
print("")
print(f"FIXED WORKFLOW BEAT THE AGENT ON THE FIXED TASK: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Reach for an agent when the path is unknown. Next: build the loop by hand.")
