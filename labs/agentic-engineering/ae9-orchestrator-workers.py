#!/usr/bin/env python3
"""
LAB AE9: Multi-agent systems. Orchestrator, workers, and the shared-state race.

The dominant real multi-agent shape is orchestrator-worker: one agent owns the
goal and splits it into independent subtasks, each worker handles exactly one,
and the orchestrator merges the results. The first reason to split is CONTEXT
ISOLATION, not speed: each worker only ever sees its own slice, so its prompt
stays small while a single agent doing all three would carry all three slices at
once. This lab measures that, then reproduces the failure that bites every team
that shares mutable state between workers, and fixes it.

Run: python3 modules/academy-content/labs/agentic-engineering/ae9-orchestrator-workers.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# ── The work: three independent reviews. Independent is the precondition for ──
# fanning out at all. Anything where task B needs task A's output is a HANDOFF,
# which is sequential by definition, and no amount of parallelism helps it.
TASKS = [
    ("r1", "I love this laptop, the screen is excellent"),
    ("r2", "the battery is terrible and the fans are awful"),
    ("r3", "it arrived on tuesday in a box"),
]
TRUTH = {"r1": "positive", "r2": "negative", "r3": "neutral"}


def worker(task_id, text, state):
    """One worker agent. It sees ONE review, never the other two. `state` is the
    dict it is told to report into, which is the whole point of this lab."""
    prompt = "Classify the sentiment of this review: %s" % text
    state["result"] = complete(prompt)          # <-- the bug, kept on purpose
    state["by_id"][task_id] = complete(prompt)  # <-- the fix, same call
    return len(prompt)


# ── STEP 1: context isolation, measured. ──────────────────────────────────────
solo_prompt = "Classify the sentiment of each review: " + " | ".join(t for _, t in TASKS)
print("STEP 1: context isolation")
worker_sizes = [len("Classify the sentiment of this review: %s" % t) for _, t in TASKS]
print("  one agent, all three reviews : %d prompt chars" % len(solo_prompt))
print("  each worker, its own review  : %s chars (max %d)" % (worker_sizes, max(worker_sizes)))
isolated = max(worker_sizes) < len(solo_prompt)
print("  worker context is smaller    : %s" % isolated)

# ── STEP 2: the hazard. Every worker writes the SAME key. ─────────────────────
print("")
print("STEP 2: shared mutable state, all workers writing state['result']")
shared = {"result": None, "by_id": {}}
for task_id, text in TASKS:
    worker(task_id, text, shared)
    print("  after %s -> state['result'] = %r" % (task_id, shared["result"]))
racy = [shared["result"]]
print("  aggregated results survived  : %d of %d" % (len(racy), len(TASKS)))

# ── STEP 3: the fix. One namespace per worker, merged by the orchestrator. ────
print("")
print("STEP 3: isolated per-worker state, merged at the end")
merged = {}
for task_id, text in TASKS:
    own = {"result": None, "by_id": {}}   # this worker's private scratch space
    worker(task_id, text, own)
    merged.update(own["by_id"])           # the orchestrator owns the merge
    print("  worker %s returned %r" % (task_id, own["by_id"][task_id]))
print("  aggregated results survived  : %d of %d" % (len(merged), len(TASKS)))

# ── STEP 4: did the orchestrator actually get the right answers? ──────────────
print("")
print("STEP 4: correctness of the merged result")
for task_id, expected in sorted(TRUTH.items()):
    print("  %s expected %-8s got %-8s %s" % (task_id, expected, merged[task_id],
                                              "ok" if merged[task_id] == expected else "WRONG"))

lost_to_race = len(TASKS) - len(racy)
ok = (isolated
      and lost_to_race == 2
      and merged == TRUTH
      and len(merged) == len(TASKS))

print("")
print("  results lost to the shared key : %d" % lost_to_race)
print("ISOLATED WORKER STATE FIXED THE RACE AND THE ORCHESTRATOR MERGED ALL 3: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Split for context first, parallelism second. One agent with better tools often wins.")
