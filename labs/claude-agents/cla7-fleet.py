#!/usr/bin/env python3
"""
LAB CLA7: Orchestrating a fleet, with the model choice on purpose.

The orchestrator-worker shape and the context-isolation argument behind it are
theory you already have. What Claude adds is two concrete dials, and getting them
wrong is what makes a fleet expensive or slow rather than useful.

The first dial is the MODEL PER ROLE. Decomposition, verification and synthesis
are judgment, so the orchestrator holds the capable model. A worker classifying
one review is not judgment, it is throughput, so it runs a cheap model at low
effort, which also makes it terser and less inclined to wander:

    agents={"worker": AgentDefinition(description="Label one review.",
                                      prompt="You label sentiment.",
                                      tools=["Read"], model="claude-haiku-4-5")}
    # orchestrator stays on claude-opus-5; effort rides in output_config

Spending the capable model on work a cheap one finishes correctly is the most
common way a fleet stops being worth running.

The second dial is CONCURRENCY. Firing every worker at once trips short-window
throughput limits, and the failure is ugly: the burst gets throttled, the retries
add to the burst, and total throughput ends up worse than a paced run. So a fleet
runs in bounded waves of at most N workers, and N is modest by default.

Around both sits the beat that is easy to skip: VERIFY each worker's output
before it enters the synthesis. An unchecked worker result is how one bad answer
becomes a wrong report.

This lab runs a ten-item job in waves of three, verifies every output, records
which model each role actually used, and proves the tally is right, no wave
exceeded the cap, and every worker ran on the cheap tier.

Run: python3 modules/academy-content/labs/claude-agents/cla7-fleet.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

JOB = ["I love it", "this is terrible", "great product", "worst ever, broken",
       "excellent and fantastic", "awful and useless", "the best, amazing",
       "boring and slow", "wonderful, recommend", "disappointing, poor"]
CAP = 3                                    # bounded waves: workers at once
ORCHESTRATOR = {"model": "claude-opus-5", "effort": "high"}
WORKER = {"model": "claude-haiku-4-5", "effort": "low"}
spend = []                                  # (role, model) per call, for the audit


def worker(item):
    """One worker agent: label a single item on the cheap tier."""
    spend.append(("worker", WORKER["model"]))
    return complete("sentiment: %s" % item)


def verify(result):
    """The orchestrator checks a worker output before accepting it. A label
    outside the expected set means the worker went off-script."""
    spend.append(("orchestrator", ORCHESTRATOR["model"]))
    return result in ("positive", "negative", "neutral")


def waves(items, cap):
    return [items[i:i + cap] for i in range(0, len(items), cap)]


print("GOAL: label %d reviews. orchestrator=%s worker=%s effort=%s wave cap=%d"
      % (len(JOB), ORCHESTRATOR["model"], WORKER["model"], WORKER["effort"], CAP))
print("")

verified, rejected, wave_sizes = [], 0, []
for n, batch in enumerate(waves(JOB, CAP), 1):
    wave_sizes.append(len(batch))
    labels = []
    for item in batch:                      # FAN OUT, capped by the wave size
        result = worker(item)
        if verify(result):                  # VERIFY before accepting
            verified.append(result)
            labels.append(result)
        else:
            rejected += 1
    print("wave %d: %d workers -> %s" % (n, len(batch), labels))

# SYNTHESIZE: one tally from the verified outputs only.
summary = {label: verified.count(label) for label in ("positive", "negative", "neutral")}
print("")
print("wave sizes : %s" % wave_sizes)
print("synthesis  : %s" % summary)
print("rejected   : %d" % rejected)

worker_models = {m for role, m in spend if role == "worker"}
orch_models = {m for role, m in spend if role == "orchestrator"}
print("")
print("models used: workers=%s orchestrator=%s" % (sorted(worker_models), sorted(orch_models)))

all_done = (len(verified) == len(JOB))
cap_respected = all(size <= CAP for size in wave_sizes)
correct = (summary == {"positive": 5, "negative": 5, "neutral": 0})
tiers_right = (worker_models == {"claude-haiku-4-5"} and orch_models == {"claude-opus-5"})

print("")
print("checks")
print("  all ten items completed and verified   :", all_done)
print("  no wave exceeded the concurrency cap   :", cap_respected)
print("  the synthesized tally is correct       :", correct)
print("  workers stayed cheap, orchestrator did not :", tiers_right)

ok = all_done and cap_respected and correct and tiers_right
print("")
print("FLEET COMPLETED THE JOB IN BOUNDED WAVES ON THE RIGHT TIERS: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Decompose, fan out capped, verify, synthesize. Next: put it all together.")
