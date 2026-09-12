#!/usr/bin/env python3
"""
LAB AE2: The loop, by hand, and the three ways it dies.

The loop is observe, decide, act, append, repeat. The transcript IS the state: a
list that grows by one entry per step, and the only memory the agent has. Three
things kill a loop in production: nothing reads the stop signal, the model names
a tool that does not exist, and the model never signals done at all. Each has a
specific fix, and this lab runs all three and proves the guarded loop survives.

Run: python3 modules/academy-content/labs/agentic-engineering/ae2-loop-by-hand.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route

REGISTRY = {
    "weather": lambda city: {"paris": 22, "tokyo": 30}[city.strip().lower()],
    "calculator": lambda expr: sum(int(x) for x in expr.replace("+", " ").split()),
}

def act(tool, arg):
    """FIX for death 2: an unknown tool or a thrown error becomes an OBSERVATION
    the loop can read, never an exception that escapes and kills the run."""
    if tool not in REGISTRY:
        return "ERROR: no tool %r. available: %s" % (tool, sorted(REGISTRY))
    try:
        return REGISTRY[tool](arg)
    except Exception as e:
        return "ERROR: %s failed: %s" % (tool, e)

def run(goal, script, max_steps, honor_stop=True):
    """The whole loop. `script` stands in for the model's chosen calls; a real
    agent parses these out of the model's output each turn."""
    transcript = ["GOAL: " + goal]          # state = a list that only grows
    answer, steps, ended_by = None, 0, "step cap"
    while steps < max_steps:                # FIX for death 3: the cap
        tool, arg = script[steps] if steps < len(script) else ("weather", "paris")
        steps += 1
        if tool == "final":                 # the model's stop signal
            if honor_stop:                  # FIX for death 1: somebody reads it
                answer, ended_by = arg, "stop signal"
                transcript.append("step %d: STOP with %r" % (steps, arg))
                break
            transcript.append("step %d: STOP ignored" % steps)
            continue
        obs = act(tool, arg)
        transcript.append("step %d: %s(%r) -> %r" % (steps, tool, arg, obs))
    return answer, steps, ended_by, transcript

GOAL = "the temperature in Paris plus 10"
GOOD = [("weather", "paris"), ("calculator", "22 + 10"), ("final", "32")]

print("DEATH 1: nothing reads the stop signal")
a, s, why, _ = run(GOAL, GOOD, max_steps=6, honor_stop=False)
print("  unguarded: %d steps, answer %r, ended by %s" % (s, a, why))
a1, s1, why1, t1 = run(GOAL, GOOD, max_steps=6, honor_stop=True)
print("  guarded  : %d steps, answer %r, ended by %s" % (s1, a1, why1))
d1 = (s == 6 and a is None) and (s1 == 3 and a1 == "32" and why1 == "stop signal")

print("")
print("DEATH 2: the model calls a tool that does not exist")
try:
    REGISTRY["database"]("paris")           # what a naive dispatcher does
except KeyError as e:
    print("  naive dispatch raised KeyError(%s) and the run would be over" % e)
a2, s2, why2, t2 = run(GOAL, [("database", "paris")] + GOOD, 6)
print("  guarded  : %d steps, answer %r, ended by %s" % (s2, a2, why2))
print("  recovery line -> %s" % t2[1])
d2 = a2 == "32" and any("ERROR" in line for line in t2)

print("")
print("DEATH 3: the model never signals done")
a3, s3, why3, t3 = run(GOAL, [("weather", "paris"), ("calculator", "22 + 10")], 4)
print("  guarded  : %d steps, answer %r, ended by %s" % (s3, a3, why3))
d3 = a3 is None and s3 == 4 and why3 == "step cap"

print("")
print("transcript grew to %d entries, one per step plus the goal" % len(t2))
print("routing still works on the ask: %r" % tool_route("what is the weather in Paris", list(REGISTRY)))
ok = d1 and d2 and d3
print("")
print("LOOP TERMINATED CORRECTLY IN ALL THREE DEATH MODES: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Stop rule, error-as-observation, step cap. Next: the tools it calls.")
