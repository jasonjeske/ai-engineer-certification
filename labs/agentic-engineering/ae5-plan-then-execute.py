#!/usr/bin/env python3
"""
LAB AE5: Plan-then-execute versus a bare ReAct loop.

A ReAct loop decides one step at a time and keeps no record of what is left to
do, so on a task with ordered dependencies it can re-observe the same thing
forever. Plan-then-execute writes the ordered sub-steps down FIRST, which costs
one extra call and buys two things: the loop always knows what remains, and a
human can read the plan before anything happens. Then a VERIFICATION LOOP checks
the answer against independently computed evidence instead of trusting the
agent's own claim of success. Same task, both strategies, in one run.

Run: python3 modules/academy-content/labs/agentic-engineering/ae5-plan-then-execute.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import re

INVOICES = [120, 340, 90]           # the facts on the table
TASK = "add invoices 120 and 340 and 90 then take 10 percent of the total"


def calculator(expr, register=None):
    """Arithmetic on ONE binary expression. `$R` means "the last result", which
    is how a planned step refers to the step before it."""
    if register is not None:
        expr = expr.replace("$R", str(register))
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*percent\s+of\s+(-?\d+(?:\.\d+)?)", expr, re.I)
    if m:
        return float(m.group(1)) / 100.0 * float(m.group(2))
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("cannot parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    return {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else 0.0}[op]


# --- STRATEGY 1: bare ReAct. One decision per step, no record of what remains.
print("STEP 1: bare ReAct on the whole ask, 4 steps max")
react_obs, seen = None, []
for step in range(4):
    tool = tool_route(TASK, ["calculator", "search", "weather"])
    try:
        react_obs = calculator(TASK)        # the tool needs ONE expression, not three
    except ValueError as e:
        react_obs = "error: %s" % e
    seen.append(str(react_obs))
    print(f"  step {step + 1}: tool={tool} observation={react_obs}")
# Nothing recorded which sub-step comes next, so every step re-sends the whole
# task and gets the identical observation back. That is wandering, not progress.
react_wandered = len(set(seen)) == 1 and react_obs != 55.0
print(f"  spent 4 steps, produced {react_obs!r}, needed 55.0 -> wandered: {react_wandered}")

# --- STRATEGY 2: plan-then-execute. Write the ordered plan BEFORE acting.
print("")
print("STEP 2: plan-then-execute, plan written before any action")
plan = [f"calculator: {INVOICES[0]} + {INVOICES[1]}",
        "calculator: $R + %d" % INVOICES[2],
        "calculator: 10 percent of $R"]
for i, line in enumerate(plan, 1):       # the plan is state, printed and re-readable
    print(f"  plan[{i}] {line}")
actions_taken = 0
register = None
for i, line in enumerate(plan, 1):
    expr = line.split("calculator:", 1)[1].strip()
    register = calculator(expr, register)
    actions_taken += 1
    print(f"  exec plan[{i}] -> {register}")
plan_answer = register

# --- VERIFICATION LOOP: check against evidence, do not take the agent's word.
print("")
print("STEP 3: verify the answer against independently computed evidence")
evidence = 0.10 * sum(INVOICES)
verified = abs(plan_answer - evidence) < 1e-9
print(f"  agent claims {plan_answer}, evidence says {evidence} -> verified: {verified}")

plan_auditable = actions_taken == len(plan)   # every action maps to a plan line
print("")
print(f"bare ReAct reached the answer      : {react_obs == 55.0}")
print(f"bare ReAct wandered instead        : {react_wandered}")
print(f"plan was written before acting     : True ({len(plan)} steps)")
print(f"every action mapped to a plan line : {plan_auditable}")
print(f"answer verified against evidence   : {verified}")

ok = react_wandered and plan_auditable and verified and plan_answer == 55.0
print("")
print(f"PLAN-THEN-EXECUTE WAS AUDITABLE AND CORRECT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("The plan is state the agent re-reads. Next: the context window it all lives in.")
