#!/usr/bin/env python3
"""
LAB AE10: Grading an agent. Trajectory and outcome are two different scores.

A single-call eval has one thing to grade: the answer. An agent has two, and
they come apart. An agent can reach the right answer down a stupid path (it
guessed, and next week the guess will be wrong), and it can walk a perfect path
to a wrong answer (one tool returned bad data). Score them SEPARATELY or you
cannot tell those two failures apart, and they need opposite fixes.

This lab grades four real transcripts of the same task, computes pass@k and
pass^k over them, and mines a regression case out of the transcript that failed.

Run: python3 modules/academy-content/labs/agentic-engineering/ae10-trajectory-vs-outcome.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

# ── The task under test, and what a competent run looks like. ─────────────────
TASK = "the temperature in Paris plus 10"
REQUIRED_PATH = ["weather", "calculator"]   # in this order; weather feeds the math
CORRECT_ANSWER = 32

# Four attempts at the SAME task. This is what a task suite records: the actions
# taken, in order, and the final answer. Not just the answer.
TRANSCRIPTS = [
    {"run": "A", "calls": [("weather", "Paris"), ("calculator", "22 + 10")], "answer": 32},
    {"run": "B", "calls": [("calculator", "22 + 10")],                       "answer": 32},
    {"run": "C", "calls": [("weather", "Paris"), ("calculator", "22 + 10")], "answer": 23},
    {"run": "D", "calls": [("search", "paris temp"), ("weather", "Paris")],  "answer": 15},
]


def grade_trajectory(calls):
    """Did it take a sane PATH? Two checks a real grader always wants: every
    required tool was used, and they happened in a workable order. Extra tools
    are not fatal on their own, they are noise worth reporting."""
    used = [t for t, _ in calls]
    missing = [t for t in REQUIRED_PATH if t not in used]
    order_ok = False
    if not missing:
        order_ok = used.index(REQUIRED_PATH[0]) < used.index(REQUIRED_PATH[1])
    extra = [t for t in used if t not in REQUIRED_PATH]
    return {"pass": (not missing) and order_ok, "missing": missing, "extra": extra}


def grade_calls(calls):
    """Grade each tool call on its own. The final answer can hide a broken call,
    and a broken call is the thing you actually have to fix."""
    results = []
    for tool, arg in calls:
        if tool == "weather":
            good = arg.strip().lower() == "paris"
        elif tool == "calculator":
            good = "+ 10" in arg or "+10" in arg
        else:
            good = False        # a tool that is not in the required path is not "good"
        results.append((tool, arg, good))
    return results


def grade_outcome(answer):
    """Did it get the right ANSWER? One comparison, and it tells you nothing
    about how."""
    return answer == CORRECT_ANSWER


print("STEP 1: grade every transcript on both axes")
print("  run  trajectory  outcome  note")
graded = []
for t in TRANSCRIPTS:
    traj = grade_trajectory(t["calls"])
    out = grade_outcome(t["answer"])
    note = ""
    if traj["missing"]:
        note = "skipped %s" % ",".join(traj["missing"])
    elif traj["extra"]:
        note = "extra call to %s" % ",".join(traj["extra"])
    if not out:
        note = (note + "; " if note else "") + "answered %r" % t["answer"]
    graded.append({"run": t["run"], "traj": traj["pass"], "out": out})
    print("  %-4s %-11s %-8s %s" % (t["run"],
                                    "PASS" if traj["pass"] else "FAIL",
                                    "PASS" if out else "FAIL", note))

# ── STEP 2: the cases that matter are the ones where the two scores disagree. ─
print("")
print("STEP 2: where the two graders disagree")
disagreements = [g for g in graded if g["traj"] != g["out"]]
for g in disagreements:
    if g["out"] and not g["traj"]:
        print("  run %s: right answer, bad path. It never called weather, it "
              "assumed 22. Luck, and it will not hold." % g["run"])
    else:
        print("  run %s: good path, wrong answer. The loop was fine, a step "
              "computed wrong. Fix the step, not the plan." % g["run"])
print("  disagreements: %d of %d" % (len(disagreements), len(graded)))

# ── STEP 3: per-call grading on the worst run. ─────────────────────────────────
print("")
print("STEP 3: individual tool calls, run D")
for tool, arg, good in grade_calls(TRANSCRIPTS[3]["calls"]):
    print("  %-11s %-16r %s" % (tool, arg, "ok" if good else "BAD CALL"))

# ── STEP 4: pass@k is optimism. pass^k is reliability. ────────────────────────
print("")
print("STEP 4: reliability over %d attempts" % len(graded))
fully_correct = [g for g in graded if g["traj"] and g["out"]]
k = len(graded)
pass_at_k = 1 if fully_correct else 0
pass_pow_k = 1 if len(fully_correct) == k else 0
print("  fully correct runs : %d of %d (%s)" % (len(fully_correct), k,
                                                ",".join(g["run"] for g in fully_correct)))
print("  pass@%d (any one)   : %d   <- what a demo reports" % (k, pass_at_k))
print("  pass^%d (all of k)  : %d   <- what production needs" % (k, pass_pow_k))

# ── STEP 5: a failure in the wild becomes a regression case. ──────────────────
print("")
print("STEP 5: mine a regression case from the failing transcript")
worst = TRANSCRIPTS[3]
case = {"task": TASK, "forbid_tool": "search", "require_path": REQUIRED_PATH,
        "expect_answer": CORRECT_ANSWER, "from_run": worst["run"]}
print("  new suite case: %s" % case)

ok = (len(disagreements) == 2
      and pass_at_k == 1
      and pass_pow_k == 0
      and len(fully_correct) == 1)
print("")
print("TRAJECTORY AND OUTCOME DISAGREED ON 2 OF 4 RUNS AND PASS^K CAUGHT IT: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("One green demo run is not a passing agent. Grade the path, the calls, and all of k.")
