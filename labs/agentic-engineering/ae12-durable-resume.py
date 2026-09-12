#!/usr/bin/env python3
"""
LAB AE12: Durable execution. A loop that survives the process dying.

An interactive agent that crashes is an annoyance, because a human retries it. A
background agent that crashes on step 3 of 5 is a data problem: the earlier steps
already touched the world, and a naive restart does them again. Durable execution
is two rules. Persist state after every step, so a new process resumes instead of
restarting. Make every tool idempotent, because there is always a window where
the effect landed and the checkpoint did not, and the only thing that can cover
that window is the tool checking for its own effect before applying it.

This lab kills the loop twice, including once INSIDE that window, and proves the
work was neither lost nor duplicated.

Run: python3 modules/academy-content/labs/agentic-engineering/ae12-durable-resume.py
"""
import sys, os, json, tempfile
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

PLAN = ["fetch", "parse", "enrich", "score", "publish"]
STATE = os.path.join(tempfile.gettempdir(), "ae12_agent_state.json")

# WORLD stands in for real side effects: rows written, emails sent, payments
# taken. A duplicate entry here is the bug. ATTEMPTS counts how many times a tool
# was CALLED, which is a different number, and the gap between them is the point.
WORLD = []
ATTEMPTS = []


class Crash(Exception):
    """A process death, not a handled error. Nothing gets to clean up."""


def save(state):
    """Atomic write. A half-written state file is worse than none, because the
    next process would read it and trust it. Write a temp file, then rename:
    rename is the one filesystem operation that cannot partially happen."""
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE)


def load():
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {"done": [], "results": {}}


def do_step(name):
    """An IDEMPOTENT tool. Before applying anything it looks for its own effect,
    keyed on the step name, and does nothing if it is already there. The key must
    live where the EFFECT lives, not in the agent's state file, because the whole
    failure being covered is the state file not getting written."""
    ATTEMPTS.append(name)
    if name in WORLD:
        print("      %-8s effect already present, tool is a no-op" % name)
        return "%s:ok" % name
    WORLD.append(name)
    return "%s:ok" % name


def run(label, crash_before=None, crash_after_effect=None):
    """One 'process'. It loads whatever the last one left behind and continues.
    Nothing in here knows whether it is the first attempt or the third."""
    state = load()
    print("  [%s] resumed with %d/%d done: %s"
          % (label, len(state["done"]), len(PLAN), state["done"] or "nothing"))
    for name in PLAN:
        if name in state["done"]:
            continue
        if name == crash_before:
            print("      %-8s <-- DIES BEFORE the tool runs" % name)
            raise Crash(name)
        result = do_step(name)
        if name == crash_after_effect:
            # The dangerous window: the effect is real, the checkpoint is not.
            print("      %-8s <-- DIES AFTER the effect, BEFORE the checkpoint" % name)
            raise Crash(name)
        state["done"].append(name)
        state["results"][name] = result
        save(state)                       # the checkpoint: after EVERY step
        print("      %-8s done, state persisted (%d/%d)"
              % (name, len(state["done"]), len(PLAN)))
    return state


if os.path.exists(STATE):
    os.remove(STATE)

# ── STEP 1: a clean crash. Nothing was half-done. ────────────────────────────
print("STEP 1: first process dies before 'enrich' even runs")
try:
    run("process 1", crash_before="enrich")
    print("  did not crash, which this lab requires"); sys.exit(1)
except Crash as e:
    print("  [process 1] killed at %r" % str(e))
print("  state on disk : %s" % load()["done"])
print("  world         : %s" % WORLD)

# ── STEP 2: the dangerous crash, inside the window. ──────────────────────────
print("")
print("STEP 2: second process dies AFTER 'enrich' takes effect but BEFORE it saves")
try:
    run("process 2", crash_after_effect="enrich")
    print("  did not crash, which this lab requires"); sys.exit(1)
except Crash as e:
    print("  [process 2] killed at %r" % str(e))
mid = load()
print("  state on disk : %s   <- does NOT know enrich happened" % mid["done"])
print("  world         : %s   <- but it DID happen" % WORLD)
assert "enrich" in WORLD and "enrich" not in mid["done"], "the window was not reproduced"

# ── STEP 3: resume. The tool's own guard is all that prevents a duplicate. ───
print("")
print("STEP 3: third process resumes, and retries 'enrich' because state says it is pending")
final = run("process 3")

# ── STEP 4: a full retry after success is a no-op. ───────────────────────────
print("")
print("STEP 4: fourth process runs the finished plan again")
again = run("process 4")

# ── STEP 5: the proof. ──────────────────────────────────────────────────────
print("")
print("STEP 5: lost work, or duplicated work?")
print("  steps completed       : %d of %d" % (len(final["done"]), len(PLAN)))
print("  tool calls attempted  : %d" % len(ATTEMPTS))
print("  effects in the world  : %d %s" % (len(WORLD), WORLD))
print("  enrich attempted      : %d times, applied %d time"
      % (ATTEMPTS.count("enrich"), WORLD.count("enrich")))
print("  duplicated effects    : %d" % (len(WORLD) - len(set(WORLD))))

ok = (final["done"] == PLAN
      and again["done"] == PLAN
      and WORLD == PLAN
      and len(WORLD) == len(set(WORLD))
      and ATTEMPTS.count("enrich") == 2
      and WORLD.count("enrich") == 1)
os.remove(STATE)
print("")
print("DURABLE LOOP RESUMED TWICE WITH NO LOST WORK AND NO DUPLICATED EFFECT: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Checkpoint after every step, key every tool on its own effect, and a crash costs one step.")
