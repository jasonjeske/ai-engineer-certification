#!/usr/bin/env python3
"""
LAB AE7: Three capability tiers, an approval gate, and an audit trail.

Every tool an agent holds sits in one of three tiers: ALLOW (act freely), ASK
(pause for a human), NEVER (refuse, no question asked), chosen by BLAST RADIUS,
which is what the action costs when it is wrong. The gate goes BEFORE the side
effect, so a run stopped at a gate leaves nothing half done. Every decision
lands in an audit list written by the harness and not by the model, so it is
evidence rather than narration. Here one delete is attempted twice, denied once
and approved once, against real files in a temp directory.

Run: python3 modules/academy-content/labs/agentic-engineering/ae7-approval-gate.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import tempfile, glob

TIERS = {"search": "ALLOW",             # read only, blast radius: nothing
         "delete_file": "ASK",          # irreversible, blast radius: one file
         "wipe_workspace": "NEVER"}     # blast radius: everything, so never
AUDIT = []                              # harness-owned log, not model narration

workdir = tempfile.mkdtemp(prefix="ae7_")
for name in ("cache.tmp", "keep.txt"):
    open(os.path.join(workdir, name), "w").write("data\n")


def search(**kw):
    return "3 notes matched"


def delete_file(path):
    os.remove(path)
    return "deleted " + os.path.basename(path)


def wipe_workspace(**kw):
    raise AssertionError("NEVER-tier tool must never be reachable")


TOOLS = {"search": search, "delete_file": delete_file,
         "wipe_workspace": wipe_workspace}


def gated_call(tool, approver, **args):
    """The whole mechanism: classify, gate, log, only then execute."""
    tier = TIERS[tool]
    decision = {"ALLOW": "auto-allowed", "NEVER": "refused-by-policy"}.get(tier)
    if tier == "ASK":
        decision = "approved" if approver(tool, args) else "denied"
    executed = decision in ("auto-allowed", "approved")
    result = TOOLS[tool](**args) if executed else None
    AUDIT.append({"step": len(AUDIT) + 1, "tool": tool, "tier": tier,
                  "args": args, "decision": decision, "executed": executed})
    print(f"  {tool} [{tier}] -> {decision}, executed={executed}, result={result!r}")
    return result


answers = iter([False, True])           # the human says no, then yes
def approver(tool, args):
    return next(answers)


target = os.path.join(workdir, "cache.tmp")
print("STEP 1: an ALLOW-tier read runs with no gate")
picked = tool_route("search the notes for the vendor list", list(TOOLS))
gated_call(picked, approver)

print("")
print("STEP 2: the same delete, twice, with the human answering no then yes")
gated_call("delete_file", approver, path=target)
blocked_survived = os.path.exists(target)
print(f"  file still present after the denial: {blocked_survived}")
gated_call("delete_file", approver, path=target)
approved_removed = not os.path.exists(target)

print("")
print("STEP 3: a NEVER-tier tool is refused without asking anyone")
gated_call("wipe_workspace", approver)
never_refused = AUDIT[-1]["decision"] == "refused-by-policy"
survivors = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(workdir, "*")))]
print(f"  workspace survivors: {survivors}")

print("")
print("STEP 4: the audit trail, one row per decision")
for row in AUDIT:
    print(f"  {row['step']}. {row['tool']:<15} {row['tier']:<6} {row['decision']}")
audit_complete = len(AUDIT) == 4 and all(r["decision"] for r in AUDIT)
# Interruptible by construction: the gate precedes every side effect, so
# stopping at any row above leaves no partially applied action behind.
clean_to_interrupt = survivors == ["keep.txt"]
for p in glob.glob(os.path.join(workdir, "*")):
    os.remove(p)
os.rmdir(workdir)

print("")
print(f"denied delete did not execute      : {blocked_survived}")
print(f"approved delete did execute        : {approved_removed}")
print(f"NEVER tier refused without asking  : {never_refused}")
print(f"every decision logged (4 rows)     : {audit_complete}")
print(f"no half-applied state at any pause : {clean_to_interrupt}")

ok = all([blocked_survived, approved_removed, never_refused, audit_complete, clean_to_interrupt])
print("")
print(f"APPROVAL GATE BLOCKED THE UNAPPROVED DELETE AND LOGGED EVERY DECISION: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Least privilege is a table plus a gate. Next: the protocol that carries tools.")
