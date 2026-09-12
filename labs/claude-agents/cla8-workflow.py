#!/usr/bin/env python3
"""
LAB CLA8: A complete agentic workflow.

The capstone for this course. One agent, real multi-step work, every mechanic you
built showing up once and doing its job.

The task: read a sales note, pull the deal size out of it, project double for the
forecast, get an independent read on the client's mood, and file a record, while
a guard blocks an action nobody granted.

What it composes:
  - the tier decision first, before any code (this lab, and lab CLA1's shape)
  - a skill routed by its description, not a prompt that swallowed everything (CLA6)
  - tools called under a PreToolUse guard, least privilege (CLA5)
  - one focused piece delegated to a subagent with its own context (CLA5)
  - structured output validated against a schema before it is trusted (CLA2)
  - a verify step before anything is reported (CLA3, CLA7)

The tier decision deserves its own beat, because it is the judgment call that
separates an engineer from a demo. A bounded classify-and-return job wants a
plain Messages client: one call, one shape, nothing to host. A job that has to
read and edit files, run commands and search the web on its own wants the Claude
Agent SDK, which ships that harness and those built-in tools. Reaching for the
agent tier when a single call would do is the most expensive habit in this field,
and the reverse, hand-rolling a filesystem harness that already exists, is the
second. Pick the smallest tier that actually does the job.

Nothing below is new. That is the point: a production agent is not one enormous
clever prompt, it is a small set of mechanics composed cleanly, and you can now
name every one of them.

Run: python3 modules/academy-content/labs/claude-agents/cla8-workflow.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

NOTE = ("Deal with Acme closed. My name is Jordan and the number is 40. "
        "Great client, excellent to work with.")
RECORD_SCHEMA = {"deal": int, "projection": int, "mood": str}


# --- 0. The tier decision, made before any code gets written. -----------------
def choose_tier(needs_files, needs_bash, needs_web):
    """A bounded call-and-return job wants a Messages client. Autonomous file,
    shell or web work wants the Claude Agent SDK harness."""
    return "claude-agent-sdk" if (needs_files or needs_bash or needs_web) else "messages-api"


tier = choose_tier(needs_files=False, needs_bash=False, needs_web=False)
other_tier = choose_tier(needs_files=True, needs_bash=True, needs_web=False)
print("STEP 0: pick the tier")
print("  this job (bounded, no file or shell work) ->", tier)
print("  a refactor-the-repo job                   ->", other_tier)

# --- 1. Route the skill by its description. ----------------------------------
SKILLS = {
    "deal-desk": {"description": "Use when reading a sales note for deal facts.",
                  "triggers": {"deal", "sales", "note", "forecast"},
                  "body": "Extract JSON with the name and the number in the note."},
    "release-notes": {"description": "Use when writing release notes from commits.",
                      "triggers": {"release", "commits", "changelog"},
                      "body": "Summarize the commits into release notes."},
}
request = "read this sales note and forecast the deal"
words = set(request.lower().split())
skill = max(SKILLS, key=lambda n: len(words & SKILLS[n]["triggers"]))
print("")
print("STEP 1: routed skill ->", skill)

# --- 2. Tools behind a PreToolUse guard (least privilege). -------------------
GRANTED = {"extract", "calculator"}          # send_email is deliberately not granted
guard_log = []


def extract(text):
    return complete("%s\n%s" % (SKILLS[skill]["body"], text))


TOOLS = {"extract": extract, "calculator": lambda n: n * 2,
         "send_email": lambda body: "sent: %s" % body}


def guarded(tool, arg):
    if tool not in GRANTED:
        guard_log.append(("BLOCKED", tool))
        return {"ran": False, "blocked": True}
    guard_log.append(("RAN", tool))
    return {"ran": True, "result": TOOLS[tool](arg)}


print("")
print("STEP 2: act under the guard")
raw = guarded("extract", NOTE)["result"]
facts = json.loads(raw)                      # wrapped by the schema check below
deal = facts["number"]
print("  extract    -> %s" % raw)
projection = guarded("calculator", deal)["result"]
print("  calculator -> %d x 2 = %d" % (deal, projection))

# --- 3. Delegate the mood read to a subagent with its own context. -----------
MOOD_AGENT = {"prompt": "You label the mood of a note.", "model": "claude-haiku-4-5"}
mood = complete("system: %s\nuser: sentiment: %s" % (MOOD_AGENT["prompt"], NOTE))
print("")
print("STEP 3: subagent on %s -> %r" % (MOOD_AGENT["model"], mood))

# --- 4. The agent reaches for something it was never granted. ----------------
blocked = guarded("send_email", "forecast is %d" % projection)
print("")
print("STEP 4: ungranted send_email ->", blocked)

# --- 5. Validate the record, then verify, then report. ----------------------
record = {"deal": deal, "projection": projection, "mood": mood}
schema_ok = all(f in record and isinstance(record[f], t) for f, t in RECORD_SCHEMA.items())
print("")
print("STEP 5: the record")
print("  %s" % json.dumps(record, sort_keys=True))
print("  matches the schema :", schema_ok)

tier_ok = (tier == "messages-api" and other_tier == "claude-agent-sdk")
skill_ok = (skill == "deal-desk")
correct = (record == {"deal": 40, "projection": 80, "mood": "positive"})
guard_ok = (blocked.get("blocked") is True
            and sum(1 for entry in guard_log if entry[0] == "RAN") == 2)

print("")
print("STEP 6: checks")
print("  the tier decision went both ways correctly :", tier_ok)
print("  the right skill was routed in               :", skill_ok)
print("  the record is correct and schema-valid      :", correct and schema_ok)
print("  exactly the granted tools ran, the rest blocked :", guard_ok)

ok = tier_ok and skill_ok and correct and schema_ok and guard_ok
print("")
print("COMPLETE CLAUDE WORKFLOW PRODUCED THE VERIFIED RECORD: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Pick the tier, route the skill, act under a guard, delegate, verify, report.")
