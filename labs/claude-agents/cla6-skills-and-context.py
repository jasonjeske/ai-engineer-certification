#!/usr/bin/env python3
"""
LAB CLA6: Skills and context files. Capability and preference as files.

Two pieces of the harness are just files on disk, and they solve two different
problems people usually try to solve by growing one giant prompt.

A SKILL is a packaged, reusable unit of capability: a directory with a SKILL.md
carrying a short frontmatter block (a name and a description of when to use it)
plus the instructions, and optionally scripts and reference files beside it. Only
the name and description sit in context. The body loads when the description
matches what you actually asked for, which is why fifty installed skills do not
cost fifty skills' worth of context. That is the whole trick: the description is
the router's input, so a vague description means a skill that never fires.

A CONTEXT FILE is standing instruction. CLAUDE.md at the project root carries
what is true about this codebase for everyone, and a user-level CLAUDE.md in your
home config carries what is true about how you work, everywhere. The agent reads
them every session, so they are the place for durable facts and preferences, not
for task detail.

The split is worth stating plainly, because it is the design decision:

    context file  = always loaded, describes how to work here      (small, durable)
    skill         = loaded on demand, describes how to do one job  (bigger, specific)
    prompt        = this request only                              (volatile)

Externalizing into files beats one enormous prompt for three reasons you can
measure: the request only pays for the capability it needs, a skill is reusable
across sessions and people without being retyped, and a preference stated once in
a context file stops being re-litigated every session.

This lab defines two tiny skills as strings standing in for their SKILL.md files,
writes the router that picks between them on their trigger words, and proves it
routes both requests correctly, declines a request no skill covers, keeps the
context file present in every composed prompt, and pays for one skill body
instead of the whole library.

Run: python3 modules/academy-content/labs/claude-agents/cla6-skills-and-context.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# -- The context file: standing instruction, loaded on every single request. ---
CLAUDE_MD = """# CLAUDE.md (project context, read every session)
- Return only the label or the record, never a preamble.
- Never invent a fact the input does not contain.
"""

# -- Two skills. In a real install each is a SKILL.md plus optional scripts. ---
SKILLS = {
    "review-triage": {
        "description": "Use when labeling the mood of a customer review.",
        "triggers": {"review", "mood", "label", "customer"},
        "body": ("You label the sentiment of one customer review as positive, "
                 "negative or neutral. Read the whole review before deciding, and "
                 "weigh the last clause most, because that is where the verdict "
                 "usually sits."),
    },
    "contact-record": {
        "description": "Use when pulling a name, number or email out of loose text.",
        "triggers": {"name", "number", "email", "record", "pull"},
        "body": ("Extract JSON holding the name, the number and the email present "
                 "in the text. Omit any field the text does not contain rather "
                 "than guessing one."),
    },
}


def route(request):
    """The router. It reads only each skill's triggers and description, never the
    bodies, which is exactly what the harness does before it loads anything."""
    words = set(request.lower().replace(":", " ").split())
    best, best_score = None, 0
    for name, skill in SKILLS.items():
        score = len(words & skill["triggers"])
        if score > best_score:
            best, best_score = name, score
    return best


def compose(request, skill_name):
    """Build the prompt the model actually sees: the context file always, then the
    ONE routed skill body, then the request. Nothing else is loaded."""
    parts = [CLAUDE_MD]
    if skill_name:
        parts.append("# skill: %s\n%s" % (skill_name, SKILLS[skill_name]["body"]))
    parts.append(request)
    return "\n".join(parts)


def tokens(text):
    return len(text.split())


REQ_REVIEW = "label the mood of this customer review: I love this product"
REQ_CONTACT = "pull the name and number from this note: my name is Ada and the number is 42"
REQ_NEITHER = "restart the staging cluster"

print("STEP 1: what the router sees before it loads anything")
for name, skill in SKILLS.items():
    print("  %-15s %s" % (name, skill["description"]))

print("")
print("STEP 2: route each request")
picked_review = route(REQ_REVIEW)
picked_contact = route(REQ_CONTACT)
picked_neither = route(REQ_NEITHER)
print("  %r -> %r" % (REQ_REVIEW[:34] + "...", picked_review))
print("  %r -> %r" % (REQ_CONTACT[:34] + "...", picked_contact))
print("  %r -> %r" % (REQ_NEITHER, picked_neither))

print("")
print("STEP 3: run each routed request")
review_out = complete(compose(REQ_REVIEW, picked_review))
contact_out = complete(compose(REQ_CONTACT, picked_contact))
print("  review-triage  ->", repr(review_out))
print("  contact-record ->", repr(contact_out))

# -- What externalizing buys, in tokens. --------------------------------------
routed_prompt = compose(REQ_REVIEW, picked_review)
everything_prompt = "\n".join([CLAUDE_MD] + [s["body"] for s in SKILLS.values()] + [REQ_REVIEW])
print("")
print("STEP 4: context cost")
print("  routed prompt (one skill loaded) :", tokens(routed_prompt), "tokens")
print("  whole library stuffed in          :", tokens(everything_prompt), "tokens")

routed_right = (picked_review == "review-triage" and picked_contact == "contact-record")
declined = (picked_neither is None)
review_ok = (review_out == "positive")
record = json.loads(contact_out)
record_ok = (record.get("name") == "Ada" and record.get("number") == 42
             and "email" not in record)
context_always = all(CLAUDE_MD.splitlines()[0] in compose(r, route(r))
                     for r in (REQ_REVIEW, REQ_CONTACT, REQ_NEITHER))
only_one_body = (SKILLS["contact-record"]["body"] not in routed_prompt
                 and tokens(routed_prompt) < tokens(everything_prompt))

print("")
print("STEP 5: checks")
print("  both requests routed to the right skill :", routed_right)
print("  the uncovered request loaded no skill   :", declined)
print("  the review was labeled correctly        :", review_ok)
print("  the record held name and number only    :", record_ok)
print("  the context file rode along every time  :", context_always)
print("  only the routed skill body was loaded   :", only_one_body)

ok = routed_right and declined and review_ok and record_ok and context_always and only_one_body
print("")
print("THE ROUTER LOADED ONLY THE SKILL THE REQUEST NEEDED: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Capability in skills, preference in context files. Next: run a whole fleet.")
