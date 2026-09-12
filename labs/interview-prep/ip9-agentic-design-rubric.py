#!/usr/bin/env python3
"""
LAB IP9: Grade an agentic system-design answer against the interviewer's rubric.

THE PROMPT: "Design an agent that monitors a support inbox and drafts replies for
human approval."

Chapter 5 graded a design on sizing math. This round grades a different thing,
because an agent design is judged on the five concerns an interviewer is actually
listening for, and candidates lose the round by answering only the first one:

  1 tool contracts       which tools, with what arguments and what failure result
  2 permission model     what it may do alone, what needs a human, what is denied
  3 memory and state      what persists between turns, and what must not
  4 evaluation plan      trajectory grading AND outcome grading, on a golden set
  5 failure and tracing  how it fails, and what a trace shows you afterward
  6 cost and latency     a real number per run, not "it should be cheap"

This lab flips you into the interviewer's seat: a weighted rubric, two candidate
answers, and a score. The strong answer earns full marks. The "boxes and arrows"
answer names its tools and nothing else, which is exactly how a real candidate
fails this round: fluent, confident, and 0.20 out of 1.00.

Run: python3 modules/academy-content/labs/interview-prep/ip9-agentic-design-rubric.py
"""
import sys

# ── Candidate A: the answer that gets the offer ───────────────────────────────
STRONG_ANSWER = """
Tools, with contracts: list_unread(since) returns message ids; get_message(id)
returns sender, subject, body; search_kb(query) returns the top 3 articles with
ids; draft_reply(id, text) writes a DRAFT only and returns a draft id. Every tool
returns a typed error rather than raising, so the loop can observe and retry once.

Permission model: the agent reads mail and writes drafts autonomously. Sending is
denied outright, it has no send tool at all, so no injected instruction can reach
one. Any refund or account change is a human-in-the-loop approval gate keyed on an
out-of-band token, never on a claim in the message text, because message bodies are
untrusted content.

Memory and state: per-thread state holds the message ids already handled so a
restart does not double-draft, keyed by thread id in a small store. Conversation
history is summarized to a 400 token thread digest. Nothing customer-identifying
persists past 30 days.

Evaluation: a golden set of 120 real tickets with approved replies. Outcome
grading scores draft quality with a rubric judge plus a deterministic check that
every claim cites a KB article id. Trajectory grading checks it called search_kb
before draft_reply and never exceeded 6 steps. The gate blocks a deploy on any
regression.

Failure modes and observability: the top failures are a hallucinated policy claim,
a loop that never terminates, and a prompt injection in a customer message aimed
at the draft. Mitigations are a hard step cap of 6, citation-required output, and
treating message bodies as data. Every run emits a trace of steps, tool calls,
tokens, and latency, so a bad draft can be replayed step by step.

Cost and latency: about 3000 input and 400 output tokens per ticket, so roughly
0.015 dollars per ticket and 12 dollars a day at 800 tickets, with a p95 of about
9 seconds, which is fine because a human reviews the draft anyway.
"""

# ── Candidate B: fluent, confident, and unhireable ────────────────────────────
WEAK_ANSWER = """
I would connect the inbox to the model. The agent uses list_unread for the ids
and get_message, which returns the sender and the body, then draft_reply to write
the response. It loops until the queue is empty. I would use a good prompt to
make the replies helpful and on brand, and I would iterate on the prompt until
the quality is high. We could add a vector database for the knowledge base. It
should be fast and cheap, and I would make sure it is safe and reliable.
"""


# ── The rubric ────────────────────────────────────────────────────────────────
def has_all(text, *terms):
    """A concern only counts as ADDRESSED if every required signal is present.
    One buzzword is not a design."""
    low = text.lower()
    return all(t in low for t in terms)


def has_number(text, *terms):
    """Cost and latency claims must carry a digit IN THE CLAIM. Scanning the
    whole answer for any digit would pass 'it should be cheap' in an answer that
    happens to mention 120 tickets elsewhere, so the digit has to sit next to the
    claim it is supposed to quantify."""
    low = text.lower()
    for t in terms:
        i = low.find(t)
        if i >= 0 and any(c.isdigit() for c in low[max(0, i - 60):i + 60]):
            return True
    return False


def build_rubric():
    """Weights sum to 1.0 so the score reads as a clean fraction."""
    return [
        ("tool contracts", lambda a: has_all(a, "returns", "get_message"), 0.20),
        ("permission model", lambda a: has_all(a, "approval", "denied"), 0.20),
        ("memory and state", lambda a: has_all(a, "state", "restart"), 0.15),
        ("evaluation plan", lambda a: has_all(a, "golden set", "trajectory", "outcome"), 0.20),
        ("failure and tracing", lambda a: has_all(a, "failure", "trace", "step cap"), 0.15),
        ("cost and latency", lambda a: has_number(a, "dollars per", "p95"), 0.10),
    ]


def grade(answer, rubric):
    """Run every check, return (score, per-check results)."""
    results, score = [], 0.0
    for name, check, weight in rubric:
        try:
            passed = bool(check(answer))
        except Exception:
            passed = False
        if passed:
            score += weight
        results.append((name, passed, weight))
    return round(score, 4), results


def report(label, score, results):
    print("%s" % label)
    for name, passed, weight in results:
        print("  [%s] %-20s (weight %.2f)" % ("x" if passed else " ", name, weight))
    print("  score = %.2f" % score)


rubric = build_rubric()
strong_score, strong_results = grade(STRONG_ANSWER, rubric)
weak_score, weak_results = grade(WEAK_ANSWER, rubric)

print("STEP 1: grade the strong answer")
report("", strong_score, strong_results)

print("\nSTEP 2: grade the boxes-and-arrows answer")
report("", weak_score, weak_results)

weak_failed = {name for name, passed, _ in weak_results if not passed}
expected_failures = {"permission model", "memory and state", "evaluation plan",
                     "failure and tracing", "cost and latency"}

strong_full = (strong_score == 1.0)
weak_lower = (weak_score < strong_score)
weak_expected = (weak_failed == expected_failures)
deterministic = (grade(STRONG_ANSWER, rubric)[0] == strong_score)

print("\nSTEP 3: what the weak answer actually missed")
for name in sorted(weak_failed):
    print("  missing: %s" % name)

print("")
print("        strong answer scores full marks : %s" % strong_full)
print("        weak answer scores lower        : %s (%.2f < %.2f)"
      % (weak_lower, weak_score, strong_score))
print("        weak fails the right concerns   : %s" % weak_expected)
print("        rubric is deterministic         : %s" % deterministic)

ok = strong_full and weak_lower and weak_expected and deterministic
print("")
print("AGENTIC DESIGN RUBRIC SCORED BOTH ANSWERS CORRECTLY: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Answer all six concerns out loud. Naming the tools is one sixth of the round.")
