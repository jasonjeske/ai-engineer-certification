#!/usr/bin/env python3
"""
LAB AE6: The context budget, and compaction that does not lose the fact.

Every tool result gets appended to the transcript, so the context window is a
resource the loop SPENDS, turn by turn, until the loop dies. The naive fix is to
drop the oldest turns, and it silently throws away the one fact the last turn
needed. The real fix is two layers: pin durable facts OUTSIDE the transcript (a
file on disk), then compact the episodic part. Because the fact lives in a file,
a fresh agent with an empty transcript can also RESUME the job, which is what
checkpointing buys you. You prove all four claims in one run.

Run: python3 modules/academy-content/labs/agentic-engineering/ae6-context-budget.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete
import tempfile, re

BUDGET = 60          # toy context window, counted in whitespace tokens
FACT = "deploy key is K-4417"
CHECKPOINT = os.path.join(tempfile.gettempdir(), "ae6_checkpoint.txt")


def tokens(lines):
    return sum(len(l.split()) for l in lines)


def turn_text(n):
    """A verbose tool result, the thing that actually eats the window."""
    return ("turn %d tool result: scanned 12 files in build/ and found no errors "
            "warnings none elapsed 4s cache hit ratio 0.81 exit 0" % n)


# --- The transcript grows. Turn 1 carries the load-bearing fact.
transcript = ["system: you are a deploy agent", "user: " + FACT]
print("STEP 1: grow the transcript and watch it cross the budget")
for n in range(1, 6):
    transcript.append(turn_text(n))
    print(f"  after turn {n}: {tokens(transcript)}/{BUDGET} tokens")
over_budget = tokens(transcript) > BUDGET
print(f"  over budget: {over_budget}")

# --- NAIVE FIX: keep the newest turns only. The fact from turn 1 is gone.
print("")
print("STEP 2: naive truncation, keep the newest lines until it fits")
naive = list(transcript)
while tokens(naive) > BUDGET:
    naive.pop(0)
naive_lost_fact = not any("K-4417" in l for l in naive)
print(f"  {tokens(naive)}/{BUDGET} tokens, fact survived: {not naive_lost_fact}")

# --- REAL FIX: pin the durable fact to disk, then compact the episodic part.
print("")
print("STEP 3: pin durable facts to a file, then compact the rest")
with open(CHECKPOINT, "w") as f:            # files-as-memory: outside the window
    f.write(FACT + "\n")
head, tail = transcript[:2], transcript[-1:]
middle = " ".join(transcript[2:-1])
# Our mock summarizer keeps the most frequent words; a real model writes a
# sentence. Both are LOSSY in the same place, which is exactly why the fact was
# pinned to disk first rather than trusted to survive a summary.
summary = "summary of %d earlier turns: %s" % (len(transcript) - 3, complete(middle))
compacted = ["system: you are a deploy agent", summary] + tail
print(f"  compacted to {tokens(compacted)}/{BUDGET} tokens")
print(f"  summary line: {summary}")
within_budget = tokens(compacted) <= BUDGET

# --- CHECKPOINT AND RESUME: a fresh agent, empty transcript, reads the file.
print("")
print("STEP 4: resume in a fresh context that never saw turn 1")
resumed = ["system: you are a deploy agent"]      # no history at all
with open(CHECKPOINT) as f:
    durable = f.read().strip()
resumed.append("durable memory: " + durable)
answer = complete("Context: %s Question: what is the deploy key?" % durable)
key = re.search(r"K-\d+", answer)
kept_fact = bool(key) and key.group(0) == "K-4417"
print(f"  recovered answer: {answer!r} -> key {key.group(0) if key else None}")
os.remove(CHECKPOINT)

print("")
print(f"transcript exceeded the budget        : {over_budget}")
print(f"naive truncation lost the fact        : {naive_lost_fact}")
print(f"compaction fit inside the budget      : {within_budget}")
print(f"fresh context recovered the fact      : {kept_fact}")

ok = over_budget and naive_lost_fact and within_budget and kept_fact
print("")
print(f"COMPACTION STAYED IN BUDGET WITHOUT LOSING THE FACT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("RAG is one more layer on this shelf, not the shelf. Next: permissions.")
