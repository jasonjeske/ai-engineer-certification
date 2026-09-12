#!/usr/bin/env python3
"""
LAB AE11: What a loop costs. Traces as trees, per-step attribution, kill switch.

A loop's cost is not linear in its steps, it is roughly quadratic in them,
because every turn resends the whole transcript. Step 1 pays for one message and
step 8 pays for eight. That is why a loop that "only" ran twice as long costs
four times as much, and why the single most effective cost lever is not a cheaper
model, it is a shorter transcript.

This lab builds a real trace TREE, attributes tokens and dollars per step, prices
the same run three ways (naive, cached prefix, routed models), and arms a kill
switch that stops the loop before the ceiling instead of after.

Run: python3 modules/academy-content/labs/agentic-engineering/ae11-trace-cost-killswitch.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

# Per-million-token prices. PLAN is the expensive reasoning model, EXEC is the
# cheap one. Cached input is the same tokens at a tenth the price.
PRICE = {"plan": {"in": 15.00, "out": 75.00}, "exec": {"in": 0.80, "out": 4.00}}
CACHE_DISCOUNT = 0.10
SYSTEM_TOKENS = 900        # the fixed prefix: system prompt plus tool schemas
TURN_TOKENS = 600          # what each completed turn adds (model text plus tool result)
OUT_TOKENS = 180           # tokens the model writes per step

USD_CEILING = 0.40         # hard stop, in dollars
STEP_CEILING = 12          # hard stop, in steps


def price(model, tok_in, tok_out, cached_in=0):
    """Cost of one call. Cached input tokens bill at a tenth, so a long stable
    prefix stops being the thing that hurts."""
    p = PRICE[model]
    fresh = tok_in - cached_in
    return (fresh * p["in"] + cached_in * p["in"] * CACHE_DISCOUNT + tok_out * p["out"]) / 1e6


# ── STEP 1: the trace is a TREE, not a flat log. ──────────────────────────────
# One agent turn branches: the model call, then the tool calls it decided on,
# and a sub-agent that has its own children. A flat log cannot show you that the
# 0.41s tool call is what made the 2.10s turn slow.
TRACE = ("turn 2", 2100, [
    ("model call (plan)", 1400, []),
    ("tool: search", 410, [("http GET", 380, [])]),
    ("sub-agent: summarize", 260, [("model call (exec)", 240, [])]),
])


def show(node, depth=0):
    name, ms, kids = node
    bar = "#" * max(1, round(ms / 100))
    print("  %s%-22s %5d ms %s" % ("  " * depth, name, ms, bar))
    for kid in kids:
        show(kid, depth + 1)


print("STEP 1: one turn of the trace, as a tree")
show(TRACE)
_, root_ms, children = TRACE
child_ms = sum(ms for _, ms, _ in children)
print("  self time in the turn itself : %d ms of %d" % (root_ms - child_ms, root_ms))
print("  slowest child                : %s" % max(children, key=lambda c: c[1])[0])

# ── STEP 2: per-step cost, growing with the transcript. ──────────────────────
print("")
print("STEP 2: per-step cost of a naive loop (full transcript resent every turn)")
print("  step  prompt_tok  out_tok      usd   cum_usd")
naive_total = 0.0
rows = []
for step in range(1, 11):
    tok_in = SYSTEM_TOKENS + TURN_TOKENS * (step - 1)
    usd = price("plan", tok_in, OUT_TOKENS)
    naive_total += usd
    rows.append((step, tok_in, usd))
    print("  %4d  %10d  %7d  %7.4f  %8.4f" % (step, tok_in, OUT_TOKENS, usd, naive_total))

first, last = rows[0][2], rows[-1][2]
print("  last step costs %.1fx the first, on identical work" % (last / first))
print("  most expensive step          : step %d" % max(rows, key=lambda r: r[2])[0])

# ── STEP 3: the same ten steps, two levers applied. ──────────────────────────
print("")
print("STEP 3: same ten steps, priced three ways")
cached_total = 0.0
routed_total = 0.0
for step in range(1, 11):
    tok_in = SYSTEM_TOKENS + TURN_TOKENS * (step - 1)
    # Lever 1: the stable prefix is cached, so only the new turn is fresh input.
    cached_total += price("plan", tok_in, OUT_TOKENS, cached_in=SYSTEM_TOKENS)
    # Lever 2: plan once on the expensive model, execute the rest on the cheap one.
    model = "plan" if step == 1 else "exec"
    routed_total += price(model, tok_in, OUT_TOKENS, cached_in=SYSTEM_TOKENS)
print("  naive                        : $%.4f" % naive_total)
print("  cached prefix                : $%.4f  (-%.0f%%)"
      % (cached_total, 100 * (1 - cached_total / naive_total)))
print("  cached + model routing       : $%.4f  (-%.0f%%)"
      % (routed_total, 100 * (1 - routed_total / naive_total)))

# ── STEP 4: the kill switch. Checked BEFORE the call, not after. ─────────────
print("")
print("STEP 4: kill switch armed at $%.2f / %d steps" % (USD_CEILING, STEP_CEILING))
spent = 0.0
step = 0
halted = None
while True:
    step += 1
    tok_in = SYSTEM_TOKENS + TURN_TOKENS * (step - 1)
    projected = spent + price("plan", tok_in, OUT_TOKENS)
    # The check that matters: would THIS call breach the ceiling? A budget check
    # that runs after the call is a receipt, not a limit. The money is gone.
    if projected > USD_CEILING:
        halted = "cost"
        print("  step %2d refused: would reach $%.4f, over the ceiling" % (step, projected))
        break
    if step > STEP_CEILING:
        halted = "steps"
        print("  step %2d refused: over the step ceiling" % step)
        break
    spent = projected
    print("  step %2d ran    : spent $%.4f" % (step, spent))

print("")
print("  halted by      : %s ceiling" % halted)
print("  final spend    : $%.4f (never exceeded $%.2f)" % (spent, USD_CEILING))

ok = (halted == "cost"
      and spent <= USD_CEILING
      and last > first * 2
      and routed_total < cached_total < naive_total
      and (root_ms - child_ms) > 0)
print("")
print("COST GREW WITH THE TRANSCRIPT AND THE KILL SWITCH STOPPED IT UNDER BUDGET: %s"
      % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Shorten the transcript, cache the prefix, route the cheap steps, and cap the run.")
