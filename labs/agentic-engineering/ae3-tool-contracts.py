#!/usr/bin/env python3
"""
LAB AE3: Tools as contracts.

A tool's JSON Schema is a contract, and its description is a prompt the model
reads on every turn. Three properties separate a toy tool from a production one.
It is idempotent, so the retry after a timeout does not double-charge anyone. Its
errors tell the model how to recover instead of dumping a stack trace. And it is
exposed only when relevant, because a registry of eight tools routes worse than a
scoped registry of two. This lab proves all three with numbers.

Run: python3 modules/academy-content/labs/agentic-engineering/ae3-tool-contracts.py
"""
import sys, os, difflib
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route

CITIES = {"paris": 22, "tokyo": 30, "london": 15}

# ── The contract. Real JSON Schema shape: typed properties, required list. ────
WEATHER = {
    "name": "weather",
    "description": "Current temperature in Celsius for one city. Read-only, safe to call twice.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name, lowercase"}},
        "required": ["city"],
    },
}

def validate(schema, args):
    """The contract enforced in code: reject a bad call before the tool runs."""
    props = schema["input_schema"]["properties"]
    for key in schema["input_schema"]["required"]:
        if key not in args:
            return "ERROR: missing required field %r. schema wants: %s" % (key, sorted(props))
    return None

def weather_opaque(city):
    return "KeyError: %r" % city.lower()            # what a stack trace gives the model

def weather_recoverable(city):
    key = city.strip().lower()
    if key in CITIES:
        return CITIES[key]
    # An error the model can ACT on: what was wrong, and what the valid values are.
    return "ERROR: unknown city %r. known cities: %s" % (city, ", ".join(sorted(CITIES)))

def agent_retry(message):
    """A one-step recovery: read the error, pick a listed value, call again."""
    if "known cities:" not in message:
        return None                                  # nothing actionable in the text
    listed = [c.strip() for c in message.split("known cities:")[1].split(",")]
    bad = message.split("'")[1]
    fix = difflib.get_close_matches(bad.lower(), listed, n=1)
    return weather_recoverable(fix[0]) if fix else None

print("1. SCHEMA VALIDATION")
print("   bad call  -> %s" % validate(WEATHER, {"town": "paris"}))
print("   good call -> %r" % weather_recoverable("paris"))
c1 = validate(WEATHER, {"town": "paris"}) is not None and validate(WEATHER, {"city": "paris"}) is None

print("")
print("2. IDEMPOTENCY (the agent retries, so every tool gets called twice)")
ledger_bad = []
ledger_good = set()
for _ in range(2):
    ledger_bad.append("ada@example.com")             # append: drifts on retry
    ledger_good.add("ada@example.com")               # keyed: same result twice
print("   non-idempotent ledger after 2 calls: %d entries" % len(ledger_bad))
print("   idempotent ledger after 2 calls    : %d entries" % len(ledger_good))
c2 = len(ledger_bad) == 2 and len(ledger_good) == 1

print("")
print("3. ERROR MESSAGES THE MODEL CAN RECOVER FROM")
opaque = weather_opaque("Pariss")
good = weather_recoverable("Pariss")
print("   opaque      : %s  -> retry got %r" % (opaque, agent_retry(opaque)))
print("   recoverable : %s" % good)
print("   recoverable -> retry got %r" % agent_retry(good))
c3 = agent_retry(opaque) is None and agent_retry(good) == 22

print("")
print("4. TOOL-COUNT BUDGET AND PROGRESSIVE DISCLOSURE")
ask = "what is the average temperature in Paris this week"
wide = ["calculator", "calendar", "search", "weather", "email", "files"]
scoped = ["weather", "search"]                       # only what this step needs
print("   ask: %r" % ask)
print("   6 tools exposed -> %r" % tool_route(ask, wide))
print("   2 tools exposed -> %r" % tool_route(ask, scoped))
c4 = tool_route(ask, wide) != "weather" and tool_route(ask, scoped) == "weather"

ok = c1 and c2 and c3 and c4
print("")
print("ALL FOUR TOOL CONTRACTS HELD: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Schema, idempotency, recoverable errors, scoping. Next: the environment.")
