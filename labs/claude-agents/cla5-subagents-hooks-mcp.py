#!/usr/bin/env python3
"""
LAB CLA5: Subagents, hooks and MCP, the three extension points of the harness.

Claude Code implements three ideas the flagship course teaches in the abstract.
Here is how each one actually attaches to the harness.

DELEGATION. A subagent is a delegated agent with its OWN context window, its own
prompt, its own tool grant, and optionally its own model. You define one and the
main agent invokes it through the Agent tool, so Agent has to be in the main
agent's allowed tools for delegation to be possible at all:

    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Agent"],
        agents={"reviewer": AgentDefinition(
            description="Classify the sentiment of one review.",
            prompt="You classify sentiment.", tools=["Read"],
            model="claude-haiku-4-5")})

GUARDRAILS. Hooks run your code at fixed points in the lifecycle: PreToolUse,
PostToolUse, Stop, SessionStart, SessionEnd, UserPromptSubmit. You attach them
with a matcher so a hook only fires for the tools you name:

    options=ClaudeAgentOptions(hooks={
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[my_gate])]})

PreToolUse is the one that matters for safety, because it runs BEFORE the tool
and returns allow, deny, or ask. Two policies ride on it: least privilege (an
allowlist, everything else denied) and human in the loop (an irreversible action
is refused until a person approves it). PostToolUse is where the audit trail gets
written, after the fact.

TOOLS FROM OUTSIDE. An MCP server publishes tools behind one interface and the
harness lists them at startup and calls them by name, so a capability becomes a
plug-in with no code change:

    options=ClaudeAgentOptions(mcp_servers={
        "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}})

The discovered tools land under the same permission system as the built-ins,
which is the part worth internalizing: discovery does not imply trust.

This lab wires all three at once. An agent with no hardcoded external tools
discovers an MCP server, calls a discovered tool through a PreToolUse gate, gets
refused on a tool it was never granted, gets held at an approval gate on a
destructive one, and delegates two reviews to isolated subagents.

Run: python3 modules/academy-content/labs/claude-agents/cla5-subagents-hooks-mcp.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


class MCPServer:
    """A minimal MCP-style server: it owns tools and exposes two protocol
    operations. The agent knows none of these names ahead of time."""
    def __init__(self):
        self._tools = {
            "mcp__units__convert": {"description": "Convert miles to kilometers.",
                                    "fn": lambda arg: round(float(arg) * 1.60934, 3)},
            "mcp__docs__lookup": {"description": "Look up a capital city.",
                                  "fn": lambda arg: {"france": "Paris"}[arg.lower()]},
        }

    def list_tools(self):
        return [{"name": n, "description": t["description"]} for n, t in self._tools.items()]

    def call_tool(self, name, arg):
        return self._tools[name]["fn"](arg)


# -- The harness: built-in tools, a discovered MCP server, and the hook layer. -
BUILT_INS = {"Read": lambda arg: "the release is excellent",
             "Write": lambda arg: "wrote %s" % arg,
             "Bash": lambda arg: "ran %s" % arg}
GRANTED = {"Read", "Write", "Agent"}        # Bash is deliberately not granted
SENSITIVE = {"Write"}                        # irreversible: needs a human go
audit = []                                   # PostToolUse writes here


def pre_tool_use(tool, approved, discovered):
    """PreToolUse hook. Discovered MCP tools are allowed only because they were
    added to the grant at connect time; discovery alone is not permission."""
    if tool not in GRANTED and tool not in discovered:
        return {"decision": "deny", "reason": "not granted"}
    if tool in SENSITIVE and not approved:
        return {"decision": "ask", "reason": "irreversible action needs approval"}
    return {"decision": "allow"}


def post_tool_use(tool, result):
    audit.append((tool, result))


class Agent:
    """Hardcodes no external tools. It learns them from the server at runtime, and
    every call it makes passes the gate first."""
    def __init__(self):
        self.discovered = []

    def connect(self, server):
        self.discovered = [t["name"] for t in server.list_tools()]

    def call(self, tool, arg, server=None, approved=False):
        decision = pre_tool_use(tool, approved, self.discovered)
        if decision["decision"] != "allow":
            return {"ran": False, **decision}
        result = server.call_tool(tool, arg) if tool in self.discovered else BUILT_INS[tool](arg)
        post_tool_use(tool, result)
        return {"ran": True, "decision": "allow", "result": result}


def spawn_subagent(defn, task):
    """One subagent, one isolated context. It receives ONLY this task, builds its
    own transcript, and returns its result plus that transcript so isolation is
    checkable rather than assumed."""
    transcript = ["system: " + defn["prompt"], "user: sentiment of this review: " + task]
    label = complete("\n".join(transcript))
    transcript.append("assistant: " + label)
    return {"result": label, "context": transcript, "model": defn["model"]}


server, agent = MCPServer(), Agent()
print("STEP 1: MCP discovery")
print("  hardcoded external tools :", [])
agent.connect(server)
print("  discovered over MCP      :", agent.discovered)

print("")
print("STEP 2: every call goes through the PreToolUse gate")
mcp_call = agent.call("mcp__docs__lookup", "France", server=server)
print("  mcp__docs__lookup('France') ->", mcp_call)
denied = agent.call("Bash", "rm -rf build", server=server)
print("  Bash(...)  [not granted]    ->", denied)
asked = agent.call("Write", "report.md", server=server)
print("  Write(...) [no approval]    ->", asked)
approved = agent.call("Write", "report.md", server=server, approved=True)
print("  Write(...) [approved]       ->", approved)

REVIEWER = {"description": "Classify the sentiment of one review.",
            "prompt": "You are a sentiment classifier. Read one review and label it.",
            "tools": ["Read"], "model": "claude-haiku-4-5"}
REVIEWS = ["I love this product, it is excellent",
           "this is terrible and broken, worst ever"]

print("")
print("STEP 3: delegate one review per subagent")
outs = [spawn_subagent(REVIEWER, r) for r in REVIEWS]
for i, out in enumerate(outs, 1):
    print("  subagent %d on %s -> %r" % (i, out["model"], out["result"]))

# Isolation: subagent i's context holds its own review and none of the others.
isolation_ok = True
for i, out in enumerate(outs):
    blob = "\n".join(out["context"])
    if REVIEWS[i] not in blob:
        isolation_ok = False
    for j, other in enumerate(REVIEWS):
        if j != i and other in blob:
            isolation_ok = False

used_discovered = (mcp_call["ran"] and mcp_call["result"] == "Paris"
                   and "mcp__docs__lookup" in agent.discovered)
blocked = (denied["ran"] is False and denied["decision"] == "deny")
gated = (asked["ran"] is False and asked["decision"] == "ask")
approved_ran = (approved["ran"] and approved["result"] == "wrote report.md")
labels_ok = ([o["result"] for o in outs] == ["positive", "negative"])
audit_clean = (audit == [("mcp__docs__lookup", "Paris"), ("Write", "wrote report.md")])

print("")
print("STEP 4: checks")
print("  called a tool it never hardcoded        :", used_discovered)
print("  ungranted Bash denied at the gate       :", blocked)
print("  irreversible Write held for approval    :", gated)
print("  and ran once a human approved it        :", approved_ran)
print("  each subagent saw only its own review   :", isolation_ok)
print("  both subagents labeled correctly        :", labels_ok)
print("  the audit trail lists only what ran     :", audit_clean)

ok = (used_discovered and blocked and gated and approved_ran and isolation_ok
      and labels_ok and audit_clean)
print("")
print("DELEGATION, HOOKS, AND MCP ALL HELD IN ONE RUN: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Delegate, gate, discover. Next: package capability into skills and context files.")
