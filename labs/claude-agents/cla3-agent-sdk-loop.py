#!/usr/bin/env python3
"""
LAB CLA3: The loop, by hand and by tool runner.

One tool call is not an agent. The agent is the LOOP, and in Claude's API that
loop has one exact condition: keep going while stop_reason is "tool_use".

    while resp.stop_reason == "tool_use":
        for block in resp.content:
            if block.type == "tool_use":
                out = run_tool(block.name, block.input)
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": out}]})
        resp = client.messages.create(model="claude-opus-5", max_tokens=16000,
                                      tools=tools, messages=messages)
    # exits on stop_reason "end_turn": that is the final answer

The SDK will drive that loop for you. In Python you decorate your functions with
@beta_tool and hand them to client.beta.messages.tool_runner(...), then call
runner.until_done(); in TypeScript it is betaZodTool plus
client.beta.messages.toolRunner(...). You write the tool functions, it handles
request, execute, feed back, repeat, with per-turn hooks for approval gates,
logging, and retries.

Know the naming trap cold, because it costs people days. The API TOOL RUNNER and
the CLAUDE AGENT SDK are different packages with different scope:

  - Tool Runner lives in the regular Anthropic SDK (anthropic /
    @anthropic-ai/sdk) at client.beta.messages.tool_runner. It loops over tools
    YOU define. No built-in tools, no filesystem, no sandbox.
  - Claude Agent SDK (claude-agent-sdk / @anthropic-ai/claude-agent-sdk) is
    Claude Code packaged as a library: query(prompt, options) plus built-in Read,
    Write, Edit, Bash, Glob, Grep, WebSearch and WebFetch, context management,
    hooks, subagents, permissions and sessions.

Both are harness only, you still host them. This lab writes the loop by hand,
then wraps the identical mechanics in a tool_runner helper, and proves the two
reach the same verified answer with the runner bounded by a max_iterations guard.

Run: python3 modules/academy-content/labs/claude-agents/cla3-agent-sdk-loop.py
"""
import sys, os, re
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route


def calculator(expression):
    """One real tool. Deterministic, so the loop's answer is checkable."""
    m = re.search(r"(-?\d+)\s*(?:plus|\+)\s*(-?\d+)", expression.lower())
    return str(int(m.group(1)) + int(m.group(2))) if m else "error"


TOOLS = ["calculator", "weather", "search"]
FNS = {"calculator": lambda args: calculator(args["expression"])}


def model_turn(messages):
    """Stands in for one client.messages.create call. If the last message carries
    a tool_result the model has what it needs and finishes (end_turn); otherwise
    it reads the ask and requests a tool (tool_use)."""
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], list) \
            and last["content"][0].get("type") == "tool_result":
        return {"stop_reason": "end_turn",
                "content": [{"type": "text",
                             "text": "The answer is %s." % last["content"][0]["content"]}]}
    ask = last["content"] if isinstance(last["content"], str) else ""
    if tool_route(ask, TOOLS) == "calculator":
        return {"stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "toolu_01", "name": "calculator",
                             "input": {"expression": ask}}]}
    return {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "I can answer that directly."}]}


ASK = "Please calculate 2 plus 2 with the calculator."


# --- 1. The loop, written by hand. -------------------------------------------
messages = [{"role": "user", "content": ASK}]
resp = model_turn(messages)
hand_rounds = 0
print("STEP 1: the hand-written loop")
while resp["stop_reason"] == "tool_use" and hand_rounds < 8:
    hand_rounds += 1
    for block in resp["content"]:
        if block["type"] == "tool_use":
            out = FNS[block["name"]](block["input"])
            print("  round %d: %s(%r) -> %r" % (hand_rounds, block["name"],
                                                block["input"]["expression"], out))
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block["id"], "content": out}]})
    resp = model_turn(messages)
hand_answer = "".join(b.get("text", "") for b in resp["content"])
print("  stop_reason :", resp["stop_reason"])
print("  final answer:", repr(hand_answer))


# --- 2. The same mechanics, wrapped the way the Tool Runner wraps them. ------
def tool_runner(messages, fns, max_iterations=8):
    """What client.beta.messages.tool_runner does for you: request, execute, feed
    back, repeat, until the model stops asking. Returns the final message and a
    trace, and refuses to spin past max_iterations, because a real loop always
    bounds itself against a model that will not converge."""
    trace = []
    resp = model_turn(messages)
    rounds = 0
    while resp["stop_reason"] == "tool_use":
        if rounds >= max_iterations:
            return {"stop_reason": "max_iterations", "content": []}, trace
        rounds += 1
        for block in resp["content"]:
            if block["type"] != "tool_use":
                continue
            out = fns[block["name"]](block["input"])
            trace.append((block["name"], out))
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block["id"], "content": out}]})
        resp = model_turn(messages)
    return resp, trace


final, trace = tool_runner([{"role": "user", "content": ASK}], FNS)
runner_answer = "".join(b.get("text", "") for b in final["content"])
print("")
print("STEP 2: the tool_runner helper")
print("  tool calls it drove :", trace)
print("  stop_reason         :", final["stop_reason"])
print("  final answer        :", repr(runner_answer))

# --- 3. A runner that cannot converge is stopped by its own guard. -----------
starved, _ = tool_runner([{"role": "user", "content": ASK}], FNS, max_iterations=0)
print("")
print("STEP 3: the guard, with max_iterations set to 0")
print("  stop_reason :", starved["stop_reason"])

looped = hand_rounds >= 1 and len(trace) == 1
agree = (hand_answer == runner_answer and "4" in runner_answer)
ended_clean = (resp["stop_reason"] == "end_turn" and final["stop_reason"] == "end_turn")
bounded = (starved["stop_reason"] == "max_iterations")

print("")
print("STEP 4: checks")
print("  both paths ran exactly one tool round     :", looped)
print("  hand loop and tool_runner agree on 4      :", agree)
print("  both exited on stop_reason end_turn       :", ended_clean)
print("  the iteration guard stopped a runaway loop:", bounded)

ok = looped and agree and ended_clean and bounded
print("")
print("TOOL RUNNER AND HAND LOOP AGREE ON END_TURN: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Ask, execute, feed back, repeat. Next: the harness built on this loop.")
