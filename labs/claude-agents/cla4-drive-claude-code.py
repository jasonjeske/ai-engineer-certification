#!/usr/bin/env python3
"""
LAB CLA4: Claude Code as an agent you drive.

Claude Code is not a chat box with a terminal theme. Architecturally it is a
harness: the loop you built in the last chapter, wrapped around a fixed set of
built-in tools (Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch), a
permission system that decides which of them may run, and session state that
survives between runs. The Claude Agent SDK is that same harness as a library.

So you can drive it two ways, and neither of them is typing into a prompt.

Programmatically:

    from claude_agent_sdk import query, ClaudeAgentOptions
    async for message in query(prompt="summarize the failing test",
                               options=ClaudeAgentOptions(allowed_tools=["Read", "Grep"])):
        print(message)

Or from the command line, which is what you reach for in CI and cron:

    claude -p "summarize the failing test" \\
      --output-format json --allowed-tools "Read,Grep"
    claude -p "now fix it" --resume <session_id>

Both give you the same three things, and they are the whole interface worth
learning. First, a STREAM of typed messages: an init message advertising the
session and the tools actually available, then assistant turns carrying tool_use
blocks, then the tool results. Second, a RESULT envelope that ends every run,
carrying is_error, num_turns, the final text, a session_id and the run's cost.
Third, a SESSION you can resume by id, which is how a second run continues the
first instead of starting cold.

The permission surface is the part people skip and regret. A tool that is not
granted is not merely discouraged, it never executes, and the run reports the
refusal rather than dying.

This lab drives a mock harness: one run with Read and Grep granted, an attempt at
Bash that is refused, and a second run resumed on the first run's session id.

Run: python3 modules/academy-content/labs/claude-agents/cla4-drive-claude-code.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

BUILT_INS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"]
REPO = {"notes/review.txt": "The release is excellent and the team is fantastic"}
SESSIONS = {}          # session_id -> transcript, the state a resume reads back


def run_tool(name, arg):
    if name == "Read":
        return REPO[arg]
    if name == "Grep":
        return [p for p, body in REPO.items() if arg.lower() in body.lower()]
    raise KeyError(name)


def query(prompt, allowed_tools, plan, resume=None):
    """One harness run. Emits the message stream a real run emits, ending in the
    result envelope. `plan` is the tool calls the model wants this run; anything
    not in allowed_tools is refused at the permission layer and never executes."""
    session_id = resume or "sess_%02d" % (len(SESSIONS) + 1)
    transcript = SESSIONS.get(session_id, [])
    stream = [{"type": "system", "subtype": "init", "session_id": session_id,
               "tools": [t for t in BUILT_INS if t in allowed_tools],
               "resumed": resume is not None, "prior_turns": len(transcript)}]
    refusals = []
    for name, arg in plan:
        stream.append({"type": "assistant",
                       "content": [{"type": "tool_use", "name": name, "input": arg}]})
        if name not in allowed_tools:
            # Not granted: the call is refused, the tool function is never called.
            refusals.append(name)
            stream.append({"type": "user", "content": [
                {"type": "tool_result", "is_error": True,
                 "content": "permission denied: %s is not in allowed_tools" % name}]})
            continue
        stream.append({"type": "user", "content": [
            {"type": "tool_result", "is_error": False, "content": run_tool(name, arg)}]})
    # The model's final text, from whatever the granted tools actually returned.
    read = next((e["content"][0]["content"] for e in stream
                 if e["type"] == "user" and not e["content"][0]["is_error"]
                 and isinstance(e["content"][0]["content"], str)), "")
    answer = complete("sentiment: %s" % read) if read else "nothing to read"
    transcript = transcript + [prompt, answer]
    SESSIONS[session_id] = transcript
    stream.append({"type": "result", "subtype": "success", "is_error": False,
                   "session_id": session_id, "num_turns": len(transcript) // 2,
                   "result": answer, "refused": refusals, "total_cost_usd": 0.0})
    return stream


# --- 1. A run with Read and Grep granted, and an ungranted Bash attempt. -----
first = query("read notes/review.txt and judge the mood",
              allowed_tools=["Read", "Grep"],
              plan=[("Grep", "excellent"), ("Read", "notes/review.txt"),
                    ("Bash", "rm -rf build")])
init = first[0]
result = first[-1]

print("STEP 1: the message stream")
for event in first:
    if event["type"] == "system":
        print("  init    : session=%s tools=%s" % (event["session_id"], event["tools"]))
    elif event["type"] == "assistant":
        b = event["content"][0]
        print("  assistant: tool_use %s(%r)" % (b["name"], b["input"]))
    elif event["type"] == "user":
        b = event["content"][0]
        print("  result   : is_error=%s %r" % (b["is_error"], b["content"]))
    else:
        print("  envelope : is_error=%s turns=%s result=%r refused=%s"
              % (event["is_error"], event["num_turns"], event["result"], event["refused"]))

# --- 2. Resume the same session by id: the second run continues the first. ---
second = query("and the one before it?", allowed_tools=["Read"],
               plan=[("Read", "notes/review.txt")], resume=result["session_id"])
print("")
print("STEP 2: resumed run")
print("  resumed session :", second[0]["session_id"], "prior_turns=%s" % second[0]["prior_turns"])
print("  envelope turns  :", second[-1]["num_turns"])

bash_never_ran = ("Bash" in result["refused"] and "Bash" not in init["tools"])
granted_ran = any(e["type"] == "user" and e["content"][0]["is_error"] is False for e in first)
envelope_ok = (result["is_error"] is False and result["subtype"] == "success"
               and result["result"] == "positive" and result["session_id"].startswith("sess_"))
resumed_ok = (second[0]["resumed"] and second[0]["prior_turns"] == 2
              and second[-1]["num_turns"] == 2)

print("")
print("STEP 3: checks")
print("  init advertised only the granted tools     :", "Bash" not in init["tools"])
print("  the ungranted Bash call never executed     :", bash_never_ran)
print("  the granted tools did run                  :", granted_ran)
print("  the result envelope is clean and answerable :", envelope_ok)
print("  the resumed run continued the transcript   :", resumed_ok)

ok = bash_never_ran and granted_ran and envelope_ok and resumed_ok
print("")
print("CLAUDE CODE RUN RETURNED A CLEAN RESULT ENVELOPE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Drive it, grant it, resume it. Next: subagents, hooks and MCP inside it.")
