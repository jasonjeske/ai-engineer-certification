#!/usr/bin/env python3
"""
LAB CLA1: The Messages API and tool use. One shape, one handshake.

Every Claude call you make from code goes through POST /v1/messages and one SDK
call, and tools are a feature of that same endpoint, not a separate API:

    from anthropic import Anthropic
    client = Anthropic()                      # ANTHROPIC_API_KEY, or a saved profile
    resp = client.messages.create(
        model="claude-opus-5",                # the most capable Opus today
        max_tokens=16000,                     # non-streaming default; ~64000 when streaming
        system="You label sentiment.",        # a top-level field, NOT a message
        tools=[{"name": "weather",
                "description": "Call this to get the current weather for a city.",
                "input_schema": {"type": "object",
                                 "properties": {"city": {"type": "string"}},
                                 "required": ["city"]}}],
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    )

The reply is not a string. It is an object with three parts you must read: a
`content` LIST of typed blocks, a `usage` object carrying input_tokens and
output_tokens, and a `stop_reason` that says why generation stopped.

When Claude decides to use a tool it does NOT run it. It stops with stop_reason
"tool_use" and returns a tool_use block holding an id, the tool name, and the
input it chose. Your code executes the tool and sends the answer back as a
tool_result matched by tool_use_id. `tool_choice` steers that decision: "auto"
(default), "any" (some tool), a named tool, or "none". If the model returns
several tool_use blocks at once, run them and return ALL the tool_result blocks
in a SINGLE user message, because splitting them teaches Claude to stop calling
tools in parallel.

This lab builds that exact shape over the offline provider and proves the whole
handshake end to end.

Run: python3 modules/academy-content/labs/claude-agents/cla1-messages-and-tools.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import chat, tool_route


def count_tokens(text):
    # A token here is a whitespace word. Real code calls
    # client.messages.count_tokens; never size a Claude prompt with a foreign
    # tokenizer, it counts differently and will be wrong.
    return len(text.split())


TOOLS = [
    {"name": "weather",
     "description": "Call this to get the current weather for a city.",
     "input_schema": {"type": "object",
                      "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
    {"name": "calculator",
     "description": "Call this to evaluate an arithmetic expression.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
]


class Message:
    """What client.messages.create returns. `content` is a list of typed blocks,
    never a bare string, so downstream code reads the SHAPE and not just text."""
    def __init__(self, model, content, in_tokens, out_tokens, stop_reason):
        self.model = model
        self.content = content
        self.usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}
        self.stop_reason = stop_reason

    @property
    def text(self):
        return "".join(b.get("text", "") for b in self.content if b["type"] == "text")


def create(messages, model="claude-opus-5", max_tokens=16000, system=None,
           tools=None, tool_choice="auto"):
    """One endpoint, one shape. `system` is a separate top-level field. The API is
    stateless, so the CALLER resends the whole history on every request."""
    convo = ([{"role": "system", "content": system}] if system else []) + messages
    sent = " ".join(m["content"] for m in convo if isinstance(m["content"], str))
    if tools and tool_choice != "none":
        ask = [m for m in messages if m["role"] == "user"][-1]["content"]
        picked = tool_route(ask, [t["name"] for t in tools])
        if picked:
            block = {"type": "tool_use", "id": "toolu_01ABC", "name": picked,
                     "input": {"city": "Paris"} if picked == "weather"
                     else {"expression": ask}}
            return Message(model, [block], count_tokens(sent), 12, "tool_use")
    reply = chat(convo)
    return Message(model, [{"type": "text", "text": reply}],
                   count_tokens(sent), count_tokens(reply), "end_turn")


# --- 1. The model pauses to ask for a tool. -----------------------------------
history = [{"role": "user", "content": "What is the weather in Paris right now?"}]
resp = create(history, tools=TOOLS)

print("STEP 1: the model asked for a tool instead of answering")
print("  model       :", resp.model)
print("  stop_reason :", resp.stop_reason)
print("  content     :", resp.content)
print("  usage       :", resp.usage)

# --- 2. You run the tool and return a tool_result linked by tool_use_id. -----
tool_use = next((b for b in resp.content if b["type"] == "tool_use"), None)


def weather(city):
    return "18C and clear in %s" % city


tool_result = {"type": "tool_result", "tool_use_id": tool_use["id"],
               "content": weather(tool_use["input"]["city"])}
# Append the assistant turn, then ONE user message holding every tool_result.
history.append({"role": "assistant", "content": resp.content})
history.append({"role": "user", "content": [tool_result]})

print("")
print("STEP 2: you execute, then answer the model")
print("  tool_use    :", tool_use["name"], tool_use["input"])
print("  tool_result :", tool_result)

# --- 3. A plain text turn: no tools, a system prompt sets the job. -----------
text_turn = create([{"role": "user", "content": "I love this product, it works perfectly"}],
                   system="You classify the sentiment of a review.", tool_choice="none")
print("")
print("STEP 3: a text turn, stop_reason end_turn")
print("  stop_reason :", text_turn.stop_reason)
print("  .text       :", repr(text_turn.text))

shape_ok = (isinstance(resp.content, list) and resp.content
            and set(resp.usage.keys()) == {"input_tokens", "output_tokens"})
paused = (resp.stop_reason == "tool_use")
right_tool = (tool_use is not None and tool_use["name"] == "weather")
block_ok = (tool_use is not None and set(tool_use.keys()) >= {"type", "id", "name", "input"})
links_back = (tool_result["tool_use_id"] == tool_use["id"])
text_ok = (text_turn.stop_reason == "end_turn" and text_turn.text == "positive")
usage_ok = (resp.usage["input_tokens"] > 0 and text_turn.usage["output_tokens"] > 0)

print("")
print("STEP 4: checks")
print("  content is a list of typed blocks plus usage :", shape_ok)
print("  the model paused with stop_reason tool_use   :", paused)
print("  it chose the weather tool                    :", right_tool)
print("  the tool_use block has id, name, input       :", block_ok)
print("  the tool_result links back by tool_use_id    :", links_back)
print("  the no-tool turn ended clean with an answer  :", text_ok)
print("  tokens were accounted on both turns          :", usage_ok)

ok = shape_ok and paused and right_tool and block_ok and links_back and text_ok and usage_ok
print("")
print("MESSAGES API TOOL HANDSHAKE IS WELL FORMED: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("The model asks, you execute, you answer. Next: streaming, schemas, caching.")
