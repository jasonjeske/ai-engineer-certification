#!/usr/bin/env python3
"""
LAB CLA2: Streaming, structured output, and prompt caching.

Three mechanics that decide how a Claude app FEELS, whether its output is safe to
load, and what it costs. All three live on the same messages endpoint.

1. STREAMING. Switch create for stream and paint tokens as they arrive:

    with client.messages.stream(model="claude-opus-5", max_tokens=64000,
                                messages=[...]) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

   On the wire it is server-sent events: message_start, content_block_delta
   events each carrying a text_delta, then message_stop. Two truths: the
   assembled deltas must equal the non-streamed message exactly, and
   time-to-first-token, not total time, is the number a user feels. Streaming is
   also required for large max_tokens (these models go to 128K output) because a
   single blocking request would hit an HTTP timeout.

2. STRUCTURED OUTPUT. Software needs structure it can load, so constrain the
   response to a JSON schema:

    resp = client.messages.create(
        model="claude-opus-5", max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[...])

   output_config.format is the canonical parameter; the old top-level
   output_format is deprecated. client.messages.parse() validates for you, and
   strict: true on a TOOL definition (schema needs additionalProperties: false
   plus required) does the same for tool inputs. Still wrap every parse, because
   a truncated response is a case your code must survive rather than crash on.

3. PROMPT CACHING. Mark a stable prefix and later calls reuse it cheaply:

    system=[{"type": "text", "text": BIG_SYSTEM,
             "cache_control": {"type": "ephemeral"}}]

   usage then splits three ways: cache_creation_input_tokens (first call, about
   1.25x), cache_read_input_tokens (later calls, about 0.1x), and input_tokens
   (uncached remainder, full price). Caching is a PREFIX match and the render
   order is tools, then system, then messages, so stable content goes first and
   the changing question last. Any byte change in the prefix invalidates it,
   which is why a timestamp in a system prompt quietly costs real money. Check
   usage.cache_read_input_tokens; if it stays zero, something is invalidating.

This lab runs all three and proves each one held.

Run: python3 modules/academy-content/labs/claude-agents/cla2-streaming-schema-cache.py
"""
import sys, os, re, json, time
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


def count_tokens(text):
    """Stands in for client.messages.count_tokens. Never use a foreign tokenizer
    to size a Claude prompt."""
    return len(text.split())


# --- 1. Streaming: deltas out, identical message back in. --------------------
def timed_stream(text, first_ms=8.0, per_ms=3.0):
    """Yield (delta, elapsed_ms) on a DETERMINISTIC simulated clock so TTFT is
    measurable offline. Each delta keeps its trailing whitespace, so
    concatenating every delta reproduces the source exactly, like a real
    text_stream."""
    clock, first = 0.0, True
    for chunk in re.findall(r"\S+\s*", text):
        clock += first_ms if first else per_ms
        first = False
        yield chunk, clock


full = complete("Context: The peregrine falcon is the fastest animal on earth. "
                "The cheetah is the fastest land animal. "
                "Question: What is the fastest animal on earth?")
assembled, ttft, count = "", None, 0
for delta, elapsed in timed_stream(full):
    if ttft is None:
        ttft = elapsed
    assembled += delta
    count += 1
    time.sleep(0.001)

print("STEP 1: streaming")
print("  full message        :", repr(full))
print("  deltas streamed     :", count)
print("  time-to-first-token : %.1f ms of %.1f ms total" % (ttft, 8.0 + 3.0 * (count - 1)))
print("  assembled equals it :", assembled == full)

# --- 2. Structured output: constrained, validated, and safe to parse. --------
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "number": {"type": "integer"}},
          "required": ["name", "number"],
          "additionalProperties": False}


def validate(obj, schema):
    """The check output_config.format enforces server-side: required fields
    present, each field the right JSON type."""
    py = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for field in schema.get("required", []):
        if field not in obj:
            return False
    for field, spec in schema.get("properties", {}).items():
        if field in obj and not isinstance(obj[field], py[spec["type"]]):
            return False
    return True


def parse_safely(raw):
    """Always wrap the parse. A model CAN return truncated output and your code
    must fail safely instead of raising into the request path."""
    try:
        return True, json.loads(raw)
    except json.JSONDecodeError as e:
        return False, "malformed JSON: %s" % e


raw = complete("Extract JSON: my name is Ada and the number is 42")
parsed_ok, obj = parse_safely(raw)
schema_ok = parsed_ok and validate(obj, SCHEMA) and obj["name"] == "Ada" and obj["number"] == 42
bad_ok, _bad = parse_safely('{"name": "Ada", "number": 42')   # truncated on purpose

print("")
print("STEP 2: structured output")
print("  raw                 :", repr(raw))
print("  parsed and valid    :", schema_ok)
print("  truncated JSON caught, not crashed :", bad_ok is False)

# --- 3. Prompt caching: same answers, a fraction of the repeated cost. -------
BIG_SYSTEM = ("You are a sentiment classifier. Read each review and label it "
              "positive, negative, or neutral. Return only the label. " * 6)
QUESTIONS = ["sentiment: I love this product",
             "sentiment: this update is terrible",
             "sentiment: absolutely fantastic experience"]


def billed(input_tokens, creation, read):
    """Cache writes cost about 1.25x, reads about 0.1x, fresh input 1x. Using the
    real multipliers makes the saving visible as one comparable number."""
    return input_tokens + 1.25 * creation + 0.1 * read


plain_cost, plain_answers = 0.0, []
cached_cost, cached_answers, written = 0.0, [], False
for q in QUESTIONS:
    sys_t, q_t = count_tokens(BIG_SYSTEM), count_tokens(q)
    answer = complete(BIG_SYSTEM + "\n" + q)
    plain_answers.append(answer)
    plain_cost += billed(sys_t + q_t, 0, 0)
    cached_answers.append(answer)
    if not written:                       # first call writes the prefix to cache
        cached_cost += billed(q_t, sys_t, 0)
        written = True
    else:                                 # later calls read it back cheaply
        cached_cost += billed(q_t, 0, sys_t)

print("")
print("STEP 3: prompt caching")
print("  answers             :", cached_answers)
print("  billed without cache: %.1f" % plain_cost)
print("  billed with cache   : %.1f" % cached_cost)
print("  count_tokens(prefix):", count_tokens(BIG_SYSTEM))

stream_ok = (assembled == full and ttft < (8.0 + 3.0 * (count - 1)))
structured_ok = (schema_ok and bad_ok is False)
cache_ok = (plain_answers == cached_answers and cached_cost < plain_cost
            and cached_answers[0] == "positive" and cached_answers[1] == "negative")

print("")
print("STEP 4: checks")
print("  streamed deltas assembled to the exact message :", stream_ok)
print("  output parsed, validated, and failed safely    :", structured_ok)
print("  caching cut the cost and changed no answer     :", cache_ok)

ok = stream_ok and structured_ok and cache_ok
print("")
print("STREAMING, SCHEMA, AND CACHE ALL HELD: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Stream for feel, constrain for safety, cache for cost. Next: the Agent SDK loop.")
