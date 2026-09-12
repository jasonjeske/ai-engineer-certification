#!/usr/bin/env python3
"""
LAB AE8: The MCP shape, and why the shape outlives the product.

An MCP server is a JSON-RPC endpoint exposing three primitives: TOOLS (actions),
RESOURCES (data), PROMPTS (templates), and it shares no memory with its client,
only JSON strings. You build one, then build a second server that is not a tool
server at all: an AGENT publishing an agent card and accepting agent/invoke. One
unchanged client drives both, because the envelope is identical and only the
method names differ. The typed JSON-RPC contract is the durable part; the SDK
name on top of it is the part that rots.

Run: python3 modules/academy-content/labs/agentic-engineering/ae8-protocol-pattern.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete
import json

TICKETS = [{"id": 1, "status": "open"}, {"id": 2, "status": "closed"},
           {"id": 3, "status": "open"}]
POLICY = ("Open tickets are triaged within 24 hours. Closed tickets are archived "
          "after 30 days.")


def _envelope(rid, result=None, code=None, message=""):
    err = code is not None
    body = {"error": {"code": code, "message": message}} if err else {"result": result}
    return json.dumps(dict({"jsonrpc": "2.0", "id": rid}, **body))


class TicketServer:
    """An MCP-shaped server: tools, resources, prompts, over JSON strings."""
    NAME = "academy-tickets"

    def transport(self, line):
        req = json.loads(line)
        rid, method, p = req.get("id"), req.get("method"), req.get("params", {})
        if method == "initialize":
            return _envelope(rid, {"protocolVersion": "2024-11-05",
                                   "serverInfo": {"name": self.NAME},
                                   "capabilities": {"tools": {}, "resources": {}, "prompts": {}}})
        if method == "tools/list":
            return _envelope(rid, {"tools": [{"name": "ticket_count",
                                              "inputSchema": {"status": {"type": "string"}}}]})
        if method == "tools/call":
            n = len([t for t in TICKETS if t["status"] == p["arguments"]["status"]])
            return _envelope(rid, {"content": [{"type": "text", "text": str(n)}]})
        if method == "resources/read":
            return _envelope(rid, {"contents": [{"uri": p["uri"], "text": POLICY}]})
        if method == "prompts/get":
            filled = "Context: %s Question: %s" % (POLICY, p["arguments"]["question"])
            return _envelope(rid, {"messages": [{"role": "user", "content": filled}]})
        return _envelope(rid, code=-32601, message="unknown method " + str(method))


class TriageAgent:
    """Not a tool server. An AGENT, discovered by its card and called by skill,
    speaking the SAME envelope. Note it is itself a client of TicketServer."""
    CARD = {"name": "triage-agent", "version": "1.0", "inputModes": ["text"],
            "skills": [{"id": "triage", "description": "Count and triage tickets."}]}

    def transport(self, line):
        req = json.loads(line)
        rid, method, p = req.get("id"), req.get("method"), req.get("params", {})
        if method == "agent/card":
            return _envelope(rid, self.CARD)
        if method == "agent/invoke":
            if p.get("skill") != "triage":
                return _envelope(rid, code=-32602, message="unknown skill")
            downstream = rpc(TicketServer(), 99, "tools/call", name="ticket_count",
                             arguments={"status": "open"})
            n = downstream["result"]["content"][0]["text"]
            return _envelope(rid, {"parts": [{"type": "text",
                                              "text": "%s tickets need triage" % n}]})
        return _envelope(rid, code=-32601, message="unknown method " + str(method))


def rpc(server, rid, method, **params):
    """The ONE client. It writes a JSON line and reads a JSON line. It knows
    nothing about tools, agents, or who is on the other end."""
    return json.loads(server.transport(json.dumps(
        {"jsonrpc": "2.0", "id": rid, "method": method, "params": params})))


tickets, agent = TicketServer(), TriageAgent()

print("STEP 1: drive the MCP-shaped tool server, all three primitives")
init = rpc(tickets, 1, "initialize")
listed = rpc(tickets, 2, "tools/list")["result"]["tools"]
count = rpc(tickets, 3, "tools/call", name="ticket_count",
            arguments={"status": "open"})["result"]["content"][0]["text"]
res = rpc(tickets, 4, "resources/read", uri="tickets://policy")["result"]["contents"][0]
pr = rpc(tickets, 5, "prompts/get", name="triage",
         arguments={"question": "how fast are open tickets triaged?"})
answered = complete(pr["result"]["messages"][0]["content"])  # prompt feeds the model
print("  initialize     ->", init["result"]["serverInfo"])
print("  tools/list     ->", [t["name"] for t in listed])
print("  tools/call     -> open tickets:", count)
print("  resources/read ->", res["text"][:38] + "...")
print("  prompts/get    -> model answered:", answered)

print("")
print("STEP 2: the SAME client function, pointed at an agent instead of a tool")
card = rpc(agent, 6, "agent/card")["result"]
print("  agent/card ->", card["name"], "skills:", [s["id"] for s in card["skills"]])
invoked = rpc(agent, 7, "agent/invoke", skill="triage", input="what needs work")
text = invoked["result"]["parts"][0]["text"]
print("  agent/invoke ->", text)

print("")
print("STEP 3: both reject an unknown method as an error, neither crashes")
e1 = rpc(tickets, 8, "tools/teleport")
e2 = rpc(agent, 9, "agent/teleport")
print("  tool server:", e1["error"]["code"], "| agent:", e2["error"]["code"])

tools_ok = count == "2"
resource_ok = "24 hours" in res["text"]
prompt_ok = "24 hours" in answered
card_ok = card["skills"][0]["id"] == "triage"
invoke_ok = text == "2 tickets need triage"
errors_ok = e1["error"]["code"] == -32601 and e2["error"]["code"] == -32601

print("")
print(f"tools, resources and prompts answered : {tools_ok and resource_ok and prompt_ok}")
print(f"agent card discovered its skill       : {card_ok}")
print(f"agent/invoke answered via the tool    : {invoke_ok}")
print(f"both returned -32601, no traceback    : {errors_ok}")

ok = all([tools_ok, resource_ok, prompt_ok, card_ok, invoke_ok, errors_ok])
print("")
print(f"ONE JSON-RPC CLIENT DROVE A TOOL SERVER AND AN AGENT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("The envelope is the durable part. Product names rot; typed contracts do not.")
