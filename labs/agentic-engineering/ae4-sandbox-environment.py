#!/usr/bin/env python3
"""
LAB AE4: Environments and side effects.

An agent's environment is the set of things it can change, and it is a design
decision, not an afterthought. This lab gives an agent a filesystem made of a
Python dict, then hands it a badly expanded glob, the single most common way a
coding agent destroys work. In the sandbox the damage is total and completely
reversible, because a snapshot exists and the dict is the only writable surface.
Against a real filesystem the identical call has no snapshot behind it, which is
why a coding agent's real environment needs version control before it needs
capability. A path guard refuses every attempt to reach outside the sandbox.

NOTHING here touches your real filesystem: the tools close over a dict and the
module never opens a file.

Run: python3 modules/academy-content/labs/agentic-engineering/ae4-sandbox-environment.py
"""
import sys, os, fnmatch
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

# ── The ENVIRONMENT. Three files, and a snapshot that stands in for git. ──────
FS = {"notes.md": "# raven outline\n", "main.py": "print('ship it')\n",
      "secrets.env": "TOKEN=abc123\n"}
SNAPSHOT = dict(FS)                      # the commit the agent can be rolled back to

def guard(path):
    """The boundary. A sandbox is only a sandbox if escape is impossible: no
    absolute paths, no parent traversal, no home expansion, no separators."""
    if path.startswith(("/", "~")) or ".." in path or os.sep in path:
        return "REFUSED: %r is outside the sandbox" % path
    return None

def delete(pattern):
    """The destructive tool. A glob is a loaded weapon: '*' matches everything."""
    bad = guard(pattern)
    if bad:
        return bad
    hit = [k for k in FS if fnmatch.fnmatch(k, pattern)]
    for k in hit:
        del FS[k]
    return "deleted %d file(s): %s" % (len(hit), sorted(hit))

print("ENVIRONMENT before: %s" % sorted(FS))
print("")
print("STEP 1: the agent means 'clean the .tmp files' and emits delete('*')")
print("  -> %s" % delete("*"))
print("  environment now: %s  (empty)" % sorted(FS))
destroyed = FS == {}

print("")
print("STEP 2: roll back from the snapshot, the thing a real filesystem lacks")
FS.update(SNAPSHOT)
print("  environment now: %s" % sorted(FS))
recovered = FS == SNAPSHOT and FS["secrets.env"] == "TOKEN=abc123\n"
print("  contents byte-identical to the snapshot: %s" % recovered)

print("")
print("STEP 3: the same call aimed outside the sandbox")
escapes = ["/etc/passwd", "../../.ssh/id_ed25519", "~/Documents/taxes.pdf"]
refused = 0
for attempt in escapes:
    result = delete(attempt)
    refused += result.startswith("REFUSED")
    print("  delete(%r) -> %s" % (attempt, result))
blocked = refused == len(escapes)

print("")
print("WHY THIS MATTERS ON A REAL FILESYSTEM")
print("  Same delete('*'), same agent, no snapshot: main.py and secrets.env are")
print("  gone and unrecoverable. A coding agent's environment is files plus a")
print("  shell plus version control, and the version control is the snapshot.")
print("  Rich environment, real capability, real blast radius.")

ok = destroyed and recovered and blocked
print("")
print("SANDBOX CONTAINED THE DESTRUCTION AND BLOCKED EVERY ESCAPE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Choose the environment first. Capability and risk arrive together.")
