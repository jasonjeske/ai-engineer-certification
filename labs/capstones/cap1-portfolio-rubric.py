#!/usr/bin/env python3
"""
CAPSTONE 1 (ch1): The portfolio-readiness rubric.

Before you build a single project, you need to know what a hiring manager
actually screens for. The research is blunt: they screen for SHIPPED, EVALUATED,
end-to-end work with a real outcome, and a generic tutorial clone is a
disqualifying signal, not a neutral one. This lab turns that into a rubric you
can score any project spec against, so "is this portfolio-ready" stops being a
feeling and becomes a number.

The rubric weights six things a manager can verify in thirty seconds: is it
DEPLOYED (a live demo link), is it EVALUATED (real metrics, not a screenshot),
does it use a REAL corpus / real workflow (not a toy), does it have a
README-in-30-seconds, does it state a BUSINESS OUTCOME, and is it ORIGINAL
(not a renamed tutorial). Each project earns weighted points; the score is a
percentage; a project is portfolio-ready only if it clears the bar.

We score three specs: a hireable RAG assistant, a bare tutorial clone, and a
half-built project that works but was never evaluated or deployed. The lab
asserts the rubric ranks them correctly and gates on the bar, then prints the
success invariant and exits 0.

Run: python3 modules/academy-content/labs/capstones/cap1-portfolio-rubric.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
# This capstone needs no model call: readiness is a deterministic rubric. The
# shared mock is importable here for parity with the other capstones.
from academy_llm import complete  # noqa: F401  (available to every capstone)

# ── The rubric: six weighted, manager-verifiable criteria. ───────────────────────
# Weights sum to 100. DEPLOYED and EVALUATED carry the most weight because they
# are the two signals that separate a hire from a notebook.
RUBRIC = [
    ("deployed",   25, "a live demo link a manager can click"),
    ("evaluated",  25, "real metrics reported, not just a screenshot"),
    ("real_corpus",15, "a real document set or workflow, not a toy example"),
    ("readme",     15, "a README that explains it in thirty seconds"),
    ("outcome",    12, "states the business outcome, not the cleverness"),
    ("original",    8, "not a renamed tutorial clone"),
]
BAR = 70  # a project scoring below this is not portfolio-ready.


def score_spec(spec):
    """Score one project spec against the rubric. `spec` is a dict of
    criterion -> bool. Returns (points, percent, missing_criteria)."""
    earned, missing = 0, []
    for key, weight, _ in RUBRIC:
        if spec.get(key):
            earned += weight
        else:
            missing.append(key)
    total = sum(w for _, w, _ in RUBRIC)
    return earned, round(100 * earned / total), missing


# ── Three real-world project specs. ──────────────────────────────────────────────
SPECS = {
    "RAG assistant over company docs": {
        "deployed": True, "evaluated": True, "real_corpus": True,
        "readme": True, "outcome": True, "original": True,
    },
    "tutorial to-do chatbot clone": {
        "deployed": False, "evaluated": False, "real_corpus": False,
        "readme": False, "outcome": False, "original": False,
    },
    "local agent that works but was never shipped": {
        "deployed": False, "evaluated": False, "real_corpus": True,
        "readme": True, "outcome": False, "original": True,
    },
}


def main():
    print("PORTFOLIO READINESS RUBRIC (bar = %d%%)\n" % BAR)
    print("%-46s %-8s %-8s %s" % ("PROJECT", "SCORE", "READY", "MISSING"))
    scored = {}
    for name, spec in SPECS.items():
        pts, pct, missing = score_spec(spec)
        scored[name] = (pts, pct, missing)
        ready = pct >= BAR
        print("%-46s %-8s %-8s %s" % (
            name, "%d%%" % pct, "YES" if ready else "no",
            ", ".join(missing) if missing else "none"))

    hireable = scored["RAG assistant over company docs"][1]
    clone = scored["tutorial to-do chatbot clone"][1]
    half = scored["local agent that works but was never shipped"][1]

    # The invariants a real rubric must satisfy:
    #  (a) the hireable project clears the bar,
    #  (b) the tutorial clone fails hard,
    #  (c) a working-but-unshipped project still fails (deploy + evals are the gap),
    #  (d) the ranking is strictly correct: hireable > half-built > clone.
    assert hireable >= BAR, "hireable project must clear the bar"
    assert clone < BAR, "tutorial clone must fail the bar"
    assert half < BAR, "unshipped project must fail: no deploy, no evals"
    assert hireable > half > clone, "ranking must be hireable > half-built > clone"

    print("")
    print("  hireable %d%% >= bar, clone %d%% and unshipped %d%% both < bar" %
          (hireable, clone, half))
    print("  ranking correct: hireable > half-built > clone")
    ok = hireable >= BAR and clone < BAR and half < BAR and hireable > half > clone
    print("")
    print("PORTFOLIO RUBRIC SCORED AND RANKED THE SPECS: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Now you know the target. The next four capstones each hit it. Build one.")


if __name__ == "__main__":
    main()
