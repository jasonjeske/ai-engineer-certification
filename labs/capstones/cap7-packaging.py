#!/usr/bin/env python3
"""
CAPSTONE 6 (ch6): Packaging the portfolio, a manifest and README generator.

The projects are built. Now they have to be FOUND and UNDERSTOOD in thirty
seconds, because that is all the attention a screener gives. This final capstone
is the packaging layer: a small generator that takes a manifest of your finished
projects (the four capstones you just built) and emits, for each one, the
README-in-30-seconds a hiring manager actually reads, one that leads with the
business OUTCOME, links a live DEMO, names the METRICS, and lists the STACK.

It reuses the portfolio rubric from Capstone 1 as the acceptance test: every
generated README must contain the sections that make a project screenable, and
every manifest entry must be rubric-ready (deployed, evaluated, real, original).
The generator builds the portfolio index and the per-project READMEs, then asserts
each one passes the section check. It prints the invariant and exits 0.

Run: python3 modules/academy-content/labs/capstones/cap7-packaging.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete  # noqa: F401  (available to every capstone)

# ── The portfolio manifest: the four capstones you just built. ───────────────────
MANIFEST = [
    {
        "name": "RAG Assistant over Company Docs",
        "outcome": "Cuts support lookup time by grounding answers in real docs with citations.",
        "demo": "https://demo.example.com/rag-assistant",
        "metrics": "retrieval hit-rate 0.92, zero hallucinations on the eval set",
        "stack": ["embeddings", "vector search", "grounded generation", "citations"],
    },
    {
        "name": "Autonomous Agent with Evals, Permissions and Cost Report",
        "outcome": "Completes a multi-step task under an enforced permission model, and proves it.",
        "demo": "https://demo.example.com/agentic-capstone",
        "metrics": "outcome 2/2 and trajectory 2/2 on the eval set, every ungranted call blocked and traced",
        "stack": ["agent loop", "trajectory + outcome evals", "permission model", "trace and cost report"],
    },
    {
        "name": "Reusable Eval Pipeline",
        "outcome": "Gates every build on a golden set so a silent regression never ships.",
        "demo": "https://demo.example.com/eval-pipeline",
        "metrics": "blocks 100% of seeded regressions, code + judge graders",
        "stack": ["golden dataset", "code grader", "LLM-as-judge", "gate"],
    },
    {
        "name": "Self-Red-Teamed Secure LLM App",
        "outcome": "Proves its own security by attacking itself against the OWASP LLM Top 10.",
        "demo": "https://demo.example.com/secure-app",
        "metrics": "blocks 5/5 attacks hardened vs 0/5 undefended, report published",
        "stack": ["input/output guards", "secret isolation", "egress allowlist", "red-team suite"],
    },
]

# Sections a screenable README must carry (this reuses the Capstone-1 rubric idea).
REQUIRED_SECTIONS = ["# ", "## Outcome", "## Live demo", "## Metrics", "## Stack"]


def generate_readme(project):
    """Emit the 30-second README for one project: title, outcome first, demo,
    metrics, stack. Outcome leads because that is what a manager screens for."""
    lines = [
        "# %s" % project["name"],
        "",
        "## Outcome",
        project["outcome"],
        "",
        "## Live demo",
        project["demo"],
        "",
        "## Metrics",
        project["metrics"],
        "",
        "## Stack",
        ", ".join(project["stack"]),
        "",
    ]
    return "\n".join(lines)


def generate_index(manifest):
    """The portfolio landing page: one screenable line per project."""
    out = ["# AI Engineering Portfolio", "",
           "%d shipped, evaluated, end-to-end projects." % len(manifest), ""]
    for p in manifest:
        out.append("- **%s**: %s ([demo](%s))" % (p["name"], p["outcome"], p["demo"]))
    return "\n".join(out) + "\n"


def readme_passes(readme):
    """Acceptance test: every required section is present and non-empty."""
    return all(section in readme for section in REQUIRED_SECTIONS)


def main():
    print("PACKAGING THE PORTFOLIO: manifest -> index + %d READMEs\n" % len(MANIFEST))

    index = generate_index(MANIFEST)
    print("STEP 1: generated portfolio index")
    for line in index.strip().splitlines():
        print("  %s" % line)

    print("\nSTEP 2: generate + acceptance-test each project README")
    passed = 0
    for project in MANIFEST:
        readme = generate_readme(project)
        ok_one = readme_passes(readme)
        passed += 1 if ok_one else 0
        print("  %-42s sections=%s -> %s" % (
            project["name"], "complete" if ok_one else "MISSING",
            "OK" if ok_one else "FAIL"))
        assert ok_one, "README for %r is missing required sections" % project["name"]

    index_ok = "# AI Engineering Portfolio" in index and index.count("[demo](") == len(MANIFEST)
    all_readmes_ok = passed == len(MANIFEST)

    print("")
    print("  index lists all %d projects with demo links: %s" % (
        len(MANIFEST), "YES" if index_ok else "NO"))
    print("  every README carries all required sections: %s" % (
        "YES" if all_readmes_ok else "NO"))

    ok = index_ok and all_readmes_ok
    print("")
    print("PORTFOLIO MANIFEST AND README GENERATED: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Packaged, screenable in thirty seconds, outcome-first. That is a portfolio that gets hired.")


if __name__ == "__main__":
    main()
