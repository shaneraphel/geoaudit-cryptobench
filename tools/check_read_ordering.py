#!/usr/bin/env python3
"""Did every preregistration precede the read it licensed? Ask the git graph.

Why this exists, and why the check it replaces was weaker than it looked
------------------------------------------------------------------------
Each read of the held-out fold that claims a plan records that plan's commit and
a boolean, ``is_ancestor_of_head``. That boolean is nearly vacuous: **every
commit in the history is an ancestor of HEAD.** It would be false only for a plan
on an abandoned branch, so it establishes that the plan exists, not that it came
first.

The claim the paper actually makes is about *order*: the plan fixed the statistic
before the fold was scored under it. The check for that is different --- the plan's
commit must be an ancestor of the commit that **introduced the read artifact** --
and nothing was running it. A reviewer asking for "the repository with full git
history so the ordering can be verified" is asking for exactly this, and the
history has been complete and pushed the whole time; what was missing was one
command that walks all thirteen reads and says so.

So this tool derives both commits from the graph at verification time and reports
the ordering it finds, rather than reading a recorded boolean:

* the plan's commit, taken from the read artifact's own ``plan`` block;
* the commit that first added the read artifact, from
  ``git log --diff-filter=A -- <path>``;
* whether the first is an ancestor of the second, by ``git merge-base``.

Three outcomes and all three are reported. **ordered** is the plan strictly
preceding the read. **same commit** means the plan and the read landed together,
which is not preregistration and is called out rather than counted as ordered.
**out of order** is the failure the paper's honesty section would not survive.

What it deliberately does not do
---------------------------------
It does not judge whether a read *needed* a plan. Reads 1--4 predate the
preregistration discipline and say so; they are listed with no plan and that is
the honest state, not a violation. Inventing a plan for them after the fact is
the thing preregistration exists to prevent.

It also reports rather than assumes the history is usable: a shallow clone cannot
answer any of this, and it says so instead of passing.

Nothing here reads a label, a score or a fold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.read_ordering.v1"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
OUT = ROOT / "results/official_fold/READ_ORDERING.json"

# Where a read artifact may record the commit that fixed its plan.
#
# This list started as four names and reported "no plan recorded" for five reads,
# one of which is called PREREGISTERED_READ.json -- so the checker was grading
# spelling, which is precisely the failure AGENT_MEMORY 2k records for the
# train-fold declaration gate. The list is now derived from what the tree
# actually uses: any top-level key whose name mentions a plan or a
# preregistration and whose value is an object carrying a commit. Enumerating
# the spellings present beats guessing them, and _plan_of falls through to that
# structural rule when none of the named keys matches.
PLAN_KEYS = ("plan", "preregistration", "prereg", "plan_artifact",
             "provenance_of_the_plan", "preregistered_plan", "plan_provenance")
COMMIT_KEYS = ("committed_in", "commit", "plan_commit", "committed_at_commit")
PLAN_NAME = ("plan", "prereg")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def _is_shallow() -> bool:
    return (ROOT / ".git/shallow").exists()


def _added_in(rel: str) -> str | None:
    """The commit that first added a path, or None if git does not know it."""
    out = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", a, b],
        capture_output=True).returncode == 0


def _commit_in(block: dict) -> str | None:
    for ck in COMMIT_KEYS:
        if block.get(ck):
            return str(block[ck])
    return None


def _plan_artifact_of(doc: dict) -> tuple[str | None, str | None]:
    """The plan artifact a read names, and any commit the read claims for it.

    Three shapes are in the tree and all three are read. A named key holding an
    object with a commit (``plan``, ``provenance_of_the_plan``); an object with
    an artifact and no commit (``provenance_of_the_choice``); and a bare path
    string (``selection_provenance``). A checker that knew one shape would grade
    spelling, which is the failure ``AGENT_MEMORY`` 2k records for the
    train-fold gate --- this one reported "no plan recorded" for a file named
    ``PREREGISTERED_READ.json`` before the fallbacks were added.

    The returned commit is the read's *claim*, kept only to be cross-checked.
    The commit this tool orders against is derived from the graph.
    """
    # Not keyed on the field name at all. Three key spellings were tried and
    # three were wrong -- `provenance_of_the_choice` and `selection_provenance`
    # contain neither "plan" nor "prereg", so a name-based rule reported "no
    # plan recorded" for reads that plainly have one. The plans themselves are
    # named consistently: every one is results/architecture_sweep/PREREGISTERED_*.
    # So the document is walked for any string that resolves to such a file, and
    # the commit is taken from the graph. This grades the tree's actual
    # convention rather than a guess about its field names.
    found: str | None = None
    claimed: str | None = None

    def walk(node) -> None:
        nonlocal found, claimed
        if isinstance(node, dict):
            if found is None:
                for v in node.values():
                    if isinstance(v, str) and _is_plan_path(v):
                        found = v
                        claimed = _commit_in(node)
                        return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return found, claimed


def _is_plan_path(s: str) -> bool:
    if not s.endswith(".json") or "PREREGISTERED" not in s:
        return False
    return (ROOT / s).is_file()


def build(write: bool) -> int:
    if _is_shallow():
        raise SystemExit(
            "this is a shallow clone, so the ordering of a plan against a read "
            "cannot be established from it. Re-clone without --depth, or fetch "
            "the full history with `git fetch --unshallow`. This tool reports "
            "nothing rather than passing on a history it cannot see.")

    ledger = json.loads(LEDGER.read_text())
    rows = []
    for r in ledger["indexed_read_sequence"]:
        rel = r["artifact"]
        path = ROOT / rel
        doc = json.loads(path.read_text()) if path.exists() else {}
        plan_rel, claimed = _plan_artifact_of(doc)
        # The ordering is derived from when the plan artifact entered the
        # history, not from what the read says about it. A read that records its
        # own plan commit gets that claim cross-checked against the graph; a
        # read that only names the plan by path is checked just as strongly.
        plan_commit = _added_in(plan_rel) if plan_rel else None
        read_commit = _added_in(rel)
        claim_agrees = (None if not (claimed and plan_commit)
                        else claimed == plan_commit
                        or _is_ancestor(claimed, plan_commit)
                        or _is_ancestor(plan_commit, claimed))

        if plan_rel is None:
            verdict = "no plan recorded"
        elif plan_commit is None:
            verdict = "plan artifact not in history"
        elif read_commit is None:
            verdict = "read artifact not in history"
        elif plan_commit == read_commit:
            verdict = "same commit"
        elif _is_ancestor(plan_commit, read_commit):
            verdict = "ordered"
        else:
            verdict = "OUT OF ORDER"

        rows.append({
            "read_index": r["read_index"],
            "artifact": rel,
            "kind": r["kind"],
            "plan_artifact": plan_rel,
            "plan_added_in": plan_commit,
            "commit_the_read_claims_for_its_plan": claimed,
            "the_claim_agrees_with_the_graph": claim_agrees,
            "read_added_in": read_commit,
            "verdict": verdict,
            "plan_is_ancestor_of_head": (
                _is_ancestor(plan_commit, "HEAD") if plan_commit else None),
        })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    failures = [r for r in rows if r["verdict"] in ("OUT OF ORDER", "same commit")]
    disagreeing = [r for r in rows
                   if r["the_claim_agrees_with_the_graph"] is False]

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": (
            "for every indexed read of the held-out fold that claims a plan, "
            "did the plan's commit precede the commit that introduced the read"),
        "why_this_is_stronger_than_what_the_artifacts_record": (
            "each read records is_ancestor_of_head, and every commit in a "
            "history is an ancestor of HEAD. That establishes the plan exists, "
            "not that it came first. The ordering claim needs the plan to be an "
            "ancestor of the commit that added the read, which is what is "
            "derived here from the graph at verification time"),
        "history": {
            "is_shallow": False,
            "n_commits": int(_git("rev-list", "--count", "HEAD") or 0),
            "head": _git("rev-parse", "HEAD"),
            "why_recorded": (
                "a shallow clone cannot answer any of this and the tool refuses "
                "on one, so the depth it actually saw belongs in the artifact"),
        },
        "verdict_counts": counts,
        "n_reads": len(rows),
        "n_ordered": counts.get("ordered", 0),
        "n_without_a_plan": counts.get("no plan recorded", 0),
        "n_out_of_order": counts.get("OUT OF ORDER", 0),
        "n_same_commit": counts.get("same commit", 0),
        "n_reads_whose_recorded_plan_commit_disagrees_with_the_graph":
            len(disagreeing),
        "why_the_claim_is_cross_checked": (
            "a read that records its own plan commit is reporting on itself. "
            "The ordering here is derived from when the plan artifact entered "
            "the history, and the read's claim is compared against that rather "
            "than used in place of it"),
        "reads": rows,
        "what_no_plan_recorded_means": (
            "reads 1-4 predate the preregistration discipline and say so in "
            "their own artifacts. They are listed without a plan because that "
            "is the honest state; inventing one for them now is the thing "
            "preregistration exists to prevent"),
        "what_same_commit_means": (
            "the plan and the read landed in one commit, which is not "
            "preregistration. It is separated from `ordered` rather than "
            "counted with it"),
    }

    w = max(len(r["artifact"]) for r in rows)
    print(f"history: {doc['history']['n_commits']} commits, not shallow\n")
    print(f"{'#':>2}  {'artifact':<{w}}  {'plan':>8}  {'read':>8}  verdict")
    for r in rows:
        pc = (r["plan_added_in"] or "")[:8] or "--"
        rc = (r["read_added_in"] or "")[:8] or "--"
        print(f"{r['read_index']:>2}  {r['artifact']:<{w}}  {pc:>8}  {rc:>8}  "
              f"{r['verdict']}")
    print("\n" + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if disagreeing:
        print("\nrecorded plan commit disagrees with the graph on: "
              + ", ".join(str(r["read_index"]) for r in disagreeing))
    if failures:
        print("\nFAILED: " + "; ".join(
            f"read {f['read_index']} is {f['verdict']}" for f in failures))

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
