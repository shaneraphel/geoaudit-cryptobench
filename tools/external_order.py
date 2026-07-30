"""Establish, from git alone, the order in which the external validation happened.

Whether the 57-unit result may be called confirmatory rests on a sequence:

    the set is frozen -> the plan is committed -> predictions are generated
    -> the result is read

Each artifact records the commit it believes it was written at, but a record
inside an artifact is a claim by the artifact. This checks the claim against the
commit graph, which nobody involved can rewrite without rewriting history.

It also reports the one link the graph cannot settle. The predictions and the
read landed in a single commit, so their relative order is not a fact about the
graph; it is established instead by the read being a deterministic function of
those exact prediction files, which `make extread` recomputes. That is a
different kind of evidence and is labelled as such rather than folded in.

Needs a clone with history. An export without .git cannot run this, and the
right response to that is to say so, not to pass.

Usage: PYTHONPATH=src:tools python3.12 tools/external_order.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pocket_bench.paths import ROOT

OUT = ROOT / "results/external/EXTERNAL_ORDER.json"

# The four stages, in the order they must have happened, named by the artifact
# whose first appearance marks the stage.
STAGES = [
    ("set_frozen", "results/external/EXTERNAL_SET.json"),
    ("plan_committed", "results/external/PREREGISTERED_EXTERNAL.json"),
    ("predictions_generated", "results/external/predictions/p2rank.json"),
    ("result_read", "results/external/EXTERNAL_READ.json"),
]


def _git(*a: str) -> str:
    p = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                       text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def _have_history() -> None:
    if not (ROOT / ".git").exists():
        raise SystemExit(
            "no .git here, so the order of the external validation cannot be "
            "established. An export can still check the artifacts; it cannot "
            "check when they were written. Clone the repository to run this.")
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this clone is shallow, so commits before the cut are invisible "
            "and an ancestry check would pass by not looking. Fetch the full "
            "history with `git fetch --unshallow`.")


def _introduced(rel: str) -> str:
    sha = _git("log", "--format=%H", "--diff-filter=A", "-1", "--", rel)
    if not sha:
        raise SystemExit(f"no commit in this history introduces {rel}")
    return sha


def _is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base",
                           "--is-ancestor", a, b]).returncode == 0


def _blob_sha256(commit: str, rel: str) -> str | None:
    p = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"],
                       capture_output=True)
    if p.returncode != 0:
        return None
    return hashlib.sha256(p.stdout).hexdigest()


def build() -> dict:
    _have_history()
    head = _git("rev-parse", "HEAD")
    marks = [(name, rel, _introduced(rel)) for name, rel in STAGES]

    links = []
    for (a_name, _, a_sha), (b_name, _, b_sha) in zip(marks, marks[1:]):
        same = a_sha == b_sha
        links.append({
            "from": a_name,
            "to": b_name,
            "same_commit": same,
            "ordered_by_the_graph": (not same) and _is_ancestor(a_sha, b_sha),
            "how": ("both artifacts entered the history in one commit, so the "
                    "graph does not order them"
                    if same else
                    f"{a_sha[:12]} is an ancestor of {b_sha[:12]}"),
        })

    # The plan pins the set it was written against, and the read pins the plan.
    # Checking those digests against the blobs as committed is what stops a
    # later edit of either from passing as the thing that was preregistered.
    plan = json.loads((ROOT / STAGES[1][1]).read_text())
    read = json.loads((ROOT / STAGES[3][1]).read_text())
    set_at_freeze = _blob_sha256(marks[0][2], STAGES[0][1])
    plan_at_commit = _blob_sha256(marks[1][2], STAGES[1][1])
    # The plan records the set under the_set.sha256 and the read under
    # set_sha256. Reaching for the wrong one of those made this gate report that
    # the plan had failed to pin the set, when the two digests were equal all
    # along; a gate that names a field that does not exist reads as evidence of
    # a problem it invented.
    pins = {
        "plan_pins_the_set_as_frozen":
            plan.get("the_set", {}).get("sha256") == set_at_freeze,
        "read_pins_the_set_as_frozen": read.get("set_sha256") == set_at_freeze,
        "read_pins_the_plan_as_committed":
            read.get("plan_sha256") == plan_at_commit,
        "read_names_the_plan_commit": read.get("plan", {}).get(
            "committed_in") == marks[1][2],
    }

    unproven = [x for x in links if not x["ordered_by_the_graph"]]
    return {
        "schema": "geoaudit.external_order.v1",
        "clinical_grade": False,
        "head": head,
        "stages": [{"stage": n, "artifact": r, "introduced_in": s,
                    "at": _git("log", "-1", "--format=%cI", s)}
                   for n, r, s in marks],
        "links": links,
        "digest_pins": pins,
        "n_links_the_graph_orders": len(links) - len(unproven),
        "n_links_it_does_not": len(unproven),
        "what_the_graph_does_not_settle": [
            f"{x['from']} -> {x['to']}" for x in unproven],
        "how_the_remaining_link_is_established": (
            "the predictions and the read entered the history together, so "
            "their order is not a property of the graph. What stands in its "
            "place is recomputation: tools/external_read.py --check derives the "
            "frozen read from those exact prediction files and reproduces it "
            "byte for byte, so the read is a function of those inputs and of "
            "nothing else. This is weaker than a timestamp in one respect -- it "
            "cannot exclude the predictions having been produced and inspected "
            "before the read was run -- and stronger in another, since a "
            "timestamp would not show the read used those inputs at all."
            if unproven else
            "every link is ordered by the commit graph"),
        "verdict": (
            "confirmatory: the set was frozen and the plan committed before "
            "anything was scored, and both are pinned by digest"
            if all(pins.values())
               and links[0]["ordered_by_the_graph"]
               and links[1]["ordered_by_the_graph"]
            else "the order does not support a confirmatory reading"),
    }


def _report(d: dict) -> None:
    for s in d["stages"]:
        print(f"  {s['stage']:<22} {s['introduced_in'][:12]}  {s['at']}")
    for x in d["links"]:
        mark = "ordered" if x["ordered_by_the_graph"] else "NOT ordered"
        print(f"  {x['from']} -> {x['to']}: {mark} ({x['how']})")
    bad = [k for k, v in d["digest_pins"].items() if not v]
    print(f"  digest pins: {len(d['digest_pins']) - len(bad)}/"
          f"{len(d['digest_pins'])} hold" + (f", failing {bad}" if bad else ""))
    print(f"  {d['verdict']}")


def check() -> int:
    got = build()
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    have = json.loads(OUT.read_text())
    volatile = {"head", "stages"}
    moved = [k for k in got if k not in volatile and got[k] != have.get(k)]
    if moved:
        print(f"FAILED: the recorded order no longer matches the history: "
              f"{moved}")
        for k in moved:
            print(f"  - {k}\n      recorded: {json.dumps(have.get(k))[:200]}"
                  f"\n      history:  {json.dumps(got[k])[:200]}")
        return 1
    bad = [k for k, v in got["digest_pins"].items() if not v]
    if bad:
        print(f"FAILED: {bad} -- an artifact does not pin the version of its "
              f"input that was actually committed")
        return 1
    if not got["verdict"].startswith("confirmatory"):
        print(f"FAILED: {got['verdict']}")
        return 1
    _report(got)
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    d = build()
    OUT.write_text(json.dumps(d, indent=2) + "\n")
    _report(d)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
