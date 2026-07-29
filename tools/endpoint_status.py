#!/usr/bin/env python3
"""Which comparison in this paper is confirmatory, and which is exploratory.

The manuscript has been reporting a preregistered trimmed mean as the one
surviving resolved claim. The preregistration is real: the statistic was chosen
on the training partition, committed, and the reading tool refuses to run unless
that commit is an ancestor of HEAD. What the manuscript did not say is what the
preregistration was preregistered *relative to*.

By the time the statistic was fixed the held-out fold had already been read
several times, most of them by this architecture's own lineage; the exact counts
are recomputed here rather than quoted, because a rebase can change them. So the
commit order licenses one thing and not another. It licenses the claim that the
summary functional was not chosen after seeing what it would give on the fold. It
does not license the claim that the architecture being summarised was chosen
without the fold in the loop, because it was not: the lineage's previous readings
are in the ledger and they informed what came next.

A preregistration that fixes the statistic but not the object being measured is
worth reporting and is not worth calling confirmatory. This file derives that
distinction from the commit graph rather than asserting it, names the mean as the
primary endpoint on the grounds that it is the summary the field was compared
under from the first reading, and demotes the trimmed mean and its companions to
exploratory.

It reads no fold. Every number below already appears in an artifact two earlier
reads committed; the only thing computed here is which commit came before which.

Usage: PYTHONPATH=src:tools python3.12 tools/endpoint_status.py [--check]
"""
from __future__ import annotations

import argparse
import json
import subprocess

from pocket_bench.paths import ROOT

LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
PREREG_READ = ROOT / "results/official_fold/PREREGISTERED_READ.json"
OUT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
SCHEMA = "geoaudit.endpoint_status.v1"


def _first_commit(path: str) -> str | None:
    """The commit that introduced a file, or None if it is not committed."""
    p = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "-1", "--", path],
        cwd=ROOT, capture_output=True, text=True)
    return (p.stdout.strip() or None) if p.returncode == 0 else None


def _is_ancestor(a: str, b: str) -> bool:
    """Whether commit ``a`` is an ancestor of commit ``b``.

    Topological rather than by timestamp: a rebase or a corrected clock can
    reorder dates, and the question here is genuinely about what was knowable
    when, which is what reachability answers.
    """
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                          cwd=ROOT, capture_output=True).returncode == 0


def build() -> dict:
    ledger = json.loads(LEDGER.read_text())
    read = json.loads(PREREG_READ.read_text())
    prereg_commit = read["provenance_of_the_choice"]["committed_in"]

    before, after, uncommitted = [], [], []
    for e in ledger["indexed_read_sequence"]:
        art = e["artifact"]
        c = _first_commit(art)
        row = {"read_index": e["read_index"], "artifact": art,
               "kind": e["kind"], "method": e["method"],
               "mean_residue_auc": e.get("mean_residue_auc"),
               "first_committed_in": c[:12] if c else None}
        if c is None:
            uncommitted.append(row)
        elif _is_ancestor(c, prereg_commit):
            before.append(row)
        else:
            after.append(row)
    if uncommitted:
        raise SystemExit(
            f"{len(uncommitted)} indexed read artifacts are not in the commit "
            f"graph, so their order relative to the preregistration cannot be "
            f"checked: {[r['artifact'] for r in uncommitted]}")

    pre = read["preregistered_result"]
    mean = read["mean_reported_beside_it"]
    shape = read["shape_of_the_differences"]
    others = [c for c in read["candidates"]
              if c.get("statistic") not in (pre["statistic"], "mean")]

    def _ep(src, status, why):
        return {
            "statistic": src["statistic"],
            "delta": src["point"],
            "ci95": [src["ci_low"], src["ci_high"]],
            "p_two_sided_bootstrap": src.get("p_two_sided_bootstrap"),
            "resolves": not src["crosses_zero"],
            "status": status,
            "why": why,
        }

    n_before = len(before)
    lineage = [r for r in before if "table field" in (r["method"] or "")]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "question": "which reported comparison against P2Rank is confirmatory "
                    "and which is exploratory, decided by what the "
                    "preregistration was preregistered relative to",
        "primary_endpoint": _ep(
            mean, "primary",
            "the mean per-residue ROC-AUC is the summary this comparison was "
            "reported under from the first reading of the fold, before any "
            "statistic was chosen for it. Nothing had to be preregistered for "
            "it to be the default, which is exactly what makes it the endpoint "
            "a reader can hold the paper to"),
        "exploratory_endpoints": [
            _ep(pre, "exploratory",
                f"chosen on the training partition and committed before the "
                f"read that used it, but after {n_before} indexed readings of "
                f"the held-out fold, {len(lineage)} of them by this "
                f"architecture's own lineage. The commit order rules out "
                f"choosing the functional to fit the answer; it does not rule "
                f"out the fold having shaped the architecture the functional "
                f"is applied to")
        ] + [
            _ep({"statistic": c["statistic"], "point": c.get("point"),
                 "ci_low": c.get("ci_low"), "ci_high": c.get("ci_high"),
                 "p_two_sided_bootstrap": c.get("p_two_sided_bootstrap"),
                 "crosses_zero": c.get("crosses_zero", True)},
                "exploratory",
                "reported alongside the preregistered statistic as a "
                "robustness companion, not selected in advance")
            for c in others if c.get("point") is not None
        ],
        "what_the_preregistration_licenses": (
            "that the summary functional was fixed before the reading that "
            "used it, checked by commit ancestry and not by assertion"),
        "what_it_does_not_license": (
            f"that the architecture was fixed before the fold was seen. It was "
            f"not: {n_before} indexed reads precede the preregistration commit, "
            f"and the immediately preceding reading of this lineage is in the "
            f"ledger. A preregistered statistic over an architecture chosen "
            f"with the fold in the loop is exploratory evidence"),
        "reads_before_the_preregistration": before,
        "reads_after_the_preregistration": after,
        "n_reads_before_the_preregistration": n_before,
        "preregistration_commit": prereg_commit,
        "per_chain_outcome": {
            "n_units": shape["n"],
            "n_field_ahead": shape["n_field_ahead"],
            "n_baseline_ahead": shape["n_baseline_ahead"],
            "median_difference": shape["quantiles"]["0.5"],
            "worst_losses_mean": shape["worst_losses_mean"],
            "best_wins_mean": shape["best_wins_mean"],
            "why_it_is_here": "a summary statistic is a claim about a "
                              "distribution, so the distribution is reported "
                              "beside it. The field wins more chains than it "
                              "loses and loses harder than it wins, and those "
                              "two facts together are the whole reason a "
                              "trimmed mean and a mean disagree",
        },
        "claim_the_paper_is_entitled_to": (
            "on the primary endpoint the table field and P2Rank are not "
            "separable on this fold. Robust summaries of the same paired "
            "differences favour the field and are reported as exploratory. The "
            "case for the method rests on its being exactly decomposable and a "
            "small integer artifact, not on a resolved accuracy advantage"),
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "every number here is copied from an artifact an earlier indexed "
            "read committed; nothing is rescored and no new inference is drawn "
            "from the fold. The only thing computed is which commit is an "
            "ancestor of which"),
    }


def _report(d: dict) -> None:
    p = d["primary_endpoint"]
    print(f"primary   {p['statistic']:<12s} {p['delta']:+.4f} "
          f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  "
          f"{'resolves' if p['resolves'] else 'does not resolve'}")
    for e in d["exploratory_endpoints"]:
        print(f"explor.   {e['statistic']:<12s} {e['delta']:+.4f} "
              f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  "
              f"{'resolves' if e['resolves'] else 'does not resolve'}")
    c = d["per_chain_outcome"]
    print(f"per chain {c['n_field_ahead']} ahead, {c['n_baseline_ahead']} "
          f"behind of {c['n_units']}; median {c['median_difference']:+.4f}, "
          f"worst losses {c['worst_losses_mean']:+.4f}, best wins "
          f"{c['best_wins_mean']:+.4f}")
    print(f"\n{d['n_reads_before_the_preregistration']} indexed reads precede "
          f"the preregistration commit "
          f"{d['preregistration_commit'][:12]}:")
    for r in d["reads_before_the_preregistration"]:
        auc = ("" if r["mean_residue_auc"] is None
               else f", mean AUC {r['mean_residue_auc']:.4f}")
        print(f"  read {r['read_index']}  {r['first_committed_in']}  "
              f"{r['method']}{auc}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") is not None:
        bad.append("this restatement must not claim a read index")
    # The demotion has to follow from the commit graph as it stands now, not
    # from the graph as it stood when the file was written. A rebase that moved
    # the preregistration earlier than the reads would change the conclusion.
    live = build()
    for key in ("n_reads_before_the_preregistration", "preregistration_commit"):
        if d.get(key) != live.get(key):
            bad.append(f"{key} is recorded as {d.get(key)} and the commit "
                       f"graph now gives {live.get(key)}")
    if d.get("n_reads_before_the_preregistration", 0) == 0:
        bad.append("no read precedes the preregistration, so the endpoint "
                   "demotion this artifact exists to justify no longer has a "
                   "reason; the manuscript can promote the statistic back")
    p = d.get("primary_endpoint") or {}
    if p.get("statistic") != "mean":
        bad.append(f"the primary endpoint is recorded as {p.get('statistic')}; "
                   f"the manuscript names the mean")
    if p.get("resolves"):
        bad.append("the primary endpoint now resolves, which is a different "
                   "paper from the one written around it; say so deliberately "
                   "rather than by regenerating this file")
    # An exploratory endpoint that quietly acquires primary status is the exact
    # failure this artifact exists to prevent.
    for e in d.get("exploratory_endpoints") or []:
        if e.get("status") != "exploratory":
            bad.append(f"{e.get('statistic')} is listed among the exploratory "
                       f"endpoints with status {e.get('status')}")
    c = d.get("per_chain_outcome") or {}
    if c.get("n_field_ahead", 0) + c.get("n_baseline_ahead", 0) != c.get(
            "n_units"):
        bad.append("the per-chain wins and losses do not add up to the units")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}: primary endpoint unresolved, "
          f"{len(d['exploratory_endpoints'])} exploratory, "
          f"{d['n_reads_before_the_preregistration']} reads before the "
          f"preregistration")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
