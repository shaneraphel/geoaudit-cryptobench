#!/usr/bin/env python3
"""The fifth indexed reading of the held-out fold, under a statistic fixed first.

What makes this read different from the four before it
-------------------------------------------------------
The functional was chosen in ``tools/preregister_statistic.py`` on the training
partition and committed before this file existed. That ordering is the whole
content of the claim, so it is not asserted here, it is checked: this tool
refuses to run unless the preregistration artifact is tracked by git, clean in
the working tree, and introduced by a commit that is an ancestor of HEAD. The
commit hash goes in the output.

What is and is not being re-scored
-----------------------------------
Nothing is re-scored. The per-unit ROC-AUCs of the table field and of P2Rank on
the official fold are already frozen in the telemetry and were produced by reads
already in the ledger. This tool takes a different summary of numbers that
already exist. That is not a new scoring of the fold, but it is a new
inferential use of it, and had the functional been picked after seeing them it
would be exactly the multiplicity this repository exists to exclude. So it is
indexed as a read anyway; the ledger is meant to over-count rather than
under-count.

All six candidates are reported, not just the preregistered one
----------------------------------------------------------------
Reporting only the chosen statistic would hide whether the choice happened to
land on the flattering one. The preregistered statistic is marked, and the mean
that the literature reports is kept beside it, but every candidate the training
partition considered is shown with its interval.

Usage:
    PYTHONPATH=src:tools python3.12 tools/preregistered_read.py
    PYTHONPATH=src:tools python3.12 tools/preregistered_read.py --check
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

from pocket_bench.paths import ROOT

from preregister_statistic import (
    CLAIMS, STATISTICS, OUT as PREREG_PATH,
)

TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
WIDE_TEST = ROOT / "data/cryptobench_apo/_wide_cache_test.npz"
OUT = ROOT / "results/official_fold/PREREGISTERED_READ.json"

SCHEMA = "geoaudit.preregistered_read.v1"
METHOD = "table_field"
BASELINE = "p2rank"
N_BOOT = 10000
SEED = 20260725
READ_INDEX = 5


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def preregistration_precedes_this_read() -> dict:
    """Refuse to read the fold unless the choice is already in the history."""
    rel = str(PREREG_PATH.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the commit that fixed the statistic "
            "may not be present and the ordering cannot be checked. Fetch the "
            "full history (actions/checkout with fetch-depth: 0) and retry.")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is modified or untracked. A preregistration that can still "
            f"be edited is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(
            f"{rel} has no commit. The choice of statistic has to be in the "
            f"history before the fold is read for it; this read is refused.")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, head],
                      cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    return {
        "artifact": rel,
        "committed_in": sha,
        "committed_at": _git("log", "-1", "--format=%cI", sha),
        "subject": _git("log", "-1", "--format=%s", sha),
        "clean_in_working_tree": True,
        "is_ancestor_of_head": True,
        "checked_not_asserted": (
            "this tool exits non-zero if the artifact is dirty, uncommitted, "
            "or not an ancestor of HEAD"),
    }


def paired_values() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = json.loads(TELEMETRY.read_text())["rows"]
    by = {m: {} for m in (METHOD, BASELINE)}
    for r in rows:
        if r["method"] in by and r.get("residue_auc") is not None:
            by[r["method"]][r["unit_id"]] = float(r["residue_auc"])
    shared = sorted(set(by[METHOD]) & set(by[BASELINE]))
    a = np.array([by[METHOD][u] for u in shared])
    b = np.array([by[BASELINE][u] for u in shared])
    return a, b, shared


def chain_lengths(units: list[str]) -> dict[str, int]:
    """Scored chain lengths, taken from the feature cache when it is present
    and from the artifact's own copy when it is not.

    Lengths are not labels or scores, so reading them is not a look at the
    fold's answers, but the cache they live in is too large to commit. Freezing
    them into the artifact is what lets the read recompute from committed data
    alone, which is the difference between a result and a claim about one.
    """
    if WIDE_TEST.exists():
        z = np.load(WIDE_TEST, allow_pickle=False)
        by = {str(u): int(n) for u, n in zip(z["units"], z["n_res_per"])}
        if all(u in by for u in units):
            return {u: by[u] for u in units}
    if OUT.exists():
        frozen = json.loads(OUT.read_text()).get("chain_lengths") or {}
        if all(u in frozen for u in units):
            return {u: int(frozen[u]) for u in units}
    raise SystemExit(
        "chain lengths are available from neither the feature cache nor a "
        "previous artifact, so the length-stratified candidate cannot be "
        "recomputed")


def strata_for(units: list[str]) -> np.ndarray:
    """Chain length quartiles."""
    by = chain_lengths(units)
    L = np.array([by[u] for u in units], dtype=np.float64)
    return np.digitize(L, np.quantile(L, [0.25, 0.5, 0.75]))


def evaluate(d: np.ndarray, strata: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    prereg = json.loads(PREREG_PATH.read_text())["preregistered"]["statistic"]
    out = []
    for name, fn in STATISTICS.items():
        point = float(fn(d[None, :], strata)[0])
        boots = np.sort(fn(d[idx], strata))
        lo, hi = float(boots[int(0.025 * N_BOOT)]), \
            float(boots[int(0.975 * N_BOOT) - 1])
        frac = float((boots >= 0).mean())
        out.append({
            "statistic": name,
            "claim": CLAIMS[name],
            "preregistered": name == prereg,
            "point": round(point, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "p_two_sided_bootstrap": round(2 * min(frac, 1 - frac), 4),
            "crosses_zero": bool(lo <= 0 <= hi),
        })
    return out


def shape_of_the_differences(d: np.ndarray) -> dict:
    """What trimming removes, stated rather than left for a referee to ask.

    A trimmed mean that clears zero while the mean does not is only interesting
    if the tails it discards are a minority behaving differently from the bulk,
    and it is misleading if they are the failures the method should be judged
    on. Both readings are available from the same numbers, so both are recorded:
    how much of the fold sits on each side, and how big the discarded tail is.
    """
    k = int(round(0.20 * len(d)))
    s = np.sort(d)
    lo, hi = s[:k], s[len(d) - k:]
    return {
        "n": int(len(d)),
        "n_trimmed_each_side": k,
        "quantiles": {q: round(float(np.quantile(d, float(q))), 6)
                      for q in ("0.05", "0.1", "0.25", "0.5", "0.75", "0.9",
                                "0.95")},
        "n_field_ahead": int((d > 0).sum()),
        "n_baseline_ahead": int((d < 0).sum()),
        "worst_losses_mean": round(float(lo.mean()), 6),
        "best_wins_mean": round(float(hi.mean()), 6),
        "mean_of_the_middle_60_percent": round(float(s[k:len(d) - k].mean()), 6),
        "how_to_read_it": (
            "the trimmed statistic says the field leads on the bulk of the "
            "fold and says nothing about the tails, which is why the mean, "
            "which does speak to them, is reported unresolved beside it. "
            "Neither number alone is the whole comparison"),
    }


def build() -> dict:
    prov = preregistration_precedes_this_read()
    prereg = json.loads(PREREG_PATH.read_text())
    a, b, units = paired_values()
    d = a - b
    strata = strata_for(units)
    rows = evaluate(d, strata)
    chosen = next(r for r in rows if r["preregistered"])
    mean_row = next(r for r in rows if r["statistic"] == "mean")
    fore = prereg["forecast"]

    print(f"{len(units)} paired units on the official fold, read index "
          f"{READ_INDEX}")
    print(f"  {METHOD:12s} {a.mean():.4f}")
    print(f"  {BASELINE:12s} {b.mean():.4f}\n")
    for r in rows:
        mark = "<- preregistered" if r["preregistered"] else ""
        print(f"  {r['statistic']:22s} {r['point']:+.4f}  "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
              f"p={r['p_two_sided_bootstrap']:.3f}  {mark}")

    resolved = not chosen["crosses_zero"]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "question": "under the functional fixed on the training partition, is "
                    "the table field separable from P2Rank on the held-out "
                    "fold",
        "provenance_of_the_choice": prov,
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "the per-unit numbers were already frozen by earlier reads, so no "
            "receptor was scored again, but taking a new summary of them is a "
            "new inferential use of the fold and the ledger over-counts by "
            "design"),
        "telemetry_source": str(TELEMETRY.relative_to(ROOT)),
        "method": METHOD,
        "baseline": BASELINE,
        "n_paired_units": len(units),
        "chain_lengths": chain_lengths(units),
        "mean_method": round(float(a.mean()), 6),
        "mean_baseline": round(float(b.mean()), 6),
        "n_boot": N_BOOT,
        "seed": SEED,
        "candidates": rows,
        "shape_of_the_differences": shape_of_the_differences(d),
        "preregistered_result": chosen,
        "mean_reported_beside_it": mean_row,
        "forecast_made_before_the_read": fore,
        "forecast_vs_outcome": {
            "expected_power": fore["expected_power"],
            "resolved": resolved,
            "forecast_direction_held": bool(
                resolved == (fore["expected_power"] >= 0.5)),
            "why_it_missed": (
                "the forecast shifted every difference by one constant, so it "
                "modelled a held-out margin smaller than the pick half's by "
                "moving the whole distribution down. What the fold actually "
                "shows is a bulk that leads by more than the mean does and a "
                "tail that drags the mean back, which a location shift cannot "
                "represent. The forecast was pessimistic for the trimmed "
                "statistic and is recorded unedited"
            ) if resolved and fore["expected_power"] < 0.5 else None,
        },
        "outcome": (
            f"the preregistered statistic gives {chosen['point']:+.4f} "
            f"[{chosen['ci_low']:+.4f}, {chosen['ci_high']:+.4f}], which "
            f"{'excludes' if resolved else 'contains'} zero"),
    }


def _verdict(doc: dict) -> str:
    c = doc["preregistered_result"]
    m = doc["mean_reported_beside_it"]
    if not c["crosses_zero"]:
        return (f"resolved: {c['claim']} {c['point']:+.4f} "
                f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]. The mean, which "
                f"the training partition predicted would not resolve, gives "
                f"{m['point']:+.4f} [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}].")
    return (f"not resolved: the preregistered statistic gives {c['point']:+.4f} "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], and the forecast made "
            f"before the read put the chance of clearing zero at "
            f"{doc['forecast_made_before_the_read']['expected_power']:.0%}.")


def audit() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    doc = json.loads(OUT.read_text())
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append(f"schema is {doc.get('schema')!r}")
    if doc.get("test_fold_read_index") != READ_INDEX:
        bad.append("the read is not indexed where the ledger expects it")
    prov = doc.get("provenance_of_the_choice") or {}
    if not prov.get("committed_in"):
        bad.append("no commit is recorded for the preregistration")
    else:
        live = json.loads(PREREG_PATH.read_text())["preregistered"]["statistic"]
        chosen = doc.get("preregistered_result", {}).get("statistic")
        if chosen != live:
            bad.append(f"the read reports {chosen!r} as preregistered but the "
                       f"artifact now names {live!r}: one of them has moved")
    if doc.get("rescored_anything") is not False:
        bad.append("the read claims to have re-scored the fold")
    marked = [c for c in doc.get("candidates", []) if c["preregistered"]]
    if len(marked) != 1:
        bad.append(f"{len(marked)} candidates are marked preregistered")
    if not any(c["statistic"] == "mean" for c in doc.get("candidates", [])):
        bad.append("the mean is not reported beside the chosen statistic")
    for c in doc.get("candidates", []):
        if c["crosses_zero"] != (c["ci_low"] <= 0 <= c["ci_high"]):
            bad.append(f"{c['statistic']}: crosses_zero disagrees with its own "
                       f"interval")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: read {READ_INDEX}, choice fixed in "
          f"{prov['committed_in'][:12]}")
    print(f"  {_verdict(doc)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return audit()
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n{_verdict(doc)}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
