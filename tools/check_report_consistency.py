#!/usr/bin/env python3
"""Every number that appears in two frozen artifacts must be the same number.

Frozen reports are written by different tools at different times, and nothing
structural stops two of them from disagreeing about the same quantity. This
repository shipped two such disagreements at once:

  * ``OFFICIAL_FOLD_METRICS.json`` declared ``p2rank: DATA_UNAVAILABLE`` while
    ``OFFICIAL_MULTI_METHOD_BOOTSTRAP.json`` in the same directory reported a
    P2Rank ROC-AUC over 192 units;
  * the same detector had two different ROC-AUCs in the two files, because one
    was frozen before a parser fix and the other after.

Neither is caught by a unit test, because each file is internally consistent.
They are only caught by comparing files, which is what this does. Every check
is an equality between two things that are definitionally the same quantity:

  1. a bootstrap point estimate == the mean of the per-unit telemetry values it
     was computed from;
  2. a method's point estimate in one report == its point estimate in another;
  3. a declared baseline availability == whether that method actually has rows;
  4. the unit set is identical across every method and every report.

Exit 0 if all hold, 1 otherwise. Intended for `make verify` and CI.

Usage: PYTHONPATH=src python3.12 tools/check_report_consistency.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pocket_bench.paths import ROOT

TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
MULTI = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP.json"
VS_P2RANK = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
FOLD = ROOT / "results/official_fold/OFFICIAL_FOLD_METRICS.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"

TOL = 5e-4          # reports round; a real disagreement is orders of magnitude larger
METRICS = ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1")
# the two reports name the same four quantities differently
FOLD_METRIC = {"residue_auc": "roc_auc", "residue_pr_auc": "pr_auc",
               "residue_mcc": "mcc", "residue_f1": "f1"}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def eq(self, what: str, a: float | None, b: float | None,
           tol: float = TOL) -> None:
        self.checks += 1
        if a is None and b is None:
            return
        if a is None or b is None:
            self.failures.append(f"{what}: {a!r} vs {b!r} (one is null)")
            return
        if abs(a - b) > tol:
            self.failures.append(f"{what}: {a:.6f} vs {b:.6f} "
                                 f"(differ by {abs(a - b):.6f})")

    def same(self, what: str, a: object, b: object) -> None:
        self.checks += 1
        if a != b:
            self.failures.append(f"{what}: {a!r} != {b!r}")


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    rep = Report()

    if not TELEMETRY.exists():
        print(f"missing {TELEMETRY.relative_to(ROOT)}; nothing to cross-check")
        return 1
    rows = json.loads(TELEMETRY.read_text())["rows"]
    methods = sorted({r["method"] for r in rows})
    units = {r["unit_id"] for r in rows}

    # 4. one unit set, shared by every method
    for m in methods:
        u = {r["unit_id"] for r in rows if r["method"] == m}
        rep.same(f"unit set of {m}", u, units)

    per_unit: dict[str, dict[str, list[float | None]]] = {
        met: {m: [r[met] for r in rows if r["method"] == m] for m in methods}
        for met in METRICS
    }

    # 1. + 2. bootstrap points against the telemetry they summarise
    for path in (MULTI, VS_P2RANK):
        if not path.exists():
            rep.failures.append(f"missing report {path.relative_to(ROOT)}")
            continue
        doc = json.loads(path.read_text())["metrics"]
        tag = path.name
        for met in METRICS:
            if met not in doc:
                rep.failures.append(f"{tag}: metric {met} absent")
                continue
            for m, entry in doc[met]["per_method"].items():
                rep.eq(f"{tag} {met} {m} point vs telemetry mean",
                       entry.get("point"), _mean(per_unit[met].get(m, [])))

    # 2. the two reports against each other, method by method
    if MULTI.exists() and VS_P2RANK.exists():
        a = json.loads(MULTI.read_text())["metrics"]
        b = json.loads(VS_P2RANK.read_text())["metrics"]
        for met in METRICS:
            if met not in a or met not in b:
                continue
            shared = set(a[met]["per_method"]) & set(b[met]["per_method"])
            for m in sorted(shared):
                rep.eq(f"MULTI vs PAIRED {met} {m}",
                       a[met]["per_method"][m].get("point"),
                       b[met]["per_method"][m].get("point"))

    # 3. declared availability against rows that actually exist
    if FOLD.exists():
        fold = json.loads(FOLD.read_text())
        status = fold.get("real_ml_baseline_status") or {}
        for name in ("p2rank", "pocketminer"):
            scored = sum(1 for r in rows
                         if r["method"] == name and r["residue_auc"] is not None)
            declared = status.get(name)
            expect = "AVAILABLE" if scored else "DATA_UNAVAILABLE"
            rep.same(f"OFFICIAL_FOLD_METRICS declares {name}", declared, expect)

        # the same detector, evaluated by two independent runners
        for met in METRICS:
            boot = (fold.get("bootstrap") or {}).get(FOLD_METRIC[met], {})
            entry = (boot.get("per_method") or {}).get("geometric_foundation")
            if entry is None:
                continue
            rep.eq(f"OFFICIAL_FOLD_METRICS {FOLD_METRIC[met]} "
                   f"geometric_foundation vs telemetry",
                   entry.get("point"),
                   _mean(per_unit[met].get("geometric_foundation", [])))

        rep.same("OFFICIAL_FOLD_METRICS n_structures_scored",
                 fold.get("n_structures_scored"), len(units))

    if PER_STRUCTURE.exists() and FOLD.exists():
        ps = json.loads(PER_STRUCTURE.read_text())
        rep.same("PER_STRUCTURE unit count", len(ps), len(units))
        rep.eq("PER_STRUCTURE geometric_foundation roc_auc mean vs telemetry",
               _mean([r["geometric_foundation"]["roc_auc"] for r in ps]),
               _mean(per_unit["residue_auc"].get("geometric_foundation", [])))

    print(f"cross-artifact checks run: {rep.checks}")
    if rep.failures:
        print(f"INCONSISTENT ({len(rep.failures)}):")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print("all frozen artifacts agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
