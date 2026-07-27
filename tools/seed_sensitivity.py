#!/usr/bin/env python3
"""Does a verdict survive the resampling seed, or is it a property of the draw?

The paired MCC difference against P2Rank excludes zero at the pre-registered
seed by four ten-thousandths. A bound that thin is not a finding about the fold;
it is a finding about one pseudo-random sequence, and reporting it as
"significant" would be the most ordinary way to overstate a result. The only
honest thing to do is to resample the resampling.

For each metric the paired difference is formed on the structures where both
methods have a value -- for MCC that is fewer than all of them, since MCC is
undefined wherever a confusion matrix is degenerate -- and the bootstrap is then
repeated under many independent seeds. What is reported is how often the 95%
interval excludes zero. A verdict that holds under every seed is a property of
the data; one that holds under half of them is a coin flip wearing a p-value.

The frozen report quotes these counts, and the manuscript reports MCC as
unresolved on the strength of them even though the pre-registered seed alone
would have licensed the opposite claim.

Usage: PYTHONPATH=src python3.12 tools/seed_sensitivity.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
METRICS = ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1")
METHOD = "table_field"
BASELINE = "p2rank"
N_BOOT = 10000
N_SEEDS = 25
BASE_SEED = 20260725


def main() -> int:
    tel = json.loads(TELEMETRY.read_text())
    rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["unit_id"]] = r
    ours, theirs = by[METHOD], by[BASELINE]
    units = sorted(set(ours) & set(theirs))

    seeds = [BASE_SEED + 1000 * k for k in range(N_SEEDS)]
    report = {
        "schema": "geoaudit.seed_sensitivity.v1",
        "clinical_grade": False,
        "question": "does the paired verdict against P2Rank survive the "
                    "resampling seed?",
        "method": METHOD, "baseline": BASELINE,
        "n_boot": N_BOOT, "n_seeds": N_SEEDS, "seeds": seeds,
        "ci_level": 0.95,
        "pairing": "structures where both methods have a value",
        "metrics": {},
    }

    for metric in METRICS:
        d = np.array([ours[u][metric] - theirs[u][metric] for u in units
                      if ours[u].get(metric) is not None
                      and theirs[u].get(metric) is not None])
        n = len(d)
        excl = 0
        los, his = [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            means = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            los.append(float(lo)); his.append(float(hi))
            if lo > 0 or hi < 0:
                excl += 1
        report["metrics"][metric] = {
            "n_paired_structures": n,
            "paired_difference": float(d.mean()),
            "n_seeds_excluding_zero": excl,
            "fraction_of_seeds": excl / len(seeds),
            "ci_low_range": [min(los), max(los)],
            "ci_high_range": [min(his), max(his)],
            "verdict": ("robust" if excl == len(seeds)
                        else "seed-dependent" if excl else "unresolved"),
        }
        print(f"  {metric:15s} n={n:3d}  Δ={d.mean():+.4f}  excludes zero in "
              f"{excl:2d}/{len(seeds)} seeds  "
              f"(lower bound {min(los):+.4f} to {max(los):+.4f})  "
              f"{report['metrics'][metric]['verdict']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
