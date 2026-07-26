#!/usr/bin/env python3
"""Freeze a paired bootstrap over the official fold against a chosen baseline.

Two baselines are frozen, because they answer two different questions and a
paper needs both:

  geometric_foundation   is a detector better than the simplest thing we built?
  p2rank                 is it separable from the trained random forest that
                         defines the published baseline on this benchmark?

Both artifacts come from this one script and the same telemetry, so every point
estimate they share is identical by construction rather than by coincidence;
``tools/check_report_consistency.py`` then re-checks that they are. The previous
arrangement had one artifact with no generator in the repository at all, which
meant the numbers a reader was asked to trust could not be recomputed from the
inputs -- the reproducibility hole that this script closes.

The design is paired: each resample draws the same units for every method, so
the correlation between two detectors looking at the same pocket cancels out of
the difference instead of inflating its variance.

Usage:
  PYTHONPATH=src python3.12 tools/freeze_bootstrap.py --baseline geometric_foundation
  PYTHONPATH=src python3.12 tools/freeze_bootstrap.py --baseline p2rank
  PYTHONPATH=src python3.12 tools/freeze_bootstrap.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocket_bench.metrics_bootstrap import (
    json_safe,
    paired_bootstrap,
    per_structure_values,
)
from pocket_bench.paths import ROOT

TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT_DIR = ROOT / "results/official_fold"
N_BOOT = 10000
SEED = 20260725
METRICS = ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1")

TARGETS = {
    "geometric_foundation": (
        OUT_DIR / "OFFICIAL_MULTI_METHOD_BOOTSTRAP.json",
        "Is each detector separable from the simplest geometric baseline on "
        "the official fold?"),
    "p2rank": (
        OUT_DIR / "OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json",
        "Is an algebraic field separable from the trained random-forest "
        "baseline on the official fold?"),
}


def freeze(baseline: str, out: Path, question: str, *, quiet: bool = False) -> dict:
    telem = json.loads(TELEMETRY.read_text())
    rows = telem["rows"]
    units = sorted({r["unit_id"] for r in rows})
    scored = sum(1 for r in rows
                 if r["method"] == baseline and r["residue_auc"] is not None)
    if not scored:
        raise SystemExit(
            f"{baseline} has no scored rows in {TELEMETRY.relative_to(ROOT)}; "
            f"refusing to freeze a comparison against an absent baseline")

    report = {
        "schema": "geoaudit.bootstrap_report.v1",
        "clinical_grade": False,
        "question": question,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "telemetry_source": str(TELEMETRY.relative_to(ROOT)),
        "baseline": baseline,
        "baseline_scored_units": scored,
        "n_structures": len(units),
        "n_boot": N_BOOT,
        "seed": SEED,
        "ci_level": 0.95,
        "metrics": {},
    }
    for metric in METRICS:
        report["metrics"][metric] = paired_bootstrap(
            per_structure_values(rows, metric),
            baseline=baseline, n_boot=N_BOOT, seed=SEED, ci=0.95)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  "
          f"({len(units)} units, baseline={baseline}, n_boot={N_BOOT})")
    if not quiet:
        for metric in METRICS:
            for m, d in sorted(report["metrics"][metric]["paired_vs_baseline"].items()):
                if d.get("delta_point") is None:
                    continue
                verdict = "indistinguishable" if d["crosses_zero"] else "separated"
                print(f"  {metric:16s} {m:26s} {d['delta_point']:+.4f} "
                      f"[{d['delta_ci_low']:+.4f}, {d['delta_ci_high']:+.4f}] "
                      f"{verdict}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=sorted(TARGETS))
    ap.add_argument("--all", action="store_true",
                    help="freeze every baseline in TARGETS")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.all and not args.baseline:
        ap.error("give --baseline or --all")

    names = sorted(TARGETS) if args.all else [args.baseline]
    for name in names:
        out, question = TARGETS[name]
        freeze(name, out, question, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
