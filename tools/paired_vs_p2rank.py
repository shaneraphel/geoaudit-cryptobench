#!/usr/bin/env python3
"""Paired bootstrap of every detector against P2Rank on the official test fold.

The frozen multi-method report uses the geometric foundation as its baseline,
which answers "is the new field better than our own earlier detector". The
question a reader actually has is the other one: how does a counting-only field
stand against the trained random forest that defines the state of the art on
this benchmark. That is a different baseline, so it is a different report, and
it is computed here rather than being asserted anywhere in prose.

The resample is paired: one set of structure indices is drawn per iteration and
every method is averaged over the same draw, so the difference is estimated on
the same structures and the CI reflects the variability of the DIFFERENCE, not
the sum of two independent variabilities.

The comparison is honest in both directions. Where the CI of the difference
excludes zero, P2Rank is ahead and the report says so; where it crosses zero the
two are statistically indistinguishable on that metric, which is a weaker claim
than "matches" and is the only one supported.

Usage: PYTHONPATH=src python3.12 tools/paired_vs_p2rank.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocket_bench.metrics_bootstrap import paired_bootstrap, per_structure_values
from pocket_bench.paths import ROOT

TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT = ROOT / "results/official_fold/PAIRED_VS_P2RANK.json"
METRICS = ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry", type=Path, default=TELEMETRY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--baseline", default="p2rank")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260725)
    args = ap.parse_args(argv)

    rows = json.loads(args.telemetry.read_text())["rows"]
    methods = sorted({r.get("method") for r in rows})
    if args.baseline not in methods:
        raise SystemExit(
            f"baseline {args.baseline!r} absent from {args.telemetry.name}; "
            f"present: {methods}. Refusing to compare against nothing."
        )

    report = {}
    for met in METRICS:
        report[met] = paired_bootstrap(
            per_structure_values(rows, met), baseline=args.baseline,
            n_boot=args.n_boot, seed=args.seed,
        )

    n_units = len({r.get("unit_id") for r in rows})
    payload = {
        "schema": "geoaudit.paired_vs_p2rank.v1",
        "clinical_grade": False,
        "question": "On the official fold, how does each detector stand against "
                    "the trained random-forest baseline?",
        "benchmark": "CryptoBench official MMseqs2 10% cluster-disjoint TEST fold "
                     "(Skrhak et al., Bioinformatics 2025)",
        "baseline": args.baseline,
        "n_units": n_units,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "telemetry_source": str(args.telemetry.relative_to(ROOT)),
        "reading": "delta = method - baseline. crosses_zero=true means the two "
                   "are statistically indistinguishable on that metric at the "
                   "95% level; it does NOT mean they are equal.",
        "metrics": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False is the gate, not a formatting preference: the file is only
    # written if it is strict JSON.
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(f"=== paired vs {args.baseline}  (n={n_units}, "
          f"boot={args.n_boot}, 95% CI) ===")
    for met in METRICS:
        print(f"\n{met}")
        pm = report[met]["per_method"]
        for m, d in sorted(report[met]["paired_vs_baseline"].items(),
                           key=lambda kv: -(kv[1].get("delta_point") or -9)):
            if d.get("delta_point") is None:
                print(f"  {m:26s} (no comparable values)")
                continue
            verdict = "ns" if d["crosses_zero"] else "significant"
            print(f"  {m:26s} {pm[m]['point']:.4f}   "
                  f"d={d['delta_point']:+.4f} "
                  f"[{d['delta_ci_low']:+.4f}, {d['delta_ci_high']:+.4f}]  "
                  f"{verdict}")
    print(f"\n-> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
