"""A/B controlled experiment: pure geometry vs geometry + sequence wires.

Emits the contention census for both compiled fields and the frozen metric table,
formatted for the README.

Why two X-state numbers are reported
------------------------------------
Track A addresses 4^6 = 4096 cells and Track B addresses 4^10 = 1048576. Comparing
"X as a fraction of asserted cells" across the two is confounded: widening the
word spreads the same residues over 256x more addresses, so cells get sparser and
fewer of them can hold both classes even if nothing was actually resolved. The
per-cell number is therefore reported alongside the per-RESIDUE number -- the
fraction of test residues whose address lands in a contended cell -- which is
occupancy-weighted and cannot be moved by empty address space alone. Read the
per-residue column for the physics; the per-cell column only describes the table.

Usage: PYTHONPATH=src python3.12 tools/dual_track_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/cryptobench_apo"
RESULTS = ROOT / "results/cryptobench_official"
FIELDS = {"A": DATA / "RESOLUTION_FIELD.json", "B": DATA / "RESOLUTION_FIELD_B.json"}
METHOD_OF_TRACK = {"A": "quaternary_lut", "B": "quaternary_lut_seq"}
OUT = RESULTS / "DUAL_TRACK_AB.json"


def cell_census(track: str) -> dict:
    d = json.loads(FIELDS[track].read_text())
    asserted = int(d["n_cells_asserted"])
    return {
        "track": track,
        "n_wires": int(d["n_features"]),
        "n_cells": int(d["n_cells"]),
        "n_cells_asserted": asserted,
        "n_cells_X": int(d["n_cells_X"]),
        "n_cells_0": int(d["n_cells_0"]),
        "n_cells_1": int(d["n_cells_1"]),
        "n_cells_Z": int(d["n_cells_Z"]),
        "pct_asserted_X": 100.0 * d["n_cells_X"] / max(asserted, 1),
        "feature_names": d["feature_names"],
        "n_training_residues": d.get("n_training_residues"),
    }


def residue_census(rows: list[dict], method: str) -> dict:
    """Occupancy-weighted contention: test residues landing in an X or Z cell."""
    sel = [r for r in rows if r.get("method") == method]
    n_res = sum(int(r.get("n_residues") or 0) for r in sel)
    n_x = sum(int(r.get("n_cells_hit_X") or 0) for r in sel)
    n_z = sum(int(r.get("n_cells_hit_Z") or 0) for r in sel)
    return {
        "n_units": len(sel),
        "n_residues": n_res,
        "n_residues_in_X": n_x,
        "n_residues_in_Z": n_z,
        "pct_residues_X": 100.0 * n_x / max(n_res, 1),
        "pct_residues_Z": 100.0 * n_z / max(n_res, 1),
    }


def _fmt(entry: dict | None) -> str:
    if not entry or entry.get("point") is None:
        return "`null`"
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if lo is None or hi is None:
        return f"{entry['point']:.3f}"
    return f"{entry['point']:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    telemetry = json.loads((RESULTS / "TELEMETRY.json").read_text())
    boot = json.loads((RESULTS / "BOOTSTRAP_CI.json").read_text())
    rows = telemetry["rows"]
    auc = boot["metrics"]["residue_auc"]["per_method"]
    pr = boot["metrics"]["residue_pr_auc"]["per_method"]
    mcc = boot["metrics"]["residue_mcc"]["per_method"]
    f1 = boot["metrics"]["residue_f1"]["per_method"]

    report = {
        "schema": "geoaudit.dual_track_ab.v1",
        "clinical_grade": False,
        "question": "does residue identity resolve the contention that pure "
                    "geometry leaves in the quaternary address space",
        "tracks": {},
    }
    for track in ("A", "B"):
        m = METHOD_OF_TRACK[track]
        report["tracks"][track] = {
            "method": m,
            "cells": cell_census(track),
            "residues": residue_census(rows, m),
            "residue_auc": auc.get(m),
        }

    print("## A/B controlled experiment: does sequence resolve geometric contention\n")
    print("Same 770 training units, same geometry, same compiler. Track B appends "
          "four sequence wires.\n")
    print("| | wires | address cells | asserted | X cells | X of asserted | "
          "test residues in X |")
    print("|---|---|---|---|---|---|---|")
    for track in ("A", "B"):
        c = report["tracks"][track]["cells"]
        r = report["tracks"][track]["residues"]
        label = "A pure geometry" if track == "A" else "B geometry + sequence"
        print(f"| `{label}` | {c['n_wires']} | {c['n_cells']:,} | "
              f"{c['n_cells_asserted']:,} | {c['n_cells_X']:,} | "
              f"{c['pct_asserted_X']:.1f}% | {r['pct_residues_X']:.1f}% |")

    print("\n### Frozen metrics, official CryptoBench test fold (n=192)\n")
    print("| method | ROC-AUC | PR-AUC | MCC | F1 |")
    print("|---|---|---|---|---|")
    order = sorted(auc, key=lambda m: -(auc[m].get("point") or 0.0))
    for m in order:
        p = mcc.get(m, {}).get("point")
        f = f1.get(m, {}).get("point")
        print(f"| `{m}` | {_fmt(auc.get(m))} | "
              f"{(pr.get(m) or {}).get('point', float('nan')):.3f} | "
              f"{'`null`' if p is None else f'{p:.3f}'} | "
              f"{'`null`' if f is None else f'{f:.3f}'} |")

    paired = boot["metrics"]["residue_auc"].get("paired_vs_baseline", {})
    if paired:
        print(f"\nPaired deltas vs `{boot.get('baseline')}` "
              f"({boot['metrics']['residue_auc'].get('n_boot', boot.get('n_boot'))} "
              f"bootstrap resamples):\n")
        print("| method | delta ROC-AUC | 95% CI | excludes 0 |")
        print("|---|---|---|---|")
        for m, d in sorted(paired.items(), key=lambda kv: -kv[1]["delta_point"]):
            print(f"| `{m}` | {d['delta_point']:+.3f} | "
                  f"[{d['delta_ci_low']:+.3f}, {d['delta_ci_high']:+.3f}] | "
                  f"{'yes' if not d['crosses_zero'] else 'no'} |")

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
