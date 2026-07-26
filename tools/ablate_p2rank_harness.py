#!/usr/bin/env python3
"""Measure what the superseded P2Rank harness cost, instead of remembering it.

An earlier revision scored P2Rank by reconstructing a residue signal from pocket
centres: keep the top five pockets, take every residue within 6 A of a centre,
weight it 1/rank, tie everything else at zero. That is not the prediction P2Rank
makes -- it emits a calibrated per-residue probability over the whole chain --
and the substitution depressed its ROC-AUC by more than a tenth, which is large
enough to have inverted the paper's conclusion.

The manuscript states that cost. A stated number needs an artifact behind it, so
this recomputes BOTH scorings from the archived raw predictions and writes them
side by side. Nothing is re-executed: the same stored P2Rank output feeds both
paths, so the difference isolates the harness and nothing else.

Requires results/cryptobench_official/predictions/p2rank.json, written by the
runner. Usage: PYTHONPATH=src python3.12 tools/ablate_p2rank_harness.py
"""
from __future__ import annotations

import json

from pocket_bench.metrics import average_precision, roc_auc
from pocket_bench.paths import ROOT

PREDS = ROOT / "results/cryptobench_official/predictions/p2rank.json"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/official_fold/P2RANK_HARNESS_ABLATION.json"
TOP_K = 5
BALL = 6.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    if not PREDS.exists():
        raise SystemExit(
            f"missing {PREDS.relative_to(ROOT)}; run\n"
            f"  PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py "
            f"--dataset official --merge --resume --methods p2rank")
    from pocket_bench.pdb_io import parse_pdb_atoms

    stored = json.loads(PREDS.read_text())["units"]
    entries = json.loads(MANIFEST.read_text())["entries"]

    native_auc, native_pr, legacy_auc, legacy_pr = [], [], [], []
    n = 0
    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        pred = stored.get(unit)
        if not pred:
            continue
        label = json.loads((ROOT / e["label_path"]).read_text())
        truth = {int(r) for r in (label.get("cryptic_residues") or [])}
        if not truth:
            continue
        atoms = [a for a in parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
                 if a["record"] == "ATOM" and a["chain"] == e["chain"]]
        universe = sorted({int(a["resseq"]) for a in atoms})
        if not universe or not (truth & set(universe)):
            continue
        y = [1 if r in truth else 0 for r in universe]

        # the protocol in force: P2Rank's own calibrated per-residue column
        raw = pred.get("residue_scores") or {}
        s_native = [float(raw.get(str(r), 0.0)) for r in universe]

        # the superseded harness: top-k pocket centres, a ball of fixed radius
        # around each, weight 1/rank, everything outside tied at exactly zero
        by_res = {r: 0.0 for r in universe}
        for p in (pred.get("pockets") or [])[:TOP_K]:
            c = p.get("center_xyz")
            rank = int(p.get("rank", 10 ** 9))
            if not c or rank <= 0:
                continue
            w = 1.0 / rank
            for a in atoms:
                d2 = ((a["x"] - c[0]) ** 2 + (a["y"] - c[1]) ** 2
                      + (a["z"] - c[2]) ** 2)
                if d2 <= BALL * BALL:
                    rs = int(a["resseq"])
                    if w > by_res.get(rs, 0.0):
                        by_res[rs] = w
        s_legacy = [by_res[r] for r in universe]

        native_auc.append(roc_auc(s_native, y))
        native_pr.append(average_precision(s_native, y))
        legacy_auc.append(roc_auc(s_legacy, y))
        legacy_pr.append(average_precision(s_legacy, y))
        n += 1

    if not n:
        raise SystemExit("no units could be scored from the archived predictions")

    report = {
        "schema": "geoaudit.p2rank_harness_ablation.v1",
        "clinical_grade": False,
        "question": "How much of P2Rank's score did the superseded harness "
                    "destroy?",
        "source_predictions": str(PREDS.relative_to(ROOT)),
        "n_units": n,
        "native_residue_protocol": {
            "description": "P2Rank's own *_residues.csv probability column over "
                           "the whole chain",
            "roc_auc": _mean(native_auc),
            "pr_auc": _mean(native_pr),
        },
        "superseded_pocket_harness": {
            "description": f"top {TOP_K} pocket centres, {BALL} A ball, "
                           f"1/rank weight, everything else tied at 0",
            "roc_auc": _mean(legacy_auc),
            "pr_auc": _mean(legacy_pr),
        },
    }
    report["cost_of_the_harness"] = {
        "roc_auc": report["native_residue_protocol"]["roc_auc"]
        - report["superseded_pocket_harness"]["roc_auc"],
        "pr_auc": report["native_residue_protocol"]["pr_auc"]
        - report["superseded_pocket_harness"]["pr_auc"],
    }
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"n={n} units, both scorings from the same archived output")
    print(f"  native residue protocol   ROC-AUC "
          f"{report['native_residue_protocol']['roc_auc']:.4f}")
    print(f"  superseded pocket harness ROC-AUC "
          f"{report['superseded_pocket_harness']['roc_auc']:.4f}")
    print(f"  cost of the harness       "
          f"{report['cost_of_the_harness']['roc_auc']:+.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
