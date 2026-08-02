#!/usr/bin/env python3.12
"""How often is seam_geometry anti-ranked vs cryptic labels?

Writes an upper-bound oracle: max(seam, -seam) per unit. Not a detector.
clinical_grade = false. reads_test_fold = true (development fold).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/SEAM_INVERT_ORACLE.json"


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def one(e):
    from pocket_bench.methods import seam_geometry_field

    unit = f"{e['pdb']}_{e['chain']}"
    lab = json.loads((ROOT / e["label_path"]).read_text())
    pos = _pos(lab)
    pred = seam_geometry_field.predict(
        ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
    )
    res = sorted(int(r) for r in pred["residue_scores"])
    s = np.array([float(pred["residue_scores"][str(r)]) for r in res])
    y = np.array([1 if r in pos else 0 for r in res])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return {
        "unit": unit,
        "n": len(res),
        "npos": int(y.sum()),
        "corr": float(np.corrcoef(y, s)[0, 1]),
        "auc": float(roc_auc_score(y, s)),
        "auc_inv": float(roc_auc_score(y, -s)),
    }


def main() -> int:
    man = json.loads(
        (ROOT / "data/cryptobench_apo/official_manifest.json").read_text()
    )["entries"]
    plm = {
        u["unit_id"]: u
        for u in json.loads(
            (ROOT / "results/baselines/PLMNN_SCORES.json").read_text()
        )["units"]
    }
    rows = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(one, e) for e in man]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is None:
                continue
            unit = r["unit"]
            if unit not in plm:
                continue
            lab = json.loads(
                (ROOT / next(
                    e["label_path"] for e in man
                    if f"{e['pdb']}_{e['chain']}" == unit
                )).read_text()
            )
            pos = _pos(lab)
            ps = plm[unit]["scores"]
            res = sorted(int(k) for k in ps)
            y = np.array([1 if r in pos else 0 for r in res])
            s = np.array([float(ps[str(r)]) for r in res])
            if y.sum() == 0 or y.sum() == len(y):
                continue
            r["plm"] = float(roc_auc_score(y, s))
            rows.append(r)
            if i % 40 == 0:
                print(i, flush=True)

    neg = [r for r in rows if r["corr"] < 0]
    oracle = float(np.mean([max(r["auc"], r["auc_inv"]) for r in rows]))
    out = {
        "schema": "geoaudit.seam_invert_oracle.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "what_this_is": (
            "Upper bound if a detector could choose seam or -seam per unit. "
            "Not a method; diagnoses how much of the pLM gap is polarity."
        ),
        "n_units": len(rows),
        "n_neg_corr": len(neg),
        "mean_seam": float(np.mean([r["auc"] for r in rows])),
        "mean_inv": float(np.mean([r["auc_inv"] for r in rows])),
        "mean_oracle": oracle,
        "mean_plmnn": float(np.mean([r["plm"] for r in rows])),
        "mean_neg_corr_seam": float(np.mean([r["auc"] for r in neg])) if neg else None,
        "mean_neg_corr_inv": float(np.mean([r["auc_inv"] for r in neg])) if neg else None,
        "n_inv_beats_seam_by_0p05": int(
            sum(1 for r in rows if r["auc_inv"] - r["auc"] > 0.05)
        ),
        "n_inv_rescues_vs_plm": int(
            sum(1 for r in rows if r["auc"] < r["plm"] and r["auc_inv"] > r["plm"])
        ),
        "worst_neg_corr": sorted(neg, key=lambda r: r["corr"])[:12],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "worst_neg_corr"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
