#!/usr/bin/env python3.12
"""Can a training-free unit-level diagnostic predict seam polarity?

For each official-fold unit, seam may be correlated or anti-correlated with
cryptic labels. This screen asks whether a *label-free* scalar computed from the
receptor alone predicts which polarity wins — if yes, a switched detector can
be built without reading labels at test time.

clinical_grade = false. Development fold read.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/POLARITY_SWITCH_SCREEN.json"


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def one(e):
    from pocket_bench.methods import seam_geometry_field
    from pocket_bench.methods.cryptic_aperture import compute as ap_compute
    from pocket_bench.methods.nonlocal_seam import compute as seam_compute
    from pocket_bench.pdb_io import parse_pdb_atoms

    unit = f"{e['pdb']}_{e['chain']}"
    SKIP = {"HOH", "WAT", "DOD"}
    atoms = parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
    chain = e["chain"]
    poly = {}
    for a in atoms:
        if a["chain"] != chain or a["element"] == "H" or a["resname"] in SKIP:
            continue
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
    order = sorted(poly)
    if len(order) < 10:
        return None
    Xs = seam_compute([poly[k] for k in order], [k[0] for k in order])
    Xa = ap_compute([poly[k] for k in order], [k[0] for k in order])
    # Label-free unit diagnostics
    from pocket_bench.methods.nonlocal_seam import COLUMNS as SQ
    from pocket_bench.methods.cryptic_aperture import COLUMNS as AQ

    def col(X, names, name):
        return X[:, names.index(name)]

    hull = col(Xs, SQ, "on_convex_hull").mean()
    burial = col(Xs, SQ, "hull_depth").mean()
    core_seam = col(Xs, SQ, "core_seam_flag").mean()
    fiedler = col(Xs, SQ, "fiedler_rank_x100").std()
    aperture = col(Xa, AQ, "aperture_flag").mean()
    inv_pack = col(Xa, AQ, "inv_pack_6A_x100").mean()
    kink = col(Xa, AQ, "ca_kink_angle_x100").mean()
    n = len(order)

    pred = seam_geometry_field.predict(
        ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=chain
    )
    lab = json.loads((ROOT / e["label_path"]).read_text())
    pos = _pos(lab)
    res = sorted(int(r) for r in pred["residue_scores"])
    s = np.array([float(pred["residue_scores"][str(r)]) for r in res])
    y = np.array([1 if r in pos else 0 for r in res])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    auc = float(roc_auc_score(y, s))
    auc_inv = float(roc_auc_score(y, -s))
    want_inv = 1 if auc_inv > auc + 0.05 else 0
    return {
        "unit": unit,
        "n": n,
        "auc": auc,
        "auc_inv": auc_inv,
        "want_inv": want_inv,
        "corr": float(np.corrcoef(y, s)[0, 1]),
        "diag": {
            "n": n,
            "hull_frac": float(hull),
            "mean_hull_depth": float(burial),
            "core_seam_frac": float(core_seam),
            "fiedler_std": float(fiedler),
            "aperture_frac": float(aperture),
            "mean_inv_pack": float(inv_pack),
            "mean_kink": float(kink),
            "score_skew": float(
                ((s - s.mean()) ** 3).mean() / (s.std() ** 3 + 1e-12)
            ),
            "score_kurt": float(
                ((s - s.mean()) ** 4).mean() / (s.std() ** 4 + 1e-12)
            ),
            "top10_mass": float(np.sort(s)[-max(1, n // 10) :].sum() / (s.sum() + 1e-12)),
        },
    }


def main() -> int:
    man = json.loads(
        (ROOT / "data/cryptobench_apo/official_manifest.json").read_text()
    )["entries"]
    rows = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, e) for e in man]), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 40 == 0:
                print(i, flush=True)

    inv = [r for r in rows if r["want_inv"]]
    nor = [r for r in rows if not r["want_inv"]]
    keys = list(rows[0]["diag"].keys())
    sep = {}
    for k in keys:
        a = np.array([r["diag"][k] for r in inv])
        b = np.array([r["diag"][k] for r in nor])
        # simple separation: difference of means / pooled std
        pooled = float(np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))
        sep[k] = {
            "mean_want_inv": float(a.mean()),
            "mean_normal": float(b.mean()),
            "cohen_d": float((a.mean() - b.mean()) / pooled) if pooled else 0.0,
        }
    # Best single-threshold classifier by scanning each diagnostic
    best = {"key": None, "acc": 0.0, "thr": None, "direction": None}
    y = np.array([r["want_inv"] for r in rows])
    for k in keys:
        x = np.array([r["diag"][k] for r in rows])
        for thr in np.unique(np.percentile(x, np.linspace(5, 95, 19))):
            for direction in ("gt", "lt"):
                pred = (x > thr).astype(int) if direction == "gt" else (x < thr).astype(int)
                acc = float((pred == y).mean())
                # also require both classes predicted sometimes
                if pred.sum() == 0 or pred.sum() == len(pred):
                    continue
                if acc > best["acc"]:
                    best = {
                        "key": k,
                        "acc": acc,
                        "thr": float(thr),
                        "direction": direction,
                        "n_pred_inv": int(pred.sum()),
                        "precision": float(y[pred == 1].mean()) if pred.sum() else 0.0,
                        "recall": float(pred[y == 1].mean()) if y.sum() else 0.0,
                    }

    # Apply best rule → switched AUCs
    k, thr, direction = best["key"], best["thr"], best["direction"]
    switched = []
    for r in rows:
        flip = (
            r["diag"][k] > thr if direction == "gt" else r["diag"][k] < thr
        )
        switched.append(r["auc_inv"] if flip else r["auc"])

    out = {
        "schema": "geoaudit.polarity_switch_screen.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "n_units": len(rows),
        "n_want_inv": len(inv),
        "separation": sep,
        "best_threshold_rule": best,
        "mean_seam": float(np.mean([r["auc"] for r in rows])),
        "mean_switched": float(np.mean(switched)),
        "mean_oracle": float(np.mean([max(r["auc"], r["auc_inv"]) for r in rows])),
        "note": (
            "Threshold was fit on the same fold it is scored on — upper bound "
            "on a 1-D rule, not a confirmatory result. If mean_switched >> "
            "mean_seam, the polarity axis is worth a train-only gate."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
