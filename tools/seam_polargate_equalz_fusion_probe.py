#!/usr/bin/env python3.12
"""Equal-z fusion: seam_geometry_field ⊕ train-free polar-gate residue score.

Does NOT recompile tables. The polar score is the sum of selected raw ranks
(charge density, polar patch, charged run) — a priori, no coefficient fit.

clinical_grade = false. Development fold.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/SEAM_POLARGATE_EQUALZ_FUSION.json"
ANTI = {"4j4e_F", "2vqz_F", "3jzg_A", "3ly8_A", "5i3t_E", "6ksc_A", "6bty_B"}


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def _z(v):
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 1e-12 else np.zeros_like(v)


def _polar_score(path, chain):
    from pathlib import Path

    from pocket_bench.methods.polar_gate import COLUMNS, compute
    from pocket_bench.pdb_io import parse_pdb_atoms

    SKIP = {"HOH", "WAT", "DOD"}
    atoms = parse_pdb_atoms(Path(path).read_text())
    poly = {}
    for a in atoms:
        if a["chain"] != chain or a["element"] == "H" or a["resname"] in SKIP:
            continue
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
    order = sorted(poly)
    X = compute([poly[k] for k in order], [k[0] for k in order])
    # a priori combination of the ranks that smoke-tested well on anti-ranks
    take = [
        COLUMNS.index("rank_charge_abs_sum"),
        COLUMNS.index("rank_polar_patch"),
        COLUMNS.index("rank_seq_charged_run"),
        COLUMNS.index("charge_abs_sum_8A"),
    ]
    score = X[:, take].sum(axis=1)
    return {order[i][0]: float(score[i]) for i in range(len(order))}


def one(e):
    from pocket_bench.methods import seam_geometry_field

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        a = seam_geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        polar = _polar_score(ROOT / e["receptor_path"], e["chain"])
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _pos(lab)
        res = sorted(int(r) for r in a["residue_scores"])
        av = np.array([float(a["residue_scores"][str(r)]) for r in res])
        pv = np.array([float(polar.get(r, 0.0)) for r in res])
        fused = _z(av) + _z(pv)
        y = np.array([1 if r in pos else 0 for r in res])
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "deg"}
        return unit, {
            "seam": float(roc_auc_score(y, av)),
            "polar": float(roc_auc_score(y, pv)),
            "fused": float(roc_auc_score(y, fused)),
            "n": len(y),
        }
    except Exception as ex:  # noqa: BLE001
        return unit, {"error": f"{type(ex).__name__}:{ex}"}


def main() -> int:
    man = json.loads(
        (ROOT / "data/cryptobench_apo/official_manifest.json").read_text()
    )
    plm = {
        u["unit_id"]: u
        for u in json.loads(
            (ROOT / "results/baselines/PLMNN_SCORES.json").read_text()
        )["units"]
    }
    t0 = time.perf_counter()
    got = {}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for i, fut in enumerate(
            as_completed([ex.submit(one, e) for e in man["entries"]]), 1
        ):
            u, r = fut.result()
            got[u] = r
            if i % 30 == 0:
                print(i, flush=True)

    rows = []
    for e in man["entries"]:
        unit = f"{e['pdb']}_{e['chain']}"
        g = got.get(unit, {})
        if "fused" not in g or unit not in plm:
            continue
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _pos(lab)
        ps = plm[unit]["scores"]
        res = sorted(int(k) for k in ps)
        y = np.array([1 if r in pos else 0 for r in res])
        s = np.array([float(ps[str(r)]) for r in res])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pauc = float(roc_auc_score(y, s))
        rows.append({**g, "unit": unit, "plm": pauc, "d": g["fused"] - pauc})

    d = np.array([r["d"] for r in rows])
    rng = np.random.default_rng(0)
    boots = [
        float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)
    ]
    anti = [r for r in rows if r["unit"] in ANTI]
    out = {
        "schema": "geoaudit.seam_polargate_equalz_fusion.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "what_this_is": (
            "equal-z fusion of seam_geometry scores with a train-free "
            "polar-gate rank sum; tables untouched"
        ),
        "n_units": len(rows),
        "mean_seam": float(np.mean([r["seam"] for r in rows])),
        "mean_polar": float(np.mean([r["polar"] for r in rows])),
        "mean_fused": float(np.mean([r["fused"] for r in rows])),
        "mean_plmnn": float(np.mean([r["plm"] for r in rows])),
        "mean_delta": float(d.mean()),
        "ci95": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "n_ahead": int((d > 0).sum()),
        "n_behind": int((d < 0).sum()),
        "anti_rank": {
            "mean_seam": float(np.mean([r["seam"] for r in anti])),
            "mean_polar": float(np.mean([r["polar"] for r in anti])),
            "mean_fused": float(np.mean([r["fused"] for r in anti])),
            "mean_plm": float(np.mean([r["plm"] for r in anti])),
            "per_unit": [
                {
                    "unit": r["unit"],
                    "seam": r["seam"],
                    "polar": r["polar"],
                    "fused": r["fused"],
                    "plm": r["plm"],
                }
                for r in anti
            ],
        },
        "seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "anti_rank"}, indent=2))
    print("anti_rank", json.dumps(out["anti_rank"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
