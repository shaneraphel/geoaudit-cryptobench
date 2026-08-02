#!/usr/bin/env python3.12
"""Residue-level rescue: keep seam, replace with polar where seam is low & polar high.

A priori rules (no coefficient fit on the fold):
  R1: score = max(z_seam, z_polar)
  R2: score = z_seam; where z_seam < 0 and z_polar > 1, use z_polar
  R3: score = z_seam + relu(z_polar - z_seam)   # polar only lifts, never pulls down

clinical_grade = false.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/SEAM_POLAR_RESCUE.json"
ANTI = {"4j4e_F", "2vqz_F", "3jzg_A", "3ly8_A", "5i3t_E", "6ksc_A", "6bty_B"}


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def _z(v):
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 1e-12 else np.zeros_like(v)


def _polar_map(path, chain):
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
        polar = _polar_map(ROOT / e["receptor_path"], e["chain"])
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _pos(lab)
        res = sorted(int(r) for r in a["residue_scores"])
        av = np.array([float(a["residue_scores"][str(r)]) for r in res])
        pv = np.array([float(polar.get(r, 0.0)) for r in res])
        zs, zp = _z(av), _z(pv)
        r1 = np.maximum(zs, zp)
        r2 = np.where((zs < 0) & (zp > 1.0), zp, zs)
        r3 = zs + np.maximum(zp - zs, 0.0)
        y = np.array([1 if r in pos else 0 for r in res])
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "deg"}
        return unit, {
            "seam": float(roc_auc_score(y, av)),
            "polar": float(roc_auc_score(y, pv)),
            "max": float(roc_auc_score(y, r1)),
            "rescue": float(roc_auc_score(y, r2)),
            "relu_lift": float(roc_auc_score(y, r3)),
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
        if "seam" not in g or unit not in plm:
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
        rows.append({**g, "unit": unit, "plm": pauc})

    def pack(key):
        d = np.array([r[key] - r["plm"] for r in rows])
        rng = np.random.default_rng(0)
        boots = [
            float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)
        ]
        anti = [r for r in rows if r["unit"] in ANTI]
        return {
            "mean": float(np.mean([r[key] for r in rows])),
            "mean_delta_vs_plm": float(d.mean()),
            "ci95": [
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ],
            "n_ahead": int((d > 0).sum()),
            "n_behind": int((d < 0).sum()),
            "anti_mean": float(np.mean([r[key] for r in anti])),
        }

    out = {
        "schema": "geoaudit.seam_polar_rescue.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "n_units": len(rows),
        "mean_plmnn": float(np.mean([r["plm"] for r in rows])),
        "seam": pack("seam"),
        "polar": pack("polar"),
        "max_z": pack("max"),
        "rescue_lowseam_highpolar": pack("rescue"),
        "relu_lift": pack("relu_lift"),
        "seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
