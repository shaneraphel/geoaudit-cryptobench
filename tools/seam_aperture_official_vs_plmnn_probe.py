#!/usr/bin/env python3.12
"""Official-fold probe: seam_aperture_field vs pLM-NN.

Development read; fold heavily used. clinical_grade = false.
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
OUT = ROOT / "results/official_fold/SEAM_APERTURE_VS_PLMNN_PROBE.json"
SUMMARY = ROOT / "results/official_fold/SEAM_APERTURE_VS_PLMNN_PROBE_SUMMARY.json"


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def one(e):
    from pocket_bench.methods import (
        geometry_field,
        seam_aperture_field,
        seam_geometry_field,
    )

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        a = seam_aperture_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        s = seam_geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        g = geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _pos(lab)
        res = sorted(int(r) for r in a["residue_scores"])
        av = np.array([float(a["residue_scores"][str(r)]) for r in res])
        sv = np.array([float(s["residue_scores"][str(r)]) for r in res])
        gv = np.array([float(g["residue_scores"][str(r)]) for r in res])
        y = np.array([1 if r in pos else 0 for r in res])
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "degenerate"}
        return unit, {
            "seam_aperture_field": float(roc_auc_score(y, av)),
            "seam_geometry_field": float(roc_auc_score(y, sv)),
            "geometry_field": float(roc_auc_score(y, gv)),
            "n_pos": int(y.sum()),
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
        futs = [ex.submit(one, e) for e in man["entries"]]
        for i, fut in enumerate(as_completed(futs), 1):
            u, r = fut.result()
            got[u] = r
            if i % 30 == 0:
                print(i, flush=True)

    rows = []
    for e in man["entries"]:
        unit = f"{e['pdb']}_{e['chain']}"
        g = got.get(unit, {})
        if "seam_aperture_field" not in g or unit not in plm:
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
        rows.append(
            {
                "unit": unit,
                **{k: g[k] for k in (
                    "seam_aperture_field",
                    "seam_geometry_field",
                    "geometry_field",
                    "n_pos",
                    "n",
                )},
                "plmnn": pauc,
                "delta_aperture_minus_plmnn": g["seam_aperture_field"] - pauc,
                "delta_aperture_minus_seam": (
                    g["seam_aperture_field"] - g["seam_geometry_field"]
                ),
            }
        )

    d = np.array([r["delta_aperture_minus_plmnn"] for r in rows])
    rng = np.random.default_rng(0)
    boots = [
        float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)
    ]
    out = {
        "schema": "geoaudit.seam_aperture_vs_plmnn_probe.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "why_not_confirmatory": "development probe; fold heavily read",
        "n_units": len(rows),
        "mean_seam_aperture_field": float(
            np.mean([r["seam_aperture_field"] for r in rows])
        ),
        "mean_seam_geometry_field": float(
            np.mean([r["seam_geometry_field"] for r in rows])
        ),
        "mean_geometry_field": float(
            np.mean([r["geometry_field"] for r in rows])
        ),
        "mean_plmnn": float(np.mean([r["plmnn"] for r in rows])),
        "mean_delta_aperture_minus_plmnn": float(d.mean()),
        "ci95": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "n_ahead": int((d > 0).sum()),
        "n_behind": int((d < 0).sum()),
        "mean_delta_aperture_minus_seam": float(
            np.mean([r["delta_aperture_minus_seam"] for r in rows])
        ),
        "seconds": time.perf_counter() - t0,
        "per_unit": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    summary = {k: v for k, v in out.items() if k != "per_unit"}
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
