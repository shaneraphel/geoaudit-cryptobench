#!/usr/bin/env python3.12
"""Official-fold probe: seam_polargate_field vs pLM-NN. clinical_grade=false."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/SEAM_POLARGATE_VS_PLMNN_PROBE.json"
SUMMARY = ROOT / "results/official_fold/SEAM_POLARGATE_VS_PLMNN_PROBE_SUMMARY.json"

# Units where seam catastrophically anti-ranks; track separately.
ANTI = {"4j4e_F", "2vqz_F", "3jzg_A", "3ly8_A", "5i3t_E", "6ksc_A", "6bty_B"}


def _pos(lab):
    return {int(x) for x in lab.get("cryptic_residues", [])}


def one(e):
    from pocket_bench.methods import seam_geometry_field, seam_polargate_field

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        a = seam_polargate_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        s = seam_geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _pos(lab)
        res = sorted(int(r) for r in a["residue_scores"])
        av = np.array([float(a["residue_scores"][str(r)]) for r in res])
        sv = np.array([float(s["residue_scores"][str(r)]) for r in res])
        y = np.array([1 if r in pos else 0 for r in res])
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "degenerate"}
        return unit, {
            "seam_polargate_field": float(roc_auc_score(y, av)),
            "seam_geometry_field": float(roc_auc_score(y, sv)),
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
        if "seam_polargate_field" not in g or unit not in plm:
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
                "seam_polargate_field": g["seam_polargate_field"],
                "seam_geometry_field": g["seam_geometry_field"],
                "plmnn": pauc,
                "n": g["n"],
                "n_pos": g["n_pos"],
                "delta_pg_minus_plmnn": g["seam_polargate_field"] - pauc,
                "delta_pg_minus_seam": (
                    g["seam_polargate_field"] - g["seam_geometry_field"]
                ),
            }
        )

    d = np.array([r["delta_pg_minus_plmnn"] for r in rows])
    rng = np.random.default_rng(0)
    boots = [
        float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)
    ]
    anti = [r for r in rows if r["unit"] in ANTI]
    out = {
        "schema": "geoaudit.seam_polargate_vs_plmnn_probe.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "why_not_confirmatory": "development probe; fold heavily read",
        "n_units": len(rows),
        "mean_seam_polargate_field": float(
            np.mean([r["seam_polargate_field"] for r in rows])
        ),
        "mean_seam_geometry_field": float(
            np.mean([r["seam_geometry_field"] for r in rows])
        ),
        "mean_plmnn": float(np.mean([r["plmnn"] for r in rows])),
        "mean_delta_pg_minus_plmnn": float(d.mean()),
        "ci95": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "n_ahead": int((d > 0).sum()),
        "n_behind": int((d < 0).sum()),
        "mean_delta_pg_minus_seam": float(
            np.mean([r["delta_pg_minus_seam"] for r in rows])
        ),
        "anti_rank_subset": {
            "units": sorted(ANTI),
            "mean_seam": float(np.mean([r["seam_geometry_field"] for r in anti])),
            "mean_polargate": float(
                np.mean([r["seam_polargate_field"] for r in anti])
            ),
            "mean_plmnn": float(np.mean([r["plmnn"] for r in anti])),
            "per_unit": anti,
        },
        "seconds": time.perf_counter() - t0,
        "per_unit": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    summary = {k: v for k, v in out.items() if k != "per_unit"}
    # keep anti per_unit in summary
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    slim = {k: v for k, v in summary.items() if k != "anti_rank_subset"}
    slim["anti_rank_means"] = {
        k: summary["anti_rank_subset"][k]
        for k in ("mean_seam", "mean_polargate", "mean_plmnn")
    }
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
