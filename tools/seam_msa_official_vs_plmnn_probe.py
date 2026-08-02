#!/usr/bin/env python3.12
"""Development probe: seam_msa_field vs pLM-NN on the official fold.

clinical_grade = false. Not confirmatory.
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
OUT = ROOT / "results/official_fold/SEAM_MSA_VS_PLMNN_PROBE.json"


def _positives(lab: dict) -> set[int]:
    if "cryptic_residues" in lab:
        return {int(x) for x in lab["cryptic_residues"]}
    return set()


def one(e: dict) -> tuple[str, dict]:
    from pocket_bench.methods import seam_msa_field

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        pred = seam_msa_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"],
            unit=unit,
        )
        rs = pred.get("residue_scores")
        if not isinstance(rs, dict) or not rs:
            return unit, {"error": f"{pred.get('status')}:{pred.get('error')}"}
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _positives(lab)
        resseq = [int(r) for r in rs]
        scores = np.asarray([float(rs[str(r)]) for r in resseq], dtype=float)
        y = np.asarray([1 if r in pos else 0 for r in resseq], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "degenerate"}
        return unit, {"auc": float(roc_auc_score(y, scores)),
                      "n": len(y), "n_pos": int(y.sum())}
    except Exception as ex:  # noqa: BLE001
        return unit, {"error": f"{type(ex).__name__}: {ex}"}


def main() -> int:
    man = json.loads((ROOT / "data/cryptobench_apo/official_manifest.json").read_text())
    plm = {u["unit_id"]: u for u in json.loads(
        (ROOT / "results/baselines/PLMNN_SCORES.json").read_text())["units"]}
    seam = {u["unit"]: u for u in json.loads(
        (ROOT / "results/official_fold/SEAM_GEOMETRY_VS_PLMNN_PROBE.json").read_text()
    )["per_unit"]}

    t0 = time.perf_counter()
    got = {}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(one, e) for e in man["entries"]]
        for i, fut in enumerate(as_completed(futs), 1):
            u, rec = fut.result()
            got[u] = rec
            if i % 20 == 0:
                print(f"{i}/192 {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for e in man["entries"]:
        unit = f"{e['pdb']}_{e['chain']}"
        g = got.get(unit, {})
        if "auc" not in g or unit not in plm or unit not in seam:
            continue
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _positives(lab)
        ps = plm[unit]["scores"]
        resseqs = sorted(int(k) for k in ps)
        y = np.asarray([1 if r in pos else 0 for r in resseqs])
        s = np.asarray([float(ps[str(r)]) for r in resseqs])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pauc = float(roc_auc_score(y, s))
        rows.append({
            "unit": unit,
            "seam_msa_field": g["auc"],
            "seam_geometry_field": seam[unit]["seam_geometry_field"],
            "geometry_field": seam[unit]["geometry_field"],
            "plmnn": pauc,
            "delta_msa_minus_plmnn": g["auc"] - pauc,
            "delta_msa_minus_seam": g["auc"] - seam[unit]["seam_geometry_field"],
            "n": g["n"], "n_pos": g["n_pos"],
        })

    d = np.asarray([r["delta_msa_minus_plmnn"] for r in rows])
    rng = np.random.default_rng(0)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out = {
        "schema": "geoaudit.seam_msa_vs_plmnn_official_probe.v1",
        "clinical_grade": False,
        "method": "seam_msa_field",
        "reads_test_fold": True,
        "why_not_confirmatory": (
            "official fold heavily read; MSA columns from Swiss-Prot mmseqs2 "
            "appended to seam_geometry under counting-field topology"
        ),
        "n_jobs": N_JOBS,
        "n_units_compared": len(rows),
        "n_errors": sum(1 for v in got.values() if "error" in v),
        "errors_sample": [f"{u}:{v['error']}" for u, v in got.items()
                          if "error" in v][:12],
        "mean_seam_msa_field": float(np.mean([r["seam_msa_field"] for r in rows])),
        "mean_seam_geometry_field": float(np.mean([r["seam_geometry_field"]
                                                   for r in rows])),
        "mean_plmnn": float(np.mean([r["plmnn"] for r in rows])),
        "mean_delta_msa_minus_plmnn": float(d.mean()),
        "ci95_delta_msa_minus_plmnn": ci,
        "n_msa_ahead_of_plmnn": int((d > 0).sum()),
        "n_plmnn_ahead_of_msa": int((d < 0).sum()),
        "seconds": time.perf_counter() - t0,
        "per_unit": sorted(rows, key=lambda r: r["delta_msa_minus_plmnn"]),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", OUT)
    print(json.dumps({k: out[k] for k in out if k != "per_unit"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
