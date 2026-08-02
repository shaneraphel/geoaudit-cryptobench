#!/usr/bin/env python3.12
"""Covariates of geometry_field − pLM-NN delta on the official fold.

Development diagnostic. clinical_grade = false.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
OUT = ROOT / "results/official_fold/GEOMETRY_PLM_DELTA_COVARIATES.json"


def _one(e: dict) -> tuple[str, dict | None]:
    from pocket_bench.methods.algebraic_descriptors import algebraic_residue_features
    from pocket_bench.paths import ROOT as R

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        resseq, _F, _codes, ctr = algebraic_residue_features(
            R / e["receptor_path"], chain=e["chain"]
        )
        lab = json.loads((R / e["label_path"]).read_text())
        pos = {int(x) for x in lab.get("cryptic_residues") or []}
        ctr = np.asarray(ctr, dtype=float)
        com = ctr.mean(axis=0)
        rg = float(np.sqrt(((ctr - com) ** 2).sum(axis=1).mean()))
        d = np.linalg.norm(ctr - com, axis=1)
        res_to_i = {int(r): i for i, r in enumerate(resseq)}
        pos_i = [res_to_i[r] for r in pos if r in res_to_i]
        if not pos_i:
            return unit, None
        pos_d = d[pos_i]
        order = np.sort(d)
        pct = [float(np.searchsorted(order, x) / len(d)) for x in pos_d]
        return unit, {
            "n": int(len(resseq)),
            "n_pos": int(len(pos_i)),
            "rg": rg,
            "pos_mean_radial_pct": float(np.mean(pct)),
            "pos_surface_frac": float((pos_d >= np.percentile(d, 75)).mean()),
            "pos_core_frac": float((pos_d <= np.percentile(d, 25)).mean()),
        }
    except Exception as ex:  # noqa: BLE001
        return unit, {"error": f"{type(ex).__name__}: {ex}"}


def main() -> int:
    geo = json.loads(
        (ROOT / "results/official_fold/GEOMETRY_FIELD_VS_PLMNN_PROBE.json").read_text()
    )
    by = {r["unit"]: r for r in geo["units_sorted_by_delta"]}
    man = json.loads((ROOT / "data/cryptobench_apo/official_manifest.json").read_text())
    rows = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_one, e) for e in man["entries"]]
        for fut in as_completed(futs):
            unit, s = fut.result()
            if not s or "error" in s:
                continue
            rows.append({**by[unit], **s})

    dlt = np.array([r["delta"] for r in rows])
    n = np.array([r["n"] for r in rows], dtype=float)
    surf = np.array([r["pos_surface_frac"] for r in rows])
    core = np.array([r["pos_core_frac"] for r in rows])
    rad = np.array([r["pos_mean_radial_pct"] for r in rows])
    bins = []
    for lo, hi, name in ((0, 150, "n<150"), (150, 300, "150-300"), (300, 1e9, "n>=300")):
        m = (n >= lo) & (n < hi)
        bins.append({
            "bin": name,
            "n_units": int(m.sum()),
            "mean_delta": float(dlt[m].mean()) if m.any() else None,
            "mean_geometry": float(np.mean([r["geometry_field"] for r, keep in zip(rows, m) if keep])) if m.any() else None,
            "mean_plmnn": float(np.mean([r["plmnn"] for r, keep in zip(rows, m) if keep])) if m.any() else None,
        })
    out = {
        "schema": "geoaudit.geometry_plm_delta_covariates.v1",
        "clinical_grade": False,
        "reads_test_fold": True,
        "why_not_confirmatory": "diagnostic on a heavily read fold",
        "n_units": len(rows),
        "n_jobs": N_JOBS,
        "correlations": {
            "delta_vs_n": float(np.corrcoef(dlt, n)[0, 1]),
            "delta_vs_pos_surface_frac": float(np.corrcoef(dlt, surf)[0, 1]),
            "delta_vs_pos_core_frac": float(np.corrcoef(dlt, core)[0, 1]),
            "delta_vs_pos_radial_pct": float(np.corrcoef(dlt, rad)[0, 1]),
        },
        "length_bins": bins,
        "units": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "units"}, indent=2))
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
