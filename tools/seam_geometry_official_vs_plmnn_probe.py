#!/usr/bin/env python3.12
"""Development probe: seam_geometry_field vs cached pLM-NN on the official fold.

Official fold has been read many times — not confirmatory.
Targets the short-chain / buried-cryptic deficit. clinical_grade = false.
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
OUT = ROOT / "results/official_fold/SEAM_GEOMETRY_VS_PLMNN_PROBE.json"


def _positives(lab: dict) -> set[int]:
    if "cryptic_residues" in lab:
        return {int(x) for x in lab["cryptic_residues"]}
    if "labels" in lab and isinstance(lab["labels"], dict):
        return {int(k) for k, v in lab["labels"].items() if v}
    for key in ("positive_resseq", "binding_residues", "pocket_residues"):
        if key in lab:
            return {int(x) for x in lab[key]}
    return set()


def one(e: dict) -> tuple[str, dict]:
    from pocket_bench.methods import seam_geometry_field

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        pred = seam_geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        rs = pred.get("residue_scores")
        if not isinstance(rs, dict) or not rs:
            return unit, {"error": f"missing scores; status={pred.get('status')} "
                                   f"err={pred.get('error')}"}
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _positives(lab)
        resseq = [int(r) for r in rs]
        scores = np.asarray([float(rs[str(r)]) for r in resseq], dtype=float)
        y = np.asarray([1 if r in pos else 0 for r in resseq], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "degenerate labels"}
        return unit, {"auc": float(roc_auc_score(y, scores)),
                      "n": len(y), "n_pos": int(y.sum())}
    except Exception as ex:  # noqa: BLE001
        return unit, {"error": f"{type(ex).__name__}: {ex}"}


def main() -> int:
    man = json.loads((ROOT / "data/cryptobench_apo/official_manifest.json").read_text())
    entries = man["entries"]
    plm = json.loads((ROOT / "results/baselines/PLMNN_SCORES.json").read_text())
    plm_by = {u["unit_id"]: u for u in plm["units"]}
    geo = json.loads(
        (ROOT / "results/official_fold/GEOMETRY_FIELD_VS_PLMNN_PROBE.json").read_text()
    )
    geo_by = {u["unit"]: u for u in geo["per_unit"]}

    t0 = time.perf_counter()
    got: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(one, e) for e in entries]
        for i, fut in enumerate(as_completed(futs), 1):
            unit, rec = fut.result()
            got[unit] = rec
            if i % 10 == 0:
                print(f"{i}/{len(entries)} {time.perf_counter() - t0:.0f}s",
                      flush=True)

    rows = []
    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        g = got.get(unit, {})
        if "auc" not in g:
            continue
        pu = plm_by.get(unit)
        gu = geo_by.get(unit)
        if not pu or not gu:
            continue
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _positives(lab)
        pscores = pu["scores"]
        resseqs = sorted(int(k) for k in pscores)
        y = np.asarray([1 if r in pos else 0 for r in resseqs])
        s = np.asarray([float(pscores[str(r)]) for r in resseqs])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pauc = float(roc_auc_score(y, s))
        rows.append({
            "unit": unit,
            "seam_geometry_field": g["auc"],
            "geometry_field": gu["geometry_field"],
            "plmnn": pauc,
            "delta_seam_minus_plmnn": g["auc"] - pauc,
            "delta_seam_minus_geometry": g["auc"] - gu["geometry_field"],
            "n_pos": g["n_pos"],
            "n": g["n"],
        })

    d = np.asarray([r["delta_seam_minus_plmnn"] for r in rows], dtype=float)
    dg = np.asarray([r["delta_seam_minus_geometry"] for r in rows], dtype=float)
    ci = None
    if len(d) >= 8:
        rng = np.random.default_rng(0)
        boots = [float(d[rng.integers(0, len(d), len(d))].mean())
                 for _ in range(4000)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    short = [r for r in rows if r["n"] < 150]
    out = {
        "schema": "geoaudit.seam_geometry_vs_plmnn_official_probe.v1",
        "clinical_grade": False,
        "method": "seam_geometry_field",
        "reads_test_fold": True,
        "why_not_confirmatory": (
            "official fold heavily read; development probe of nonlocal-seam "
            "columns appended to geometry_field under the same table topology"
        ),
        "n_jobs": N_JOBS,
        "n_units_compared": len(rows),
        "n_errors": sum(1 for v in got.values() if "error" in v),
        "errors_sample": [f"{u}:{v['error']}" for u, v in got.items()
                          if "error" in v][:12],
        "mean_seam_geometry_field": float(np.mean([r["seam_geometry_field"]
                                                   for r in rows])),
        "mean_geometry_field": float(np.mean([r["geometry_field"]
                                              for r in rows])),
        "mean_plmnn": float(np.mean([r["plmnn"] for r in rows])),
        "mean_delta_seam_minus_plmnn": float(d.mean()) if len(d) else None,
        "mean_delta_seam_minus_geometry": float(dg.mean()) if len(dg) else None,
        "ci95_delta_seam_minus_plmnn": ci,
        "n_seam_ahead_of_plmnn": int((d > 0).sum()),
        "n_plmnn_ahead_of_seam": int((d < 0).sum()),
        "short_chain_n_lt_150": {
            "n_units": len(short),
            "mean_seam": float(np.mean([r["seam_geometry_field"]
                                        for r in short])) if short else None,
            "mean_geometry": float(np.mean([r["geometry_field"]
                                            for r in short])) if short else None,
            "mean_plmnn": float(np.mean([r["plmnn"] for r in short])) if short else None,
            "mean_delta_seam_minus_plmnn": float(np.mean([
                r["delta_seam_minus_plmnn"] for r in short])) if short else None,
        },
        "seconds": time.perf_counter() - t0,
        "per_unit": sorted(rows, key=lambda r: r["delta_seam_minus_plmnn"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", OUT)
    summary = {k: out[k] for k in out if k != "per_unit"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
