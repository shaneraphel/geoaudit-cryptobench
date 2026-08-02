#!/usr/bin/env python3.12
"""Development probe: geometry_field vs cached pLM-NN on the official fold.

Official fold has been read many times — this is not confirmatory.
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
OUT = ROOT / "results/official_fold/GEOMETRY_FIELD_VS_PLMNN_PROBE.json"


def _positives(lab: dict) -> set[int]:
    if "cryptic_residues" in lab:
        return {int(x) for x in lab["cryptic_residues"]}
    if "labels" in lab and isinstance(lab["labels"], dict):
        return {int(k) for k, v in lab["labels"].items() if v}
    if "residue_labels" in lab:
        return {int(k) for k, v in lab["residue_labels"].items() if v}
    # CryptoBench style
    for key in ("positive_resseq", "binding_residues", "pocket_residues"):
        if key in lab:
            return {int(x) for x in lab[key]}
    return set()


def one(e: dict) -> tuple[str, dict]:
    from pocket_bench.methods import geometry_field
    from pocket_bench.paths import ROOT

    unit = f"{e['pdb']}_{e['chain']}"
    try:
        pred = geometry_field.predict(
            ROOT / e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"]
        )
        if not isinstance(pred, dict):
            return unit, {"error": f"unexpected type {type(pred)}"}
        rs = pred.get("residue_scores")
        if not isinstance(rs, dict) or not rs:
            return unit, {"error": f"missing residue_scores; keys={list(pred)[:40]}"}
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = _positives(lab)
        resseq = [int(r) for r in rs]
        scores = np.asarray([float(rs[str(r)]) for r in resseq], dtype=float)
        y = np.asarray([1 if r in pos else 0 for r in resseq], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            return unit, {"error": "degenerate labels", "n_pos": int(y.sum()), "n": len(y)}
        auc = float(roc_auc_score(y, scores))
        return unit, {"auc": auc, "n": len(y), "n_pos": int(y.sum())}
    except Exception as ex:  # noqa: BLE001 — probe must finish
        return unit, {"error": f"{type(ex).__name__}: {ex}"}


def main() -> int:
    man = json.loads((ROOT / "data/cryptobench_apo/official_manifest.json").read_text())
    entries = man["entries"]
    plm = json.loads((ROOT / "results/baselines/PLMNN_SCORES.json").read_text())
    plm_by = {u["unit_id"]: u for u in plm["units"]}

    t0 = time.perf_counter()
    got: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(one, e) for e in entries]
        for i, fut in enumerate(as_completed(futs), 1):
            unit, rec = fut.result()
            got[unit] = rec
            if i % 10 == 0:
                print(f"{i}/{len(entries)} {time.perf_counter() - t0:.0f}s", flush=True)

    rows = []
    for e in entries:
        unit = f"{e['pdb']}_{e['chain']}"
        g = got.get(unit, {})
        if "error" in g or "auc" not in g:
            continue
        pu = plm_by.get(unit)
        if not pu:
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
            "auc": g["auc"],
            "geometry_field": g["auc"],
            "plmnn": pauc,
            "delta": g["auc"] - pauc,
            "n_pos": g["n_pos"],
            "n": g["n"],
        })

    deltas = np.asarray([r["delta"] for r in rows], dtype=float)
    gmean = float(np.mean([r["geometry_field"] for r in rows])) if rows else float("nan")
    pmean = float(np.mean([r["plmnn"] for r in rows])) if rows else float("nan")
    ci = None
    if len(deltas) >= 8:
        rng = np.random.default_rng(0)
        boots = [float(deltas[rng.integers(0, len(deltas), len(deltas))].mean())
                 for _ in range(4000)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    out = {
        "schema": "geoaudit.geometry_field_vs_plmnn_official_probe.v1",
        "clinical_grade": False,
        "method": "geometry_field",
        "reads_test_fold": True,
        "why_not_confirmatory": (
            "the official fold has been read many times; this probe only asks "
            "whether geometry_field (645+624) closes any of the cached pLM-NN gap"
        ),
        "n_jobs": N_JOBS,
        "n_units_compared": len(rows),
        "n_errors": sum(1 for v in got.values() if "error" in v),
        "errors_sample": [f"{u}:{v['error']}" for u, v in got.items() if "error" in v][:12],
        "mean_geometry_field": gmean,
        "mean_residue_auc": gmean,
        "mean_plmnn": pmean,
        "mean_delta_geometry_minus_plmnn": float(deltas.mean()) if len(deltas) else None,
        "n_geometry_ahead": int((deltas > 0).sum()) if len(deltas) else 0,
        "n_plmnn_ahead": int((deltas < 0).sum()) if len(deltas) else 0,
        "ci95_delta": ci,
        "seconds": time.perf_counter() - t0,
        "units_sorted_by_delta": sorted(rows, key=lambda r: r["delta"]),
    }
    out["per_unit"] = [
        {
            "unit": r["unit"],
            "unit_id": r["unit"],
            "auc": r["auc"],
            "geometry_field": r["geometry_field"],
            "plmnn": r["plmnn"],
            "delta": r["delta"],
            "n_pos": r["n_pos"],
            "n": r["n"],
        }
        for r in out["units_sorted_by_delta"]
    ]
    out["residue_auc_mean"] = out["mean_geometry_field"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", OUT)
    print(json.dumps({k: out[k] for k in out if k != "units_sorted_by_delta"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
