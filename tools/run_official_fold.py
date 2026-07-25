#!/usr/bin/env python3
"""Run GeoAudit's deterministic detector on the official CryptoBench MMseqs2 10%
cluster-disjoint TEST fold and freeze residue-level metrics with bootstrap CIs.

This is a RUNNER: it changes no detector, metric, or bootstrap logic. It loads the
official fold via the fail-closed adapter (which SHA-256-verifies every receptor and
label), runs the geometric foundation per structure, and computes the CryptoBench
headline metrics (ROC-AUC, PR-AUC, MCC, F1) per structure, then a bootstrap CI over
structures.

Protocol (CryptoBench residue classification, apo frame):
  * universe = all residues of the apo chain in the receptor;
  * y = 1 if the residue is in the cryptic ``apo_pocket_selection``, else 0;
  * geometric_foundation score = rank-weighted pocket membership
    (``residue_scores_from_pockets``);
  * random_residue = uniform[0,1] per-residue null (fixed seed) — a residue-level
    chance baseline. Real ML baselines (P2Rank, PocketMiner) are loaded ONLY if
    their per-residue predictions are physically present; absence is reported, never
    imputed (clinical_grade=false).

MCC/F1 operating points (threshold-dependent, stated explicitly):
  * geometric_foundation: predicted-positive == residue in any predicted pocket;
  * random_residue: predicted-positive == score >= 0.5.

Usage: PYTHONPATH=src python3.12 tools/run_official_fold.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from pocket_bench.adapters import (
    DataUnavailable,
    load_official_test_fold,
    load_pocketminer_scores,
    pocketminer_available,
)
from pocket_bench.methods import geometric_foundation
from pocket_bench.metrics import (
    average_precision,
    residue_scores_from_pockets,
    roc_auc,
)
from pocket_bench.metrics_bootstrap import f1, mcc, paired_bootstrap
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

OUT = ROOT / "results/official_fold"
_POS_EPS = 1e-9  # geometric_foundation: any nonzero pocket membership == predicted


def _chain_universe(receptor: Path, chain: str) -> list[int]:
    seen: set[int] = set()
    for a in parse_pdb_atoms(receptor.read_text()):
        if a["record"] == "ATOM" and a["chain"] == chain:
            seen.add(int(a["resseq"]))
    return sorted(seen)


def _metrics(scores: list[float], y: list[int], thr: float) -> dict[str, float | None]:
    return {
        "roc_auc": roc_auc(scores, y),
        "pr_auc": average_precision(scores, y),
        "mcc": mcc(scores, y, thr),
        "f1": f1(scores, y, thr),
    }


def main() -> int:
    manifest = load_official_test_fold()  # fail-closed + SHA-256 verified
    entries = manifest["entries"]
    rng = random.Random(42)

    # per-structure metric values, per method, aligned by index
    metric_keys = ("roc_auc", "pr_auc", "mcc", "f1")
    vals: dict[str, dict[str, list[float | None]]] = {
        m: {k: [] for k in metric_keys} for m in ("geometric_foundation", "random_residue")
    }
    per_structure: list[dict] = []
    n_scored = 0
    t_start = time.perf_counter()

    for i, e in enumerate(entries, 1):
        pdb, chain = e["pdb"], e["chain"]
        rec = ROOT / e["receptor_path"]
        lab = json.loads((ROOT / e["label_path"]).read_text())
        truth = {int(r) for r in lab["cryptic_residues"]}
        universe = _chain_universe(rec, chain)
        if not universe or not (truth & set(universe)):
            for m in vals:
                for k in metric_keys:
                    vals[m][k].append(None)
            per_structure.append({"pdb": pdb, "chain": chain, "status": "NO_JOIN"})
            continue

        pred = geometric_foundation.predict(rec, pdb_id=pdb)
        pockets = pred.get("pockets") or []
        gf_scores_by_res = residue_scores_from_pockets(pockets, universe)
        gf_scores = [gf_scores_by_res.get(r, 0.0) for r in universe]
        y = [1 if r in truth else 0 for r in universe]
        rnd_scores = [rng.random() for _ in universe]

        gf_m = _metrics(gf_scores, y, _POS_EPS)
        rnd_m = _metrics(rnd_scores, y, 0.5)
        for k in metric_keys:
            vals["geometric_foundation"][k].append(gf_m[k])
            vals["random_residue"][k].append(rnd_m[k])
        n_scored += 1
        per_structure.append({
            "pdb": pdb, "chain": chain, "status": pred.get("status"),
            "n_universe": len(universe), "n_true": sum(y),
            "geometric_foundation": gf_m, "random_residue": rnd_m,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(entries)}] scored={n_scored} "
                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)

    # bootstrap CI per metric; paired vs the residue-level chance baseline
    boot: dict[str, dict] = {}
    for k in metric_keys:
        boot[k] = paired_bootstrap(
            {m: vals[m][k] for m in vals}, baseline="random_residue",
            n_boot=10000, seed=20260725, ci=0.95,
        )

    # real ML baselines: present only if per-residue prediction files exist
    baseline_status = {
        "pocketminer": "AVAILABLE" if pocketminer_available() else "DATA_UNAVAILABLE",
        "p2rank": "DATA_UNAVAILABLE",  # no per-residue prediction files in repo
    }
    pm_join = 0
    if pocketminer_available():
        for e in entries:
            try:
                load_pocketminer_scores(e["pdb"], e["chain"])
                pm_join += 1
            except (DataUnavailable, ValueError):
                pass
    baseline_status["pocketminer_structures_with_scores"] = pm_join

    report = {
        "schema": "geoaudit.official_fold_report.v1",
        "clinical_grade": False,
        "benchmark": "CryptoBench official MMseqs2 10% cluster-disjoint TEST fold "
        "(Skrhak et al., Bioinformatics 2025, doi:10.1093/bioinformatics/btae745)",
        "source": "https://osf.io/pz4a9/  folds/test.json",
        "cryptobench_test_apo_pdbs": manifest.get("cryptobench_test_apo_pdbs"),
        "n_fold_units_single_chain": manifest.get("n_fold_units"),
        "n_excluded_multichain": manifest.get("n_excluded_multichain"),
        "n_entries_in_manifest": len(entries),
        "n_structures_scored": n_scored,
        "primary_metrics": list(metric_keys),
        "mcc_f1_operating_points": {
            "geometric_foundation": "residue in any predicted pocket",
            "random_residue": "score >= 0.5",
        },
        "real_ml_baseline_status": baseline_status,
        "real_ml_baseline_note": "P2Rank/PocketMiner per-residue predictions are not "
        "published on OSF (only a trained pLM-NN model binary); a paired CI against "
        "them is not computed from fabricated numbers. The paired baseline here is a "
        "residue-level uniform-random null.",
        "bootstrap": boot,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "OFFICIAL_FOLD_METRICS.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "PER_STRUCTURE.json").write_text(json.dumps(per_structure, indent=2) + "\n")

    print("\n=== CryptoBench official test fold — frozen metrics "
          f"(n_scored={n_scored}, boot=10000, 95% CI) ===")
    for k in metric_keys:
        pm = boot[k]["per_method"]
        gf = pm["geometric_foundation"]
        rn = pm["random_residue"]
        d = boot[k]["paired_vs_baseline"]["geometric_foundation"]
        gp = "  n/a" if gf["point"] is None else f"{gf['point']:.3f}"
        gl = "n/a" if gf["ci_low"] is None else f"{gf['ci_low']:.3f}"
        gh = "n/a" if gf["ci_high"] is None else f"{gf['ci_high']:.3f}"
        rp = "  n/a" if rn["point"] is None else f"{rn['point']:.3f}"
        print(f"{k:8s} geometric_foundation={gp} [{gl}, {gh}]  "
              f"random={rp}  "
              f"Δ={d['delta_point']:+.3f} [{d['delta_ci_low']:+.3f}, "
              f"{d['delta_ci_high']:+.3f}] "
              f"{'(CI crosses 0)' if d['crosses_zero'] else '(CI excludes 0)'}")
    print(f"\nreal ML baselines: {baseline_status}")
    print(f"-> {(OUT / 'OFFICIAL_FOLD_METRICS.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
