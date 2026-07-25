#!/usr/bin/env python3
"""Run pocket methods on the pinned CryptoBench-apo subset and write an honest report.

Metric: Top-1 DCA <= 4 A from the predicted pocket centre to the nearest atom of
the labelled cryptic binding residues (a pocket-LOCALIZATION proxy). NOTE this is
NOT the CryptoBench per-residue classification protocol (AUC/F1); a fully faithful
comparison to the CryptoBench baselines would use their residue-level metrics.

Methods: geometric_foundation (rigid), fstar_pocket (F*-breathing ablation),
p2rank (needs JAVA_HOME + P2Rank on PATH), random_bbox. TOOL_UNAVAILABLE is never
counted as a miss.

Usage: PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py
"""
from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path

from pocket_bench.methods import (
    fstar_pocket,
    geometric_foundation,
    p2rank_wrap,
    prediction,
)
from pocket_bench.metrics import score_prediction
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.telemetry import (
    aggregate,
    assert_denominator_discipline,
    declared_available_tools,
    telemetry_row,
)

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/cryptobench_apo/receptors"
LAB = ROOT / "data/cryptobench_apo/labels"
BASELINE_ENV = ROOT / "data/manifests/BASELINE_ENV.json"


def _receptor_residue_universe(rec: Path, chain: str | None) -> list[int]:
    resseqs: set[int] = set()
    for a in parse_pdb_atoms(rec.read_text()):
        if a["record"] != "ATOM":
            continue
        if chain is not None and a["chain"] != chain:
            continue
        resseqs.add(int(a["resseq"]))
    return sorted(resseqs)


def _random_baseline(rec: Path, *, pdb_id: str, seed: int = 42, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    atoms = [a for a in parse_pdb_atoms(rec.read_text()) if a["record"] == "ATOM"]
    xs = [a["x"] for a in atoms]; ys = [a["y"] for a in atoms]; zs = [a["z"] for a in atoms]
    rng = random.Random(seed)
    pockets = [
        {"rank": r, "center_xyz": [rng.uniform(min(xs), max(xs)),
         rng.uniform(min(ys), max(ys)), rng.uniform(min(zs), max(zs))],
         "score": 1.0 / r, "residues": []}
        for r in range(1, top_k + 1)
    ]
    return prediction(method="random_bbox", pdb_id=pdb_id, status="OK",
                      pockets=pockets, runtime_s=time.perf_counter() - t0)


def main() -> int:
    methods = {
        "geometric_foundation": lambda rec, pid: geometric_foundation.predict(rec, pdb_id=pid),
        "fstar_pocket": lambda rec, pid: fstar_pocket.predict(rec, pdb_id=pid),
        "p2rank": lambda rec, pid: p2rank_wrap.predict(rec, pdb_id=pid, top_k=5),
        "random_bbox": lambda rec, pid: _random_baseline(rec, pdb_id=pid),
    }
    labs = sorted(glob.glob(str(LAB / "*_labels.json")))
    env = json.loads(BASELINE_ENV.read_text())
    p2rank_version = ((env.get("tools") or {}).get("p2rank") or {}).get("version")
    per = {m: {"ok": 0, "top1_hits": 0, "unavailable": 0, "crash_empty": 0, "rows": []} for m in methods}
    telem_rows: list[dict] = []
    n_attempted = {m: 0 for m in methods}
    for lp in labs:
        lab = json.loads(Path(lp).read_text())
        pdb, ch = lab["pdb_id"], lab["chain"]
        rec = REC / f"{pdb}_{ch}_receptor.pdb"
        universe = _receptor_residue_universe(rec, ch)
        for m, fn in methods.items():
            pred = fn(rec, pdb)
            sc = score_prediction(pred, lab)
            agg = per[m]; st = sc["status"]; top1 = sc.get("top1") or {}
            n_attempted[m] += 1
            if st == "TOOL_UNAVAILABLE":
                agg["unavailable"] += 1
            elif st != "OK":
                agg["crash_empty"] += 1
            else:
                agg["ok"] += 1
                if top1.get("success") is True:
                    agg["top1_hits"] += 1
            agg["rows"].append({"pdb": pdb, "status": st,
                                "top1_success": top1.get("success"),
                                "best_dca": top1.get("best_dca")})
            telem_rows.append(
                telemetry_row(
                    method=m, pdb=pdb, split="test", status=st,
                    scored=sc, label=lab, prediction=pred,
                    universe_residues=universe,
                    tool_version=p2rank_version if m == "p2rank" else None,
                    env_sha=None, seed=42 if m == "random_bbox" else 0,
                    runtime_s=pred.get("runtime_s"),
                )
            )
    summaries = {m: {"top1_dca_le_4A_hits": a["top1_hits"],
                     "intention_to_evaluate_denominator": a["ok"] + a["crash_empty"],
                     "tool_unavailable": a["unavailable"]} for m, a in per.items()}
    report = {
        "schema": "gf4cc.cryptobench_apo.report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clinical_grade": False,
        "benchmark": "CryptoBench apo (Skrhak 2025) — pinned deterministic subset",
        "n_structures": len(labs),
        "primary_metric": "top1_dca_le_4A_to_cryptic_residue_atoms",
        "metric_caveat": "Pocket-localization proxy (DCA to cryptic-residue atoms); "
        "NOT the CryptoBench per-residue AUC/F1 protocol. DCA is more permissive for "
        "structures with many labelled residues.",
        "splits_note": "CryptoBench test splits are cluster-disjoint at 10% sequence "
        "identity; this pinned subset is a deterministic stride sample of the full "
        "label set (train+test), not exclusively the test fold. clinical_grade=false.",
        "summaries": summaries,
        "per_method": per,
    }
    # 0-masking telemetry: faithful per-residue metrics + fail-closed denominators.
    telemetry = aggregate(telem_rows, n_attempted)
    assert_denominator_discipline(telemetry, declared_available_tools(env))
    telemetry["rows"] = telem_rows
    report["telemetry_ref"] = "TELEMETRY.json"
    out = ROOT / "results/cryptobench_apo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "APO_BENCHMARK.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "TELEMETRY.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    print(json.dumps({m: s["top1_dca_le_4A_hits"] for m, s in summaries.items()}, indent=2))
    # faithful metric availability (honest: null where universe/labels cannot join)
    avail = sum(1 for r in telem_rows if r["residue_metrics_available"])
    print(f"faithful residue metrics available on {avail}/{len(telem_rows)} rows")
    print("n =", len(labs), "-> results/cryptobench_apo/APO_BENCHMARK.json + TELEMETRY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
