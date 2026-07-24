#!/usr/bin/env python3
"""Run prediction + scoring + report for the ESR1 pocket appendix.

This is the previously-missing run harness. It runs each method on
receptor-only ATOM PDBs, scores Top-1 DCA<=4A against the CORRECTED
chain-scoped labels, and writes an honest regenerated report.

Truth boundaries preserved:
- ligand labels are joined ONLY at scoring time (prediction sees receptor only);
- TOOL_UNAVAILABLE is never a miss (kept out of the primary denominator);
- splits are not cluster-disjoint, so no comparative-superiority claim is made.

Usage:
  PYTHONPATH=src python3.12 tools/run_pilot.py [--split locked_test|all]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from pocket_bench.dataset.catalog import CURATED_ENTRIES
from pocket_bench.methods import (
    foliation_pocket,
    fpocket_wrap,
    geometric_foundation,
    p2rank_wrap,
    prediction,
)
from pocket_bench.metrics import score_prediction
from pocket_bench.pdb_io import parse_pdb_atoms

ROOT = Path(__file__).resolve().parents[1]
RECEPTOR_DIR = ROOT / "data/receptors"
LABEL_DIR = ROOT / "data/labels"
HPARAMS = json.loads((ROOT / "configs/pilot_hparams.json").read_text())
GRID_STEP = float(HPARAMS["foliation_pocket_ro"]["grid_step"])
TOP_K = int(HPARAMS["foliation_pocket_ro"]["top_k"])


def _random_baseline(receptor_pdb: Path, *, pdb_id: str, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    atoms = [a for a in parse_pdb_atoms(receptor_pdb.read_text()) if a["record"] == "ATOM"]
    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    rng = random.Random(seed)
    pockets = []
    for rank in range(1, TOP_K + 1):
        pockets.append(
            {
                "rank": rank,
                "center_xyz": [
                    rng.uniform(min(xs), max(xs)),
                    rng.uniform(min(ys), max(ys)),
                    rng.uniform(min(zs), max(zs)),
                ],
                "score": 1.0 / rank,
                "residues": [],
            }
        )
    return prediction(
        method="random_bbox",
        pdb_id=pdb_id,
        status="OK",
        pockets=pockets,
        runtime_s=time.perf_counter() - t0,
        extra={"seed": seed},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="locked_test")
    args = ap.parse_args()

    entries = [
        e
        for e in CURATED_ENTRIES
        if e.get("ligand_resname")
        and (args.split == "all" or e["split"] == args.split)
    ]

    methods = {
        "foliation_pocket_ro": lambda rec, pid: foliation_pocket.predict(
            rec, pdb_id=pid, grid_step=GRID_STEP, top_k=TOP_K
        ),
        "geometric_foundation": lambda rec, pid: geometric_foundation.predict(
            rec, pdb_id=pid, top_k=TOP_K
        ),
        "fpocket": lambda rec, pid: fpocket_wrap.predict(rec, pdb_id=pid, top_k=TOP_K),
        "p2rank": lambda rec, pid: p2rank_wrap.predict(rec, pdb_id=pid, top_k=TOP_K),
        "random_bbox": lambda rec, pid: _random_baseline(rec, pdb_id=pid),
    }

    per_method: dict[str, dict] = {m: {"ok": 0, "top1_hits": 0, "unavailable": 0, "crash_empty": 0, "rows": []} for m in methods}

    for e in entries:
        pdb, chain = e["pdb_id"], e["chain"]
        rec = RECEPTOR_DIR / f"{pdb}_{chain}_receptor.pdb"
        label = json.loads((LABEL_DIR / f"{pdb}_{chain}_labels.json").read_text())
        for m, fn in methods.items():
            pred = fn(rec, pdb)
            scored = score_prediction(pred, label)
            agg = per_method[m]
            status = scored["status"]
            if status == "TOOL_UNAVAILABLE":
                agg["unavailable"] += 1
            elif status != "OK":
                agg["crash_empty"] += 1
            else:
                agg["ok"] += 1
                top1 = scored.get("top1") or {}
                if top1.get("success") is True:
                    agg["top1_hits"] += 1
            top1 = scored.get("top1") or {}
            agg["rows"].append(
                {
                    "pdb": pdb,
                    "status": status,
                    "top1_success": top1.get("success"),
                    "best_dca": top1.get("best_dca"),
                    "dcc_top1": scored.get("dcc_top1"),
                }
            )

    summaries = {}
    for m, agg in per_method.items():
        denom = agg["ok"] + agg["crash_empty"]  # intention-to-evaluate; unavailable excluded
        summaries[m] = {
            "top1_dca_le_4A_hits": agg["top1_hits"],
            "intention_to_evaluate_denominator": denom,
            "tool_unavailable": agg["unavailable"],
            "crash_or_empty": agg["crash_empty"],
        }

    report = {
        "schema": "foliation.pocket_bench.regenerated_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clinical_grade": False,
        "not_a_locked_test": True,
        "split": args.split,
        "labels": "chain_scoped_single_instance_corrected",
        "primary_metric": "top1_dca_le_4A",
        "dca_threshold_a": 4.0,
        "unit_of_analysis": "one_pdb_structure",
        "n_structures": len(entries),
        "summaries": summaries,
        "per_method": per_method,
        "truth_boundary": "Regenerated on CORRECTED chain-scoped labels. Splits "
        "are NOT cluster-disjoint, so NO comparative-superiority claim is made. "
        "TOOL_UNAVAILABLE is excluded from the denominator, never counted as a "
        "miss. clinical_grade=false.",
    }
    out = ROOT / "results/pilot/REGENERATED_PILOT_REPORT.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "n": len(entries), "out": str(out.name)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
