#!/usr/bin/env python3
"""Run pocket methods on the pinned CryptoBench-apo subset and write an honest report.

Metric: Top-1 DCA <= 4 A from the predicted pocket centre to the nearest atom of
the labelled cryptic binding residues (a pocket-LOCALIZATION proxy). NOTE this is
NOT the CryptoBench per-residue classification protocol (AUC/F1); a fully faithful
comparison to the CryptoBench baselines would use their residue-level metrics.

Methods: geometric_foundation (rigid), fstar_pocket (F*-breathing ablation),
sstar_pocket (anisotropic-shear S* oracle, dynamic spectral amplitudes),
p2rank (needs JAVA_HOME + P2Rank on PATH), random_bbox. TOOL_UNAVAILABLE is never
counted as a miss.

Usage: PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Pin BLAS/OMP to a single thread PER WORKER before numpy/scipy are imported.
# The spectral step (ARPACK shift-invert on a 3N x 3N sparse Hessian) is the
# runtime hotspot; running P worker processes that each spawn a full BLAS thread
# pool oversubscribes the cores and is slower than serial. Parallelism is taken
# across structures instead, which is where it actually scales.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pocket_bench.adapters import load_official_test_fold
from pocket_bench.methods import (
    fstar_pocket,
    geometric_foundation,
    p2rank_wrap,
    prediction,
    sstar_pocket,
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


def _pilot_items() -> list[tuple[dict, Path]]:
    """The pinned n=15 stride pilot (NOT the official fold)."""
    items: list[tuple[dict, Path]] = []
    for lp in sorted(glob.glob(str(LAB / "*_labels.json"))):
        lab = json.loads(Path(lp).read_text())
        items.append((lab, REC / f"{lab['pdb_id']}_{lab['chain']}_receptor.pdb"))
    return items


def _official_items() -> list[tuple[dict, Path]]:
    """The official CryptoBench MMseqs2 10% cluster-disjoint TEST fold.

    Fail-closed by construction: ``load_official_test_fold`` raises if the manifest
    is absent or any SHA-256 mismatches. There is deliberately NO fallback to the
    pilot — silently substituting a 15-structure stride sample for the official fold
    is the exact failure mode this flag exists to prevent.
    """
    manifest = load_official_test_fold()
    items: list[tuple[dict, Path]] = []
    for e in manifest["entries"]:
        lab = json.loads((ROOT / e["label_path"]).read_text())
        items.append((lab, ROOT / e["receptor_path"]))
    return items


def _predict(method: str, rec: Path, pdb: str) -> dict:
    if method == "geometric_foundation":
        return geometric_foundation.predict(rec, pdb_id=pdb)
    if method == "fstar_pocket":
        return fstar_pocket.predict(rec, pdb_id=pdb)
    if method == "sstar_pocket":
        return sstar_pocket.predict(rec, pdb_id=pdb)
    if method == "p2rank":
        return p2rank_wrap.predict(rec, pdb_id=pdb, top_k=5)
    if method == "random_bbox":
        return _random_baseline(rec, pdb_id=pdb)
    raise KeyError(method)


METHOD_NAMES = ("geometric_foundation", "fstar_pocket", "sstar_pocket",
                "p2rank", "random_bbox")


def _run_one(item: tuple[dict, Path],
             method_names: tuple[str, ...] = METHOD_NAMES,
             ) -> tuple[list[int], dict[str, tuple[dict, dict]]]:
    """Score every method on ONE structure. Module-level so it is picklable.

    Pure function of (label, receptor): no shared state, no RNG dependence on
    evaluation order (random_bbox is seeded per structure), so the parallel result
    is identical to the serial one.
    """
    lab, rec = item
    pdb, ch = lab["pdb_id"], lab["chain"]
    universe = _receptor_residue_universe(rec, ch)
    res: dict[str, tuple[dict, dict]] = {}
    for m in method_names:
        pred = _predict(m, rec, pdb)
        if "ligand_heavy_coords" in lab:
            sc = score_prediction(pred, lab)
        else:
            # Official apo labels carry cryptic residues but no holo ligand
            # coordinates; DCA is undefined there, so it is reported null rather
            # than fabricated. Residue-level metrics remain valid.
            sc = {"method": m, "pdb_id": pdb, "status": pred.get("status", "OK"),
                  "runtime_s": pred.get("runtime_s"),
                  "primary_metric": "residue_level_only",
                  "clinical_grade": False, "top1": None, "top3": None,
                  "dcc_top1": None, "residue_f1": {"available": False}}
        res[m] = (pred, sc)
    return universe, res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("pilot", "official"), default="pilot",
                    help="'official' = CryptoBench MMseqs2 10%% test fold "
                         "(fail-closed, never falls back to the pilot)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes over structures "
                         "(results are order-preserving and identical to --jobs 1)")
    ap.add_argument("--methods", default=",".join(METHOD_NAMES),
                    help="comma-separated subset of: " + ",".join(METHOD_NAMES))
    args = ap.parse_args(argv)
    mnames = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(mnames) - set(METHOD_NAMES)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    is_official = args.dataset == "official"
    items = _official_items() if is_official else _pilot_items()
    out = ROOT / ("results/cryptobench_official" if is_official
                  else "results/cryptobench_apo")

    # One BLAS/OpenMP thread per worker: the parallelism is across structures, so
    # nested threading would oversubscribe the cores and slow the run down.
    if args.jobs > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"

    methods = {m: None for m in mnames}
    env = json.loads(BASELINE_ENV.read_text())
    p2rank_version = ((env.get("tools") or {}).get("p2rank") or {}).get("version")
    per = {m: {"ok": 0, "top1_hits": 0, "unavailable": 0, "crash_empty": 0, "rows": []} for m in methods}
    telem_rows: list[dict] = []
    n_attempted = {m: 0 for m in methods}
    t_start = time.perf_counter()
    if args.jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            results = []
            stream = ex.map(_run_one, items, itertools.repeat(mnames), chunksize=1)
            for done, r in enumerate(stream, 1):
                results.append(r)
                if done % 10 == 0 or done == len(items):
                    el = time.perf_counter() - t_start
                    print(f"  [{done}/{len(items)}] {el:.0f}s elapsed, "
                          f"~{el / done * (len(items) - done):.0f}s left", flush=True)
    else:
        results = [_run_one(it, mnames) for it in items]
    print(f"  predictions done in {time.perf_counter() - t_start:.0f}s "
          f"(jobs={args.jobs})", flush=True)

    for idx, ((lab, rec), (universe, res)) in enumerate(zip(items, results), 1):
        pdb, ch = lab["pdb_id"], lab["chain"]
        for m in methods:
            pred, sc = res[m]
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
        "schema": "geoaudit.cryptobench.report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clinical_grade": False,
        "dataset": args.dataset,
        "is_official_mmseqs2_10pct_test_fold": is_official,
        "benchmark": (
            "CryptoBench official MMseqs2 10% cluster-disjoint TEST fold "
            "(Skrhak 2025)" if is_official else
            "CryptoBench apo (Skrhak 2025) — pinned deterministic subset"
        ),
        "n_structures": len(items),
        "primary_metric": ("residue_auc/residue_pr_auc/residue_mcc/residue_f1"
                           if is_official else
                           "top1_dca_le_4A_to_cryptic_residue_atoms"),
        "metric_caveat": (
            "Official apo labels carry cryptic residues but no holo ligand "
            "coordinates; DCA is undefined and reported null. Residue-level metrics "
            "are the primary readout." if is_official else
            "Pocket-localization proxy (DCA to cryptic-residue atoms); NOT the "
            "CryptoBench per-residue AUC/F1 protocol. DCA is more permissive for "
            "structures with many labelled residues."
        ),
        "splits_note": (
            "Official CryptoBench TEST fold: MMseqs2 clustering at 10% sequence "
            "identity, loaded fail-closed with per-file SHA-256 verification. "
            "clinical_grade=false." if is_official else
            "CryptoBench test splits are cluster-disjoint at 10% sequence identity; "
            "this pinned subset is a deterministic stride sample of the full label "
            "set (train+test), not exclusively the test fold. clinical_grade=false."
        ),
        "summaries": summaries,
        "per_method": per,
    }
    # 0-masking telemetry: faithful per-residue metrics + fail-closed denominators.
    telemetry = aggregate(telem_rows, n_attempted)
    assert_denominator_discipline(telemetry, declared_available_tools(env))
    telemetry["rows"] = telem_rows
    report["telemetry_ref"] = "TELEMETRY.json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "APO_BENCHMARK.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "TELEMETRY.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    print(json.dumps({m: s["top1_dca_le_4A_hits"] for m, s in summaries.items()}, indent=2))
    # faithful metric availability (honest: null where universe/labels cannot join)
    avail = sum(1 for r in telem_rows if r["residue_metrics_available"])
    print(f"faithful residue metrics available on {avail}/{len(telem_rows)} rows")
    print(f"dataset={args.dataset} n={len(items)} -> {out.relative_to(ROOT)}/"
          "{APO_BENCHMARK,TELEMETRY}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
