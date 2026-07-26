"""A/B controlled probe of the compiled dual-track resolution fields.

Reads the already-compiled RESOLUTION_FIELD.json (track A, 6 geometric wires)
and RESOLUTION_FIELD_B.json (track B, +4 sequence wires) and scores both on the
official CryptoBench test fold. Geometry is extracted ONCE per unit and shared
by both tracks, so the contrast is attributable to the sequence wires alone.

No compilation, no fitting, no recomputation of the fields.

Usage: PYTHONPATH=src python3.12 tools/probe_dual_track_auc.py --jobs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"


def _one(entry: dict):
    from pocket_bench.methods.quaternary_lut import (
        load_field, receptor_residue_features, apply_propensity, quantize,
    )
    from pocket_bench.metrics import roc_auc, average_precision
    try:
        lab = json.loads((ROOT / entry["label_path"]).read_text())
        # Anti-regression lock: cryptic_residues is the only accepted truth key.
        truth = {int(r) for r in (lab.get("cryptic_residues") or [])}
        if not truth:
            return {"unit": entry.get("pdb_id"), "status": "NO_TRUTH"}

        fa = load_field(track="A")
        fb = load_field(track="B")
        if fb.propensity is None:
            return {"unit": entry.get("pdb_id"), "status": "NO_PROPENSITY"}

        resseq, F, codes = receptor_residue_features(
            ROOT / entry["receptor_path"], chain=entry["chain"],
            with_sequence=True)
        y = [1 if int(r) in truth else 0 for r in resseq]
        if sum(y) == 0 or sum(y) == len(y):
            return {"unit": entry.get("pdb_id"), "status": "DEGENERATE"}

        prop = apply_propensity(codes, fb.propensity)[:, None]
        FA = F[:, :fa.n_feat]
        FB = np.concatenate([F, prop], axis=1)

        addr_a = quantize(FA, fa.edges)
        addr_b = quantize(FB, fb.edges)

        return {
            "unit": entry.get("pdb_id"),
            "status": "OK",
            "n_res": len(y),
            "n_pos": int(sum(y)),
            "A_resolved": roc_auc(list(fa.resolve(FA)), y),
            "A_cell": roc_auc(list(fa.lookup(FA)), y),
            "A_disc": roc_auc(list(fa.discriminant(FA)), y),
            "A_pr": average_precision(list(fa.resolve(FA)), y),
            "B_resolved": roc_auc(list(fb.resolve(FB)), y),
            "B_cell": roc_auc(list(fb.lookup(FB)), y),
            "B_disc": roc_auc(list(fb.discriminant(FB)), y),
            "B_pr": average_precision(list(fb.resolve(FB)), y),
            # how many queried residues land on a Z (never-asserted) address
            "A_z_frac": float((fa.tot[addr_a] == 0).mean()),
            "B_z_frac": float((fb.tot[addr_b] == 0).mean()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"unit": entry.get("pdb_id"), "status": "ERR",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/official_fold/DUAL_TRACK_AB.json")
    args = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"

    entries = json.loads(TEST_MANIFEST.read_text())["entries"]
    if args.limit:
        entries = entries[:args.limit]
    print(f"official test units in manifest: {len(entries)}", flush=True)

    t0 = time.perf_counter()
    rows = None
    if args.jobs > 1:
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.jobs,
                                     mp_context=ctx) as ex:
                rows = list(ex.map(_one, entries, chunksize=1))
        except (PermissionError, OSError, NotImplementedError) as exc:
            # Some sandboxes deny the POSIX semaphore probe that the process
            # pool performs at construction; scoring is identical either way.
            print(f"process pool unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to sequential", flush=True)
            rows = None
    if rows is None:
        rows = []
        for i, entry in enumerate(entries, 1):
            rows.append(_one(entry))
            if i % 10 == 0 or i == len(entries):
                print(f"  {i}/{len(entries)} units "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    dt = time.perf_counter() - t0

    ok = [r for r in rows if r.get("status") == "OK"]
    bad = [r for r in rows if r.get("status") != "OK"]
    print(f"scored {len(ok)}/{len(entries)} units in {dt:.0f}s", flush=True)
    for r in bad[:10]:
        print(f"  SKIP {r.get('unit')} {r.get('status')} {r.get('error','')}")

    if not ok:
        print("NO UNITS SCORED")
        return 1

    def m(k):
        return float(np.mean([r[k] for r in ok]))

    summary = {
        "schema": "geoaudit.dual_track_ab.v1",
        "clinical_grade": False,
        "recompiled": False,
        "fold": "cryptobench official test fold",
        "n_units_in_manifest": len(entries),
        "n_units_scored": len(ok),
        "n_units_skipped": len(bad),
        "truth_key": "cryptic_residues",
        "elapsed_s": round(dt, 1),
        "geometry_shared_between_tracks": True,
        "track_A": {
            "n_features": 6,
            "roc_auc_resolved": m("A_resolved"),
            "roc_auc_cell_only": m("A_cell"),
            "roc_auc_discriminant": m("A_disc"),
            "pr_auc_resolved": m("A_pr"),
            "mean_query_Z_fraction": m("A_z_frac"),
        },
        "track_B": {
            "n_features": 10,
            "roc_auc_resolved": m("B_resolved"),
            "roc_auc_cell_only": m("B_cell"),
            "roc_auc_discriminant": m("B_disc"),
            "pr_auc_resolved": m("B_pr"),
            "mean_query_Z_fraction": m("B_z_frac"),
        },
        "reference_baselines": {
            "p2rank": 0.793, "rigid": 0.664, "sstar": 0.655,
        },
        "per_unit": ok,
        "skipped": bad,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print()
    print(f"{'metric':<26}{'Track A (geom)':>16}{'Track B (geom+seq)':>20}")
    for label, ka, kb in [
        ("ROC-AUC resolved", "A_resolved", "B_resolved"),
        ("ROC-AUC cell only", "A_cell", "B_cell"),
        ("ROC-AUC discriminant", "A_disc", "B_disc"),
        ("PR-AUC resolved", "A_pr", "B_pr"),
        ("query Z fraction", "A_z_frac", "B_z_frac"),
    ]:
        print(f"{label:<26}{m(ka):>16.4f}{m(kb):>20.4f}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
