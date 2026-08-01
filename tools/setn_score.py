#!/usr/bin/env python3
"""Score Set N with the two counting fields and P2Rank. No label is opened.

Scoring is kept apart from comparing, as it is for Set A, so that the comparison can
be shown to have run once under the plan rather than to have been rehearsed. This
tool writes per-residue scores and nothing else; ``setn_read.py`` is what forms a
statistic, and the plan that governs it was committed before this ran.

Three methods here, and why one of them is new
-----------------------------------------------
``table_field`` is the 645-wire detector Set A's confirmatory read was about.
``geometry_field`` is the 1269-column detector the family ladder produced, whose
+0.0121 on twelve cluster-disjoint halvings is the strongest training-fold result in
the repository and has never been confirmed on X-ray held-out data. It cannot be run
on Set A --- a spent set cannot confirm a second method --- so Set N is the first
X-ray external evidence it can have, and the plan confines it to a secondary that
reads Set N alone.

Helpers come from ``external_score``: the residue universe, the row shape and the
tool-version string are the same functions Set A's scores were written by, so a
difference between the two sets cannot be a difference in how a score was recorded.

Usage: PYTHONPATH=src:tools python3.12 tools/setn_score.py [--jobs 9] [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MANIFEST = ROOT_DIR / "data/external/setn_manifest.json"
PREDS = ROOT_DIR / "results/external/setn_predictions"
ARCHIVE = ROOT_DIR / "results/external/setn_p2rank_raw"
SCHEMA = "geoaudit.setn_prediction.v1"
METHODS = ("table_field", "geometry_field", "p2rank")

_M: dict = {}


def _load(method: str) -> None:
    """One method's module per worker, imported on first use.

    Only the method being scored is imported. P2Rank's wrapper starts a JVM and the
    geometry field reads a 1269-column compiled table, and a worker that will never
    call either should pay for neither.
    """
    import sys
    sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]
    from importlib import import_module
    _M[method] = import_module(f"pocket_bench.methods.{method}"
                               if method != "p2rank"
                               else "pocket_bench.methods.p2rank_wrap")


def _score_one(job: tuple) -> tuple[str, dict]:
    import sys
    sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]
    from external_score import universe, _row, _tool_version
    method, entry = job
    if method not in _M:
        _load(method)
    rec = ROOT_DIR / entry["receptor_path"]
    got = hashlib.sha256(rec.read_bytes()).hexdigest()
    if got != entry["receptor_sha256"]:
        raise SystemExit(
            f"{entry['pdb']}_{entry['chain']}: the receptor no longer matches the "
            f"manifest digest. Set N is frozen; scoring a changed input would make "
            f"the comparison meaningless")
    mod = _M[method]
    if method == "p2rank":
        pred = mod.predict(rec, pdb_id=entry["pdb"], chain=entry["chain"],
                           archive_dir=ARCHIVE)
    else:
        pred = mod.predict(rec, pdb_id=entry["pdb"], chain=entry["chain"])
    row = _row(pred, universe(rec), receptor_sha256=entry["receptor_sha256"],
               tool_version=_tool_version(method, pred))
    return f"{entry['pdb']}_{entry['chain']}", row


def run(jobs: int, methods: tuple[str, ...]) -> dict:
    entries = json.loads(MANIFEST.read_text())["entries"]
    PREDS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    index, failures = {}, []
    for method in methods:
        t0 = time.time()
        work = [(method, e) for e in entries]
        by_unit = {}
        with mp.get_context("fork").Pool(jobs) as pool:
            for i, (unit, row) in enumerate(
                    pool.imap_unordered(_score_one, work, chunksize=2), 1):
                by_unit[unit] = row
                if not row["residue_scores"]:
                    failures.append({"unit": unit, "method": method,
                                     "status": row["status"],
                                     "error": row["error"]})
                if i % 50 == 0 or i == len(work):
                    print(f"  {method} {i}/{len(work)}  "
                          f"{time.time() - t0:.0f}s", flush=True)
        payload = {"schema": SCHEMA, "clinical_grade": False, "method": method,
                   "fold": "external_negative", "n_units": len(by_unit),
                   "manifest_sha256": hashlib.sha256(
                       MANIFEST.read_bytes()).hexdigest(),
                   "units": dict(sorted(by_unit.items()))}
        p = PREDS / f"{method}.json"
        p.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        index[method] = {"file": p.name, "n_units": len(by_unit),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "seconds": round(time.time() - t0, 1)}
        print(f"wrote {p.relative_to(ROOT_DIR)}", flush=True)

    idx = PREDS / "INDEX.json"
    prior = json.loads(idx.read_text())["methods"] if idx.is_file() else {}
    idx.write_text(json.dumps({
        "schema": "geoaudit.raw_predictions_index.v1",
        "clinical_grade": False, "fold": "external_negative",
        "carries_no_metric": True,
        "why": ("scoring is kept apart from comparing so that the comparison can "
                "be shown to have run once, under the plan"),
        "n_failures": len(failures), "failures": failures,
        "methods": prior | index}, indent=2) + "\n")
    return {"methods": index, "failures": failures, "n_units": len(entries)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(9, os.cpu_count() or 4))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        idx = PREDS / "INDEX.json"
        if not idx.is_file():
            print(f"MISSING {idx.relative_to(ROOT_DIR)}")
            return 1
        d = json.loads(idx.read_text())
        for m, v in d["methods"].items():
            print(f"  {m}: {v['n_units']} units, {v.get('seconds', '?')}s")
        print(f"  failures {d['n_failures']}")
        print(f"OK {idx.relative_to(ROOT_DIR)}")
        return 0
    out = run(a.jobs, tuple(m for m in a.methods.split(",") if m))
    print(f"\n{out['n_units']} units, {len(out['failures'])} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
