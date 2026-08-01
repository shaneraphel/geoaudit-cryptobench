#!/usr/bin/env python3
"""Score geometry_field on Set N without multiprocessing.

``setn_score.py`` uses a fork pool. On this machine every forked worker hit
``SystemExit`` inside ``_tool_version`` because geometry_field reported no
version, and the parent hung at 0% CPU waiting for results that would never
arrive. This path is one process, one unit at a time, with a resume checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    import sys
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
    from external_score import universe, _row, _tool_version
    import pocket_bench.methods.geometry_field as gf

    manifest = ROOT / "data/external/setn_manifest.json"
    preds = ROOT / "results/external/setn_predictions"
    ckpt = ROOT / "results/baselines/_geometry_setn_checkpoint.jsonl"
    entries = json.loads(manifest.read_text())["entries"]
    done: dict = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["unit"]] = r["row"]
    print(f"resuming {len(done)}/{len(entries)}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    by = dict(done)
    for i, e in enumerate(entries, 1):
        uid = f"{e['pdb']}_{e['chain']}"
        if uid in by:
            continue
        rec = ROOT / e["receptor_path"]
        got = hashlib.sha256(rec.read_bytes()).hexdigest()
        if got != e["receptor_sha256"]:
            raise SystemExit(f"digest mismatch {uid}")
        t1 = time.time()
        pred = gf.predict(rec, pdb_id=e["pdb"], chain=e["chain"])
        row = _row(pred, universe(rec), receptor_sha256=e["receptor_sha256"],
                   tool_version=_tool_version("geometry_field", pred))
        by[uid] = row
        with ckpt.open("a") as fh:
            fh.write(json.dumps({"unit": uid, "row": row,
                                 "seconds": round(time.time() - t1, 2)}) + "\n")
        if len(by) % 10 == 0 or len(by) == len(entries):
            print(f"geometry_field {len(by)}/{len(entries)}  "
                  f"{time.time() - t0:.0f}s  last={uid} "
                  f"{time.time() - t1:.1f}s", flush=True)
    payload = {
        "schema": "geoaudit.setn_prediction.v1",
        "clinical_grade": False,
        "method": "geometry_field",
        "fold": "external_negative",
        "n_units": len(by),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "units": dict(sorted(by.items())),
    }
    p = preds / "geometry_field.json"
    p.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    idx_p = preds / "INDEX.json"
    idx = json.loads(idx_p.read_text()) if idx_p.is_file() else {"methods": {}}
    idx.setdefault("methods", {})["geometry_field"] = {
        "file": p.name, "n_units": len(by),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    idx_p.write_text(json.dumps(idx, indent=2) + "\n")
    print(f"wrote {p.relative_to(ROOT)} n={len(by)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
