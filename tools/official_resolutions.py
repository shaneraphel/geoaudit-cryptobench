#!/usr/bin/env python3
"""Crystallographic resolution for every unit of the official test fold.

Metadata only. No label, prediction or score is opened here, and no comparison is
formed; this exists so that the read which does form one can pin the covariate it
stratifies on to an artifact with a digest, rather than fetching it mid-read where a
network hiccup would silently change a stratum's membership.

``external_inventory.describe_all`` is imported unedited. It feeds a frozen artifact
and ``AGENTS.md`` forbids editing such a tool for an unrelated reason; nothing here
needs it edited.

Usage: PYTHONPATH=src:tools python3.12 tools/official_resolutions.py [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/official_fold/OFFICIAL_RESOLUTIONS.json"
SCHEMA = "geoaudit.official_resolutions.v1"


def build() -> dict:
    import external_inventory as ei
    entries = json.loads(MANIFEST.read_text())["entries"]
    ids = sorted({e["pdb"].upper() for e in entries})
    got = ei.describe_all(ids)
    res, method = {}, {}
    for d in got:
        info = d.get("rcsb_entry_info") or {}
        combined = info.get("resolution_combined") or []
        res[d["rcsb_id"].lower()] = min(combined) if combined else None
        method[d["rcsb_id"].lower()] = info.get("experimental_method")
    rows, absent = [], []
    for e in entries:
        r = res.get(e["pdb"].lower())
        if r is None:
            absent.append(f"{e['pdb']}_{e['chain']}")
        rows.append({"unit": f"{e['pdb']}_{e['chain']}", "pdb": e["pdb"],
                     "chain": e["chain"], "resolution": r,
                     "experimental_method": method.get(e["pdb"].lower())})
    have = [r["resolution"] for r in rows if r["resolution"] is not None]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "why_this_does_not_read_the_fold": (
            "a deposition's stated resolution is a property of the experiment, not "
            "of any label, prediction or score. Nothing here is compared to "
            "anything"),
        "source": "RCSB, rcsb_entry_info.resolution_combined, minimum where several",
        "fetched_by": "tools/external_inventory.py describe_all, imported unedited",
        "n_units": len(rows),
        "n_with_a_resolution": len(have),
        "units_without_a_resolution": absent,
        "why_absent_is_listed": (
            "a unit with no stated resolution cannot enter a resolution stratum and "
            "must be visible as excluded rather than silently absent"),
        "span": {"min": min(have) if have else None,
                 "max": max(have) if have else None},
        "units": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        print(f"  {d['n_with_a_resolution']}/{d['n_units']} units carry a "
              f"resolution, {d['span']['min']}-{d['span']['max']} A")
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {d['n_with_a_resolution']}/"
          f"{d['n_units']} with a resolution, "
          f"{d['span']['min']}-{d['span']['max']} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
