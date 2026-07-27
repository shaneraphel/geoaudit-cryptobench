#!/usr/bin/env python3
"""Compile the table field from the official training fold, and only from it.

Three gates run before anything is written, because every one of them has caught
a real defect in this repository at some point.

The cluster gate refuses to compile if any MMseqs2 cluster appears in both folds.
The propensity gate refuses if the test cache carries a different residue-
frequency table than the training cache, which is the one place a label could
leak into a wire. The wire gate refuses if the two caches disagree on the number
or the names of the wires.

What lands in the artifact is auditable by hand: the two integers per cell, the
integer fan-out, the propensity counts, the table definitions, and the SHA-256 of
both the manifest and every source file the numbers depend on. Cell contents are
stored as the counts themselves rather than as ratios so that a reader can divide
them and get the same field.

Usage: PYTHONPATH=src:tools python3.12 tools/compile_table_field.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.table_field import code_sha256, compile_field

from compile_algebraic_field import (  # noqa: E402  (sibling tool)
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    _sha256,
    assert_no_leak,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CACHE_TEST = ROOT / "data/cryptobench_apo/_wide_cache_test.npz"
OUT = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    led = assert_no_leak()
    print(f"cluster gate OK: {led['train_units']} train units / "
          f"{led['train_clusters']} clusters vs {led['test_units']} test "
          f"units / {led['test_clusters']} clusters, 0 shared", flush=True)

    if not CACHE_TRAIN.exists():
        raise SystemExit(f"missing {CACHE_TRAIN}; run tools/build_wide_cache.py")
    z = np.load(CACHE_TRAIN, allow_pickle=False)
    prop = z["propensity_table"]
    names = tuple(str(s) for s in z["names"])

    if CACHE_TEST.exists():
        zt = np.load(CACHE_TEST, allow_pickle=False)
        if not np.array_equal(prop, zt["propensity_table"]):
            raise SystemExit(
                "PROPENSITY LEAK: the test cache carries a different table "
                "than the training fold, so it was not compiled on train alone")
        te_names = tuple(str(s) for s in zt["names"])
        if te_names != names:
            raise SystemExit(
                f"WIRE MISMATCH: train has {len(names)} wires, test has "
                f"{len(te_names)}; the caches were not built by the same code")
        print("propensity gate OK: test cache reuses the train table byte for "
              "byte", flush=True)
        print(f"wire gate OK: {len(names)} identically named wires in both "
              f"caches", flush=True)

    t0 = time.perf_counter()
    field = compile_field(
        z["X"], z["y"], z["ctr"], z["n_res_per"], names, prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field["cluster_ledger"] = led
    field["test_manifest_sha256"] = _sha256(TEST_MANIFEST)
    field["compile_seconds"] = round(time.perf_counter() - t0, 2)

    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")

    m = np.asarray(field["multiplicity"])
    tot = np.asarray(field["cell_total"])
    op = field["operating_point"]
    print(f"\n{field['n_wires']} wires -> {len(field['tables'])} tables of "
          f"{4 ** field['table_width']} cells = {field['n_cells']} cells")
    print(f"{field['n_cells_never_addressed']} cells never addressed "
          f"({100.0 * field['n_cells_never_addressed'] / field['n_cells']:.2f}"
          f"%), {tot[tot > 0].mean():.0f} training residues per occupied cell")
    print(f"integer fan-out in [-{field['fan_out_cap']}, "
          f"{field['fan_out_cap']}] from a ridge-{field['ridge']:g} solve: "
          f"{field['n_tables_with_nonzero_fanout']} tables carry a non-zero "
          f"weight, total {field['total_fan_out']}, "
          f"|m| median {int(np.median(np.abs(m[m != 0])))}")
    print(f"operating point q={op['q']:.2f} (train F1 "
          f"{op['train_f1_at_q']:.4f})")
    print(f"wrote -> {args.out.relative_to(ROOT)} "
          f"({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{field['compile_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
