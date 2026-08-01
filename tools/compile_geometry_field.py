#!/usr/bin/env python3
"""Compile the geometry field from the official training fold, and only from it.

What this compiles
------------------
The same construction ``compile_table_field.py`` produces --- same table
topology, same quantisation, same fan-out cap, same ridge, same operating-point
rule --- over 645 + 624 columns instead of 645. The extra columns are the four
wire families whose lift was measured at +0.0121 on twelve cluster-disjoint
halvings of the training fold, positive on 12 of 12, against a control arm that
spends the same table budget on already-deployed wires and lands at −0.0017.

Only the columns differ, which is what makes that lift a statement about the
columns rather than about the construction.

Why the column order is asserted rather than assumed
-----------------------------------------------------
The lift was measured on a specific concatenation: the 645 wide columns, then
backbone, side chain, void topology, and the temperature-factor half of the
displacement family. A field compiled over the same columns in a different order
is a different field --- the table bank pairs columns by index --- and it would
score plausibly while being a detector nobody measured. So the block widths are
checked against the four caches and the total against
``geometry_wires.N_COLUMNS``, and the compile refuses on a mismatch.

Why the alternate-conformer columns are absent
-----------------------------------------------
They were measured and are not there: ``displacement alt 48`` scored +0.00052
against its own control at +0.00052, the same number to five decimal places.
Carrying them would spend 384 tables on a measured zero, and it would push the
block past the 645 deployed wires, which the attachment harness refuses.

Leakage
-------
The cluster gate from ``compile_algebraic_field`` runs first and unchanged: the
training and test unit sets and their cluster surrogates must be disjoint. Every
column here is computed from the training receptors only, and the four caches the
geometry block is read from were built by tools that walk the training manifest.
No test-fold or external row enters this file.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.geometry_wires import (
    FAMILIES,
    N_COLUMNS as N_GEOMETRY,
    N_DISP_CARRIED,
)
from pocket_bench.methods.table_field import code_sha256, compile_field

from compile_algebraic_field import (  # noqa: E402  (sibling tool)
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    _sha256,
    assert_no_leak,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
OUT = ROOT / "data/cryptobench_apo/GEOMETRY_FIELD.json"

# (family, cache file, how many of its quantities to carry per aggregation).
# The order is the order the lift was measured in and is not cosmetic.
CACHES = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", N_DISP_CARRIED),
)


def _geometry_block(n_rows: int) -> tuple[np.ndarray, list[str]]:
    """The 624 columns from the four caches, in the measured order."""
    blocks, names = [], []
    for fam, fname, carry in CACHES:
        path = ROOT / "data/cryptobench_apo" / fname
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)}; build it with "
                f"tools/{fam}_wires.py")
        z = np.load(path, allow_pickle=False)
        C = z["C"]
        cols = [str(s) for s in z["names"]]
        z.close()
        if C.shape[0] != n_rows:
            raise SystemExit(
                f"{fname} holds {C.shape[0]} rows and the wide cache holds "
                f"{n_rows}; the two were not built over the same residues")
        if carry is not None:
            per_agg = C.shape[1] // 3
            take = [a * per_agg + q for a in range(3) for q in range(carry)]
            C = C[:, take]
            cols = [cols[i] for i in take]
        blocks.append(C)
        names.extend(f"{fam}~{c}" for c in cols)
    X = np.concatenate(blocks, axis=1)
    want = sum(3 * len(q) for _f, q in FAMILIES)
    if X.shape[1] != want or want != N_GEOMETRY:
        raise SystemExit(
            f"the geometry block is {X.shape[1]} columns, the family widths sum "
            f"to {want}, and geometry_wires declares {N_GEOMETRY}. A field "
            f"compiled over a different set of columns than the lift was "
            f"measured on is a detector nobody measured")
    return X, names


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
    Xw = z["X"]
    prop = z["propensity_table"]
    names = [str(s) for s in z["names"]]
    y, ctr, n_res_per = z["y"], z["ctr"], z["n_res_per"]
    z.close()
    print(f"wires: {Xw.shape[1]} columns over {Xw.shape[0]} residues",
          flush=True)

    Xg, gnames = _geometry_block(Xw.shape[0])
    print(f"geometry: {Xg.shape[1]} columns "
          f"({', '.join(f'{f} {3 * len(q)}' for f, q in FAMILIES)})", flush=True)

    X = np.concatenate([np.asarray(Xw, dtype=np.float32),
                        np.asarray(Xg, dtype=np.float32)], axis=1)
    all_names = tuple(names + gnames)
    if len(all_names) != X.shape[1]:
        raise SystemExit(
            f"{len(all_names)} names for {X.shape[1]} columns")

    t0 = time.perf_counter()
    field = compile_field(
        X, y, ctr, n_res_per, all_names, prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field["cluster_ledger"] = led
    field["test_manifest_sha256"] = _sha256(TEST_MANIFEST)
    field["compile_seconds"] = round(time.perf_counter() - t0, 2)
    field["detector"] = "geometry_field"
    field["what_is_different_from_table_field"] = (
        "the columns, and nothing else. Same table topology, quantisation, "
        "fan-out cap, ridge and operating-point rule; 645 deployed wires plus "
        "624 columns from four families measured at +0.0121 on twelve "
        "cluster-disjoint halvings of the training fold")
    field["geometry_families"] = {
        f: 3 * len(q) for f, q in FAMILIES}
    field["alternate_conformer_columns_excluded"] = (
        "the 48 alternate-conformer and occupancy columns of the displacement "
        "family measured +0.00052 against their own control arm at +0.00052, "
        "the same number to five decimals. Carrying them would spend 384 tables "
        "on a measured zero")

    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")

    m = np.asarray(field["multiplicity"])
    tot = np.asarray(field["cell_total"])
    op = field["operating_point"]
    print(f"\n{field['n_wires']} columns -> {len(field['tables'])} tables of "
          f"{4 ** field['table_width']} cells = {field['n_cells']} cells")
    print(f"{field['n_cells_never_addressed']} cells never addressed "
          f"({100.0 * field['n_cells_never_addressed'] / field['n_cells']:.2f}"
          f"%), {tot[tot > 0].mean():.0f} training residues per occupied cell")
    print(f"integer fan-out in [-{field['fan_out_cap']}, "
          f"{field['fan_out_cap']}]: "
          f"{field['n_tables_with_nonzero_fanout']} tables carry a non-zero "
          f"weight, total {field['total_fan_out']}")
    print(f"operating point q={op['q']:.2f} (train F1 "
          f"{op['train_f1_at_q']:.4f})")
    print(f"wrote -> {args.out.relative_to(ROOT)} "
          f"({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{field['compile_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
