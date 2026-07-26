"""Compile the linear readout of the algebraic field from the training fold.

One regularised solve of a 172x172 symmetric system, plus the operating-point
quantile that maximises TRAINING F1. No gradient, no iteration, no
auto-differentiation, and no view of the test fold. Running this twice on the
same bytes yields a byte-identical artifact.

Unlike ``compile_algebraic_field.py`` this one does fit real coefficients, and
the artifact says so in its ``fitting`` field. The readout family and the gate
were fixed beforehand on a cluster-disjoint half of the training fold
(``tools/final_readout_select.py``); this script only fits the winner on the
full fold.

Fail-closed guards, in order:
  1. train and test manifests must share no PDB-chain unit,
  2. they must share no MMseqs2 cluster id,
  3. the expanded cache must carry the training-compiled propensity table.

Usage:
  PYTHONPATH=src python3.12 tools/build_expanded_cache.py
  PYTHONPATH=src python3.12 tools/compile_algebraic_field_linear.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_field_linear import code_sha256, compile_field

from compile_algebraic_field import (  # noqa: E402  (sibling tool)
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    _sha256,
    assert_no_leak,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
CACHE_TEST = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"
OUT = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    led = assert_no_leak()
    print(f"leak gate OK: {led['train_units']} train units / "
          f"{led['train_clusters']} clusters vs {led['test_units']} test units / "
          f"{led['test_clusters']} clusters, 0 shared", flush=True)

    if not CACHE_TRAIN.exists():
        raise SystemExit(f"missing {CACHE_TRAIN}; run tools/build_expanded_cache.py")
    z = np.load(CACHE_TRAIN, allow_pickle=False)
    prop = z["propensity_table"]

    if CACHE_TEST.exists():
        prop_te = np.load(CACHE_TEST, allow_pickle=False)["propensity_table"]
        if not np.array_equal(prop, prop_te):
            raise SystemExit(
                "PROPENSITY LEAK: the test cache carries a different table than "
                "the training fold, so it was not compiled on train alone")
        print("propensity gate OK: test cache reuses the train table byte for byte",
              flush=True)

    t0 = time.perf_counter()
    field = compile_field(
        z["X"], z["y"], z["ctr"], z["n_res_per"],
        tuple(str(s) for s in z["names"]), prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field["cluster_ledger"] = led
    field["test_manifest_sha256"] = _sha256(TEST_MANIFEST)
    field["compile_seconds"] = round(time.perf_counter() - t0, 2)

    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    w = np.asarray(field["coefficients"])
    names = field["wire_names"]
    top = np.argsort(-np.abs(w))[:10]
    st = field["operating_point"]
    print(f"wires={field['n_wires']} ridge={field['ridge']:g} "
          f"|w|max={np.abs(w).max():.4f}")
    print("largest coefficients: "
          + ", ".join(f"{names[i]}={w[i]:+.3f}" for i in top))
    print(f"operating point q={st['q']:.2f} (train F1 {st['train_f1_at_q']:.4f})")
    print(f"wrote -> {args.out.relative_to(ROOT)} "
          f"({args.out.stat().st_size/1e3:.0f} kB, "
          f"{field['compile_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
