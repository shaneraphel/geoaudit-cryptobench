#!/usr/bin/env python3.12
"""Do gauge-invariant spectral quantities separate the nineteen inverted units?

The question, stated precisely
------------------------------
On the official fold, ``max(seam, -seam)`` chosen per unit reaches 0.874 against
pLM-NN's 0.8235 (``SEAM_INVERT_ORACLE.json``). Nineteen of 192 units have a
negative label-score correlation; inverting those would rescue eleven of them.
A one-dimensional label-free switch was built and **failed**, dropping the mean
to 0.802 at a precision of 0.2 (``POLARITY_SWITCH_SCREEN.json``), and the best
separating covariate anyone has found reaches Cohen's ``d`` of about 0.48.

The hypothesis this tests is that the polarity defect is the shadow of an
**unfixed gauge**. Every spectral column in the deployed bank is either
gauge-fixed by an absolute value or built from quantities that discard the
relations between residues. If the sign the detector cannot choose is the same
sign the eigensolver cannot choose, a ``G_L``-invariant chain-level statistic
should separate the nineteen where a gauge-fixed one does not.

What this is and is not
-----------------------
It is a **diagnostic**, exactly parallel to ``POLARITY_SWITCH_SCREEN`` and
reported the same way: Cohen's ``d`` per covariate, on a fold that has been read
many times. Finding a separator here does **not** produce a method. The
covariate would then have to be turned into a rule and validated on the training
fold, because a threshold chosen on the units it is scored on is an upper bound
and not a result — that is the specific mistake ``POLARITY_SWITCH_SCREEN``
records in its own note field.

The falsifier, written before the run: no gauge-invariant chain statistic
reaches ``|d| > 0.6``, which would leave the polarity axis where it already is
and say the gauge reading is wrong.

``clinical_grade`` is false. ``reads_test_fold`` is true.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.polarity_gauge_probe.v1"
N_JOBS = min(9, os.cpu_count() or 4)
ORACLE = ROOT / "results/official_fold/SEAM_INVERT_ORACLE.json"
OUT = ROOT / "results/official_fold/POLARITY_GAUGE_PROBE.json"
SKIP = frozenset({"HOH", "WAT", "DOD"})


def _chain_stats(args: tuple[str, str, str]) -> tuple[str, dict]:
    """Chain-level aggregates of the gauge-invariant residue columns."""
    unit, path, chain = args
    from pocket_bench.methods.spectral_gauge import COLUMNS, compute
    from pocket_bench.pdb_io import parse_pdb_atoms
    try:
        atoms = parse_pdb_atoms(Path(path).read_text())
        poly: dict = {}
        for a in atoms:
            if (a["chain"] != chain or a["element"] == "H"
                    or a["resname"] in SKIP):
                continue
            poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
        order = sorted(poly)
        X = compute([poly[k] for k in order], [k[0] for k in order])
        if X.shape[0] == 0:
            return unit, {"error": "empty"}
        out = {}
        for j, name in enumerate(COLUMNS):
            v = X[:, j]
            out[f"mean~{name}"] = float(v.mean())
            out[f"std~{name}"] = float(v.std())
        # Two chain-level summaries that are not per-residue aggregates: how
        # bipartite the Fiedler cut is, and how much of the chain sits on the
        # nodal boundary. Both are gauge-invariant by construction.
        from pocket_bench.methods.spectral_gauge import _COL
        out["frac_on_nodal_boundary"] = float(
            (X[:, _COL["on_nodal_boundary_v2"]] > 0).mean())
        out["nodal_balance"] = float(
            X[:, _COL["nodal_domain_size_v2"]].min()
            / max(X[:, _COL["nodal_domain_size_v2"]].max(), 1.0))
        out["n_residues"] = int(X.shape[0])
        return unit, out
    except Exception as exc:  # noqa: BLE001
        return unit, {"error": f"{type(exc).__name__}: {exc}"}


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    oracle = json.loads(ORACLE.read_text())
    want_inv = {u["unit"] for u in oracle["worst_neg_corr"]}
    n_neg = int(oracle["n_neg_corr"])

    man = json.loads((ROOT / "data/cryptobench_apo/"
                      "official_manifest.json").read_text())
    jobs = [(f"{e['pdb']}_{e['chain']}", str(ROOT / e["receptor_path"]),
             e["chain"]) for e in man["entries"]]

    stats: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_chain_stats, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            unit, rec = fut.result()
            stats[unit] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

    good = {u: r for u, r in stats.items() if "error" not in r}
    keys = sorted({k for r in good.values() for k in r
                   if isinstance(r.get(k), float)})
    inv = [u for u in good if u in want_inv]
    nor = [u for u in good if u not in want_inv]

    seps = []
    for k in keys:
        va = np.asarray([good[u][k] for u in inv], dtype=float)
        vb = np.asarray([good[u][k] for u in nor], dtype=float)
        if not (np.isfinite(va).all() and np.isfinite(vb).all()):
            continue
        d = _cohen_d(va, vb)
        seps.append({"covariate": k, "cohen_d": round(d, 4),
                     "mean_inverted": float(va.mean()),
                     "mean_normal": float(vb.mean())})
    seps.sort(key=lambda r: -abs(r["cohen_d"]))
    best = seps[0] if seps else None

    # ---- how surprising is the best of 54 covariates on 12 units? ----------
    # With this many covariates and a group of twelve, the maximum |d| under the
    # null is not small, and quoting the winner without this number is the
    # multiple-comparison version of choosing a threshold on the fold it is
    # scored on. The permutation keeps the covariate matrix fixed and shuffles
    # which units are called inverted, so it preserves every correlation among
    # the covariates and asks only whether the label assignment matters.
    rng = np.random.default_rng(20260802)
    units_all = list(good)
    M = np.asarray([[good[u][k] for k in keys] for u in units_all], dtype=float)
    n_inv = len(inv)
    n_perm = 20000
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        idx = rng.permutation(len(units_all))
        A_, B_ = M[idx[:n_inv]], M[idx[n_inv:]]
        va, vb = A_.var(axis=0, ddof=1), B_.var(axis=0, ddof=1)
        s = np.sqrt(((n_inv - 1) * va + (len(units_all) - n_inv - 1) * vb)
                    / (len(units_all) - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(s > 0, (A_.mean(axis=0) - B_.mean(axis=0)) / s, 0.0)
        null_max[p] = np.abs(dd).max()
    obs = abs(best["cohen_d"]) if best else 0.0
    p_famwise = float((null_max >= obs).mean())
    permutation = {
        "n_permutations": n_perm,
        "n_permutations_definition": "random reassignments of which units are "
                                     "called inverted, group sizes held fixed",
        "statistic": "max over covariates of |Cohen's d|",
        "observed": round(obs, 4),
        "null_median": round(float(np.median(null_max)), 4),
        "null_95th_percentile": round(float(np.percentile(null_max, 95)), 4),
        "family_wise_p": p_famwise,
        "reading": (
            "this p is family-wise over all covariates tested, so it already "
            "accounts for having chosen the best of them; it does NOT account "
            "for the covariates having been designed after the nineteen units "
            "were known to exist"),
    }

    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": True,
        "why_not_confirmatory": (
            "diagnostic on a fold that has been read many times; a covariate "
            "that separates here has been found on the units it is scored on "
            "and would still need a train-fold rule and validation before it "
            "could be a method"),
        "question": ("whether a gauge-invariant chain statistic separates the "
                     "units whose cryptic residues the field anti-ranks"),
        "n_units_scored": len(good),
        "n_units_scored_definition": ("official-fold units for which the "
                                      "spectral-gauge family returned a finite "
                                      "matrix"),
        "n_units_inverted": len(inv),
        "n_units_inverted_definition": (
            "units listed in SEAM_INVERT_ORACLE.worst_neg_corr, the ones whose "
            f"label-score correlation is most negative; the oracle reports "
            f"{n_neg} negative-correlation units in total and this is the "
            "subset it enumerated"),
        "n_covariates_tested": len(seps),
        "n_covariates_tested_definition": (
            "chain-level float statistics derived from the 26 gauge-invariant "
            "residue columns: a mean and a standard deviation of each, plus "
            "two whole-chain nodal summaries"),
        "prior_best_separator_cohen_d": 0.4848,
        "prior_best_separator": "top10_mass, from POLARITY_SWITCH_SCREEN.json",
        "falsifier_written_before_the_run": (
            "no gauge-invariant chain statistic reaches |d| > 0.6, which would "
            "leave the polarity axis where it is and say the gauge reading is "
            "wrong"),
        "best": best,
        "beats_prior_best": (bool(abs(best["cohen_d"]) > 0.4848)
                             if best else False),
        "clears_the_falsifier": (bool(abs(best["cohen_d"]) > 0.6)
                                 if best else False),
        "permutation_test": permutation,
        "survives_multiple_comparisons": bool(permutation["family_wise_p"] < 0.05),
        "top_20_separators": seps[:20],
        "n_errors": len(stats) - len(good),
        "errors": {u: r["error"] for u, r in stats.items() if "error" in r},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)
    print(f"\n{len(inv)} inverted / {len(nor)} normal units, "
          f"{len(seps)} covariates\n")
    print(f"{'covariate':44s} {'cohen_d':>9s}  {'inverted':>12s} "
          f"{'normal':>12s}")
    for r in seps[:14]:
        print(f"{r['covariate']:44s} {r['cohen_d']:+9.4f}  "
              f"{r['mean_inverted']:12.4f} {r['mean_normal']:12.4f}")
    print("\npermutation test over all covariates:")
    print(json.dumps(out["permutation_test"], indent=2))
    print(f"\nprior best |d| = 0.485 (top10_mass). "
          f"beats prior: {out['beats_prior_best']}.  "
          f"clears falsifier |d|>0.6: {out['clears_the_falsifier']}.  "
          f"survives multiple comparisons: "
          f"{out['survives_multiple_comparisons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
