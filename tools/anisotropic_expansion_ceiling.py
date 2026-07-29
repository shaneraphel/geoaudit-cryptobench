#!/usr/bin/env python3
"""Does giving the spatial expansion a direction lift the ceiling? Training folds only.

Why this axis and not another
-----------------------------
Two measurements in this repository point here together. Going from the 35 local
invariants to the 645 wires is worth +0.025 on the Fisher ceiling, and that is the
spatial expansion alone, since the wires are nothing but 43 local quantities put
through fifteen neighbourhood statistics. Going from those 645 wires to 912 by
adding 190 operator and 77 chain descriptors is worth +0.0005 at best and -0.0009
pooled (WIDE_BANK_CEILING.json). One axis carries essentially everything and the
other carries nothing, so the question worth asking is about the first one.

And the first one is isotropic. Every statistic in the expansion is taken over a
ball: the mean of a wire over the neighbours within r, its spread, the residue's
difference from that mean, its rank among them. A ball has no direction, so the
whole 645-wire bank cannot distinguish a residue whose hydrophobic neighbours all
lie on one side from one whose hydrophobic neighbours surround it. For a cryptic
site that distinction is close to the point: the mouth of a cleft is a place where
the protein is on one side and is not on the other, and the residues lining it are
asymmetric in exactly this sense while a buried residue and a flat-surface residue
are not.

What is added
-------------
One statistic per wire per radius, and it is a difference of two means over
complementary half-neighbourhoods.

For residue i at radius r let N be the neighbours within r and let

    u_i = normalise( mean_{j in N} (c_j - c_i) )

which points from the residue toward the centre of mass of its own
neighbourhood: into the protein for a surface residue, and undefined in the limit
of a perfectly surrounded one, where the mean displacement vanishes. Split N by
the sign of (c_j - c_i) . u_i into the bulkward half and the outward half, and for
each wire x report

    A_r[x]_i = mean_{bulkward} x_j  -  mean_{outward} x_j

This is a comparison, a dot product, a sum and a division. There is no fitted
quantity, no random number and no iteration, so it stays inside what the counting
construction is allowed to read. It is also exactly zero for a residue whose
neighbourhood is symmetric, which is the correct answer there rather than a
missing value: such a residue has no inside and outside to tell apart.

The degenerate case is handled and counted rather than smoothed over. When
||mean displacement|| falls below a floor the direction is noise, the split would
be arbitrary, and the column is set to zero for that residue. The artifact records
how often that happened, because a feature that is zero on most residues is a
feature that is not doing anything and the count is what says so.

Discipline
----------
Baseline is the 645 wires the deployed field reads. Fit and pick halves are
cluster-disjoint and both inside the official training fold; the same twelve
seeds, ridge, gate and metric as WIDE_BANK_CEILING.json, so the two lifts are
comparable. No test residue and no external unit is read. A Fisher discriminant
solves for arbitrary real coefficients over unquantised inputs, so every number
here is an upper bound on a counting field rather than a prediction of one.

Usage: PYTHONPATH=src:tools python3.12 tools/anisotropic_expansion_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import (  # noqa: E402  (sibling tool)
    N_SPLITS,
    RIDGE,
    SEED,
    fisher,
    gated,
)
from select_architecture_on_train import (  # noqa: E402
    cluster_half_split,
    per_unit_auc,
)

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.methods.expanded_descriptors import CHEM_NAMES, chemical_wires
from pocket_bench.methods.sequence_wires import apply_propensity, propensity_table
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.anisotropic_expansion_ceiling.v1"
CASCADE = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_aniso_cache_train.npz"
OUT = ROOT / "results/architecture_sweep/ANISOTROPIC_EXPANSION_CEILING.json"

# The three radii the isotropic difference and rank statistics already use, so
# that the anisotropic column is the same neighbourhood seen with a direction
# rather than a different neighbourhood.
ASYM_RADII = (6.0, 14.0, 20.0)
# Below this the neighbourhood's mean displacement is shorter than a tenth of an
# angstrom and the direction it defines is numerical noise, not geometry.
DIRECTION_FLOOR = 0.1
BLOCK = 768


def locals_43(z) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """The 43 local quantities build_wide_cache expands, rebuilt the same way."""
    prop = propensity_table(z["codes"], z["y"])
    local = np.concatenate(
        [z["F"], chemical_wires(z["codes"]),
         apply_propensity(z["codes"], prop)[:, None]], axis=1)
    names = tuple(FEATURE_NAMES) + CHEM_NAMES + ("propensity",)
    if local.shape[1] != len(names):
        raise SystemExit(f"assembled {local.shape[1]} local quantities but named "
                         f"{len(names)}; build_wide_cache and this tool have "
                         f"drifted apart and the wires would not correspond")
    return local, names, prop


def asymmetry(local: np.ndarray, ctr: np.ndarray, n_res_per) -> tuple[
        np.ndarray, dict]:
    """Bulkward mean minus outward mean, per wire per radius."""
    C = local.shape[1]
    out = np.zeros((local.shape[0], C * len(ASYM_RADII)), dtype=np.float64)
    n_degenerate = {float(r): 0 for r in ASYM_RADII}
    n_total = 0
    off = 0
    t0 = time.perf_counter()
    for u, n in enumerate(n_res_per):
        n = int(n)
        c = np.asarray(ctr[off:off + n], dtype=np.float64)
        blk = np.asarray(local[off:off + n], dtype=np.float64)
        n_total += n
        for k, r in enumerate(ASYM_RADII):
            r2 = r * r
            col = out[off:off + n, k * C:(k + 1) * C]
            for i in range(0, n, BLOCK):
                j = min(i + BLOCK, n)
                dv = c[i:j, None, :] - c[None, :, :]          # i - j
                d2 = (dv * dv).sum(-1)
                a = d2 <= r2
                # The displacement from i to each neighbour is -dv. Its mean over
                # the neighbourhood is the direction from the residue toward the
                # centre of mass of its own neighbours.
                cnt = np.maximum(a.sum(1), 1)[:, None]
                mdisp = (-dv * a[:, :, None]).sum(1) / cnt
                norm = np.linalg.norm(mdisp, axis=1)
                live = norm > DIRECTION_FLOOR
                n_degenerate[float(r)] += int((~live).sum())
                if not live.any():
                    continue
                uhat = np.zeros_like(mdisp)
                uhat[live] = mdisp[live] / norm[live, None]
                proj = (-dv * uhat[:, None, :]).sum(-1)       # (i, j)
                bulk = a & (proj > 0.0)
                outw = a & (proj <= 0.0)
                nb = np.maximum(bulk.sum(1), 1)[:, None]
                no = np.maximum(outw.sum(1), 1)[:, None]
                mb = (bulk.astype(np.float64) @ blk) / nb
                mo = (outw.astype(np.float64) @ blk) / no
                col[i:j] = np.where(live[:, None], mb - mo, 0.0)
        off += n
        if (u + 1) % 100 == 0:
            print(f"    {u + 1}/{len(n_res_per)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    diag = {
        "n_residues": n_total,
        "n_degenerate_per_radius": {str(k): v for k, v in n_degenerate.items()},
        "fraction_degenerate_per_radius": {
            str(k): round(v / max(n_total, 1), 6)
            for k, v in n_degenerate.items()},
        "direction_floor_angstrom": DIRECTION_FLOOR,
        "why_zero_and_not_missing": (
            "a residue whose neighbourhood has no net displacement has no inside "
            "and outside to tell apart, so zero is the answer rather than a gap. "
            "The count is reported because a column that is zero on most "
            "residues is a column doing nothing, and only the count says so"),
    }
    return out, diag


def build_or_load(workers: int) -> tuple[np.ndarray, dict, tuple[str, ...]]:
    z = np.load(CASCADE, allow_pickle=False)
    local, names, _prop = locals_43(z)
    aniso_names = tuple(f"{nm}~asym{int(r)}" for r in ASYM_RADII for nm in names)
    if CACHE.exists():
        zc = np.load(CACHE, allow_pickle=False)
        if (zc["A"].shape == (len(local), len(aniso_names))
                and list(zc["names"]) == list(aniso_names)):
            print(f"reusing {CACHE.relative_to(ROOT)}  {zc['A'].shape}")
            return zc["A"], json.loads(str(zc["diag"])), aniso_names
        print("the cached anisotropic bank does not match; rebuilding")
    print(f"computing {len(aniso_names)} anisotropic columns over "
          f"{len(z['n_res_per'])} chains", flush=True)
    A, diag = asymmetry(local, z["ctr"], z["n_res_per"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, A=A.astype(np.float32),
                        names=np.array(aniso_names), diag=json.dumps(diag))
    print(f"wrote {CACHE.relative_to(ROOT)}  {A.shape}")
    return A.astype(np.float32), diag, aniso_names


def load_wide():
    z = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    missing = [u for u in units if u not in cluster_of]
    if missing:
        raise SystemExit(f"{len(missing)} cached units absent from the manifest")
    return z["X"], z["y"], z["n_res_per"], z["ctr"], units, cluster_of


def evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits):
    rows: dict[str, list[float]] = {k: [] for k in banks}
    base_key = next(iter(banks))
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        ypick, ctr_pick = y[pick], ctr[pick]
        t0 = time.perf_counter()
        for name, X in banks.items():
            sc = fisher(X[fit], y[fit], X[pick])
            rows[name].append(per_unit_auc(gated(sc, ctr_pick, n_pick),
                                           ypick, n_pick))
        print(f"  split {s + 1}/{n_splits}  " + "  ".join(
            f"{k.split()[0]} {rows[k][-1]:.4f}" for k in banks)
            + f"  {time.perf_counter() - t0:.0f}s", flush=True)
        _ = base_key
    return {k: np.array(v) for k, v in rows.items()}


def build(n_splits: int, workers: int) -> dict:
    A, diag, aniso_names = build_or_load(workers)
    W, y, n_res, ctr, units, cluster_of = load_wide()
    if len(A) != len(W):
        raise SystemExit(
            f"the anisotropic bank has {len(A)} rows and the wide cache has "
            f"{len(W)}; they came from different manifests and concatenating "
            f"them would pair each residue with another one's descriptors")

    per_radius = {}
    C = len(aniso_names) // len(ASYM_RADII)
    for k, r in enumerate(ASYM_RADII):
        per_radius[f"wide+asym{int(r)} {W.shape[1] + C}"] = np.concatenate(
            [W, A[:, k * C:(k + 1) * C]], axis=1)
    banks = {f"wide {W.shape[1]}": W, **per_radius,
             f"wide+asym_all {W.shape[1] + A.shape[1]}":
                 np.concatenate([W, A], axis=1)}

    print(f"\n{len(units)} units, {len(W):,} residues, {W.shape[1]} wires as "
          f"the baseline, {A.shape[1]} anisotropic columns added")
    for r, f in diag["fraction_degenerate_per_radius"].items():
        print(f"  at {float(r):.0f} A the direction is degenerate on "
              f"{100 * f:.2f}% of residues")
    print(f"\nFisher ceiling over {n_splits} cluster-disjoint halvings:")
    curves = evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits)

    base_key = f"wide {W.shape[1]}"
    lifts = {}
    for k, v in curves.items():
        if k == base_key:
            continue
        d = v - curves[base_key]
        lifts[k] = {"mean": round(float(d.mean()), 6),
                    "min": round(float(d.min()), 6),
                    "max": round(float(d.max()), 6),
                    "n_splits_positive": int((d > 0).sum()),
                    "n_splits": int(len(d)),
                    "positive_on_every_split": bool((d > 0).all())}
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the expansion is the only axis measured to carry weight and "
                    "it is isotropic; does resolving its neighbourhood "
                    "statistics into a bulkward and an outward half lift the "
                    "ceiling that bounds the counting field",
        "why_this_axis": (
            "local 35 to wide 645 is +0.025 on this measurement and wide 645 to "
            "912 by adding 190 operator and 77 chain descriptors is +0.0005 at "
            "best and -0.0009 pooled. One axis carries essentially everything. "
            "That axis is a ball, and a ball has no direction, so the whole bank "
            "cannot tell a residue whose hydrophobic neighbours lie on one side "
            "from one whose hydrophobic neighbours surround it. The mouth of a "
            "cleft is a place where the protein is on one side and is not on the "
            "other"),
        "construction": {
            "statistic": "mean of the wire over the bulkward half-neighbourhood "
                         "minus its mean over the outward half",
            "direction": "u_i = normalise(mean over neighbours of (c_j - c_i)), "
                         "which points from the residue toward the centre of "
                         "mass of its own neighbourhood",
            "split": "sign of (c_j - c_i) . u_i",
            "radii_angstrom": [float(r) for r in ASYM_RADII],
            "why_these_radii": "the three the isotropic difference and rank "
                               "statistics already use, so the new column is "
                               "the same neighbourhood seen with a direction "
                               "rather than a different neighbourhood",
            "operations": "a comparison, a dot product, a sum and a division; no "
                          "fitted quantity, no random number, no iteration, so "
                          "it stays inside what the counting construction reads",
            "n_columns": len(aniso_names),
        },
        "degenerate_direction": diag,
        "baseline": {"name": base_key, "n_wires": int(W.shape[1])},
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "readout": "ridge Fisher discriminant fitted on the fit half, "
                       "columns standardised there, evaluated on the pick half",
            "ridge": RIDGE,
            "gate": "the multi-scale spatial gate the detectors use",
            "metric": "mean per-unit ROC-AUC on the pick half",
            "comparable_to": "results/architecture_sweep/WIDE_BANK_CEILING.json, "
                             "same seeds, ridge, gate and metric",
            "why_a_ceiling": "a Fisher discriminant solves for arbitrary real "
                             "coefficients over unquantised inputs; a counting "
                             "field bands every input into four levels and adds "
                             "integers, so this bounds it rather than predicting "
                             "it",
        },
        "ceilings": {k: {"mean": round(float(v.mean()), 6),
                         "min": round(float(v.min()), 6),
                         "max": round(float(v.max()), 6)}
                     for k, v in curves.items()},
        "lift_over_the_wide_bank": lifts,
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in curves.items()},
        "n_units": len(units),
        "n_residues": int(len(W)),
        "n_positive_residues": int(y.sum()),
    }


def _report(d: dict) -> None:
    base = d["baseline"]["name"]
    print(f"\n{'bank':<34} {'ceiling':>9} {'lift':>9} {'splits+':>8}")
    print(f"{base:<34} {d['ceilings'][base]['mean']:>9.4f}")
    for k, L in sorted(d["lift_over_the_wide_bank"].items(),
                       key=lambda kv: -kv[1]["mean"]):
        print(f"{k:<34} {d['ceilings'][k]['mean']:>9.4f} {L['mean']:>+9.4f} "
              f"{L['n_splits_positive']:>5}/{L['n_splits']}")
    best = max(d["lift_over_the_wide_bank"].values(), key=lambda L: L["mean"])
    rival = 0.0005
    print(f"\nthe best generated descriptor family managed {rival:+.4f} against "
          f"this same baseline.")
    if best["mean"] <= 0:
        print("direction buys nothing either. The expansion's value is in the "
              "radii, not in the geometry of the neighbourhood.")
    elif best["mean"] < rival:
        print(f"direction buys {best['mean']:+.4f}, no better. This axis is "
              f"closed too.")
    else:
        print(f"direction buys {best['mean']:+.4f} on "
              f"{best['n_splits_positive']}/{best['n_splits']} splits, which "
              f"beats the descriptor families. Worth a counting construction.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    d = build(a.splits, a.workers)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    _report(d)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
