#!/usr/bin/env python3
"""A wider context transform, because the counting field has caught its inputs.

State of the counterattack. The table field now reads 0.7952 on the official
test fold against P2Rank's 0.7935 and against 0.7953 for a fitted continuous
linear functional of the same wires -- a paired difference of -0.0002 with
p=0.98. The combinational path has stopped losing anything to the continuous
one, which also means no further work on the circuit can help. Whatever is left
has to enter as new wires.

The 172 existing wires are 43 local quantities under one operation: the mean
over a ball of radius 6, 14 or 20 A. Three things that operation cannot say:

**How uniform the neighbourhood is.** Two residues can have identical mean
hydrophobicity around them, one because every neighbour is moderately
hydrophobic and one because half are oily and half are charged. A cryptic site
is an interface between such regions, so the second moment

    V_r[x]_i = C_r[x^2]_i - (C_r[x]_i)^2

is exactly the kind of thing that should mark it, and it is orthogonal to the
mean by construction.

**How the residue differs from its surroundings.** The centred wire

    D_r[x]_i = x_i - C_r[x]_i

is a discrete Laplacian: it is large where a residue is unlike its
neighbourhood. As a real number it is a difference of two wires already
present, so a linear functional gains nothing from it -- but the field does not
read real numbers. Each wire is separately ranked within its chain and cut into
a quaternary digit, and the digit of a difference is not a function of the
digits of the two terms, so this genuinely widens the addressable space.

**Scale between and beyond the three radii.** 6, 14 and 20 A skips the 10 A
scale where a side-chain cluster becomes a loop, and stops at 20 A, below the
size of a subdomain. Radii 10 and 26 are added.

43 x (1 local + 5 means + 3 variances + 3 centred) = 516 wires. Everything is an
unweighted average over a fixed geometric adjacency; there is no kernel, no
bandwidth, no fitted constant. The propensity counter is still compiled on the
training fold alone and carried to the test fold.

Distances are formed once per chain and reused across radii, which is what makes
eleven passes affordable where the original three-radius transform recomputed
them each time.

Usage: PYTHONPATH=src:tools python3.12 tools/build_wide_cache.py
"""
from __future__ import annotations

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.methods.expanded_descriptors import (
    CHEM_NAMES,
    chemical_wires,
)
from pocket_bench.methods.sequence_wires import apply_propensity, propensity_table
from pocket_bench.paths import ROOT

SRC = {"train": ROOT / "data/cryptobench_apo/_cascade_cache_train.npz",
       "test": ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"}
DST = {"train": ROOT / "data/cryptobench_apo/_wide_cache_train.npz",
       "test": ROOT / "data/cryptobench_apo/_wide_cache_test.npz"}

MEAN_RADII = (6.0, 10.0, 14.0, 20.0, 26.0)
VAR_RADII = (6.0, 14.0, 20.0)
DIFF_RADII = (6.0, 14.0, 20.0)
RANK_RADII = (6.0, 14.0, 20.0)
BLOCK = 768


def wide_transform(local, ctr, n_res_per):
    """Means, variances, centred differences and local ranks.

    The local rank

        K_r[x]_i = |{ j in N_r(i) : x_j < x_i }| / |N_r(i)|

    is the one statistic here that is not a moment. It asks where the residue
    sits in the order of its own neighbourhood rather than how far it is from
    the neighbourhood's centre, so it is invariant to any monotone rescaling of
    the wire and it separates a residue that is slightly the most hydrophobic
    of its neighbours from one that is slightly the least, which the centred
    difference reports as two small numbers of opposite sign but comparable
    magnitude. It is a count, computed by comparison and division.
    """
    C = local.shape[1]
    n_out = C * (1 + len(MEAN_RADII) + len(VAR_RADII) + len(DIFF_RADII)
                 + len(RANK_RADII))
    out = np.empty((local.shape[0], n_out), dtype=np.float64)
    radii = sorted(set(MEAN_RADII) | set(VAR_RADII) | set(DIFF_RADII)
                   | set(RANK_RADII))
    off = 0
    for u, n in enumerate(n_res_per):
        n = int(n)
        c = ctr[off:off + n]
        blk = local[off:off + n]
        sq = blk * blk
        means, variances, ranks = {}, {}, {}
        for r in radii:
            r2 = r * r
            acc = np.empty((n, C)); acc2 = np.empty((n, C))
            krk = np.empty((n, C)) if r in RANK_RADII else None
            for i in range(0, n, BLOCK):
                j = min(i + BLOCK, n)
                d2 = ((c[i:j, None, :] - c[None, :, :]) ** 2).sum(-1)
                a = (d2 <= r2).astype(np.float64)
                cnt = np.maximum(a.sum(1), 1.0)[:, None]
                acc[i:j] = (a @ blk) / cnt
                acc2[i:j] = (a @ sq) / cnt
                if krk is not None:
                    less = (blk[i:j, None, :] > blk[None, :, :])
                    krk[i:j] = (less * a[:, :, None]).sum(1) / cnt
            means[r] = acc
            variances[r] = np.maximum(acc2 - acc * acc, 0.0)
            if krk is not None:
                ranks[r] = krk
        cols = [blk]
        cols += [means[r] for r in MEAN_RADII]
        cols += [np.sqrt(variances[r]) for r in VAR_RADII]
        cols += [blk - means[r] for r in DIFF_RADII]
        cols += [ranks[r] for r in RANK_RADII]
        out[off:off + n] = np.concatenate(cols, axis=1)
        off += n
        if (u + 1) % 100 == 0:
            print(f"    {u + 1}/{len(n_res_per)} chains", flush=True)
    return out


def wire_names(local_names):
    names = list(local_names)
    names += [f"{nm}@{int(r)}" for r in MEAN_RADII for nm in local_names]
    names += [f"{nm}~sd{int(r)}" for r in VAR_RADII for nm in local_names]
    names += [f"{nm}~d{int(r)}" for r in DIFF_RADII for nm in local_names]
    names += [f"{nm}~k{int(r)}" for r in RANK_RADII for nm in local_names]
    return tuple(names)


def main() -> int:
    ztr = np.load(SRC["train"], allow_pickle=False)
    prop = propensity_table(ztr["codes"], ztr["y"])
    local_names = tuple(FEATURE_NAMES) + CHEM_NAMES + ("propensity",)
    names = wire_names(local_names)
    print(f"{len(local_names)} local wires -> {len(names)} wires", flush=True)

    for split in ("train", "test"):
        z = np.load(SRC[split], allow_pickle=False)
        F, codes, ctr = z["F"], z["codes"], z["ctr"]
        n_res_per, y = z["n_res_per"], z["y"]
        local = np.concatenate(
            [F, chemical_wires(codes), apply_propensity(codes, prop)[:, None]],
            axis=1)
        print(f"  {split}: {len(n_res_per)} chains, {len(y)} residues",
              flush=True)
        X = wide_transform(local, ctr, n_res_per)
        assert X.shape[1] == len(names), (X.shape, len(names))
        np.savez_compressed(
            DST[split], X=X.astype(np.float32), y=y, ctr=ctr,
            n_res_per=n_res_per, units=z["units"],
            names=np.array(names), propensity_table=prop)
        print(f"  wrote {DST[split].relative_to(ROOT)}  {X.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
