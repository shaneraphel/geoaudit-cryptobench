"""Five statistics of a neighbourhood, where there used to be one.

Why this module exists
----------------------
The 172-wire set read 43 local quantities under a single operation, the mean over
a ball of radius 6, 14 or 20 A. A table field over those wires reached ROC-AUC
0.7952 on the official test fold, statistically indistinguishable from a fitted
continuous linear functional of the very same wires (0.7953, paired difference
-0.0002 at p=0.98). That equality is the useful fact: the combinational path had
stopped losing anything to the continuous one, so no further work on tables or
fusion could help, and anything more had to enter as new wires.

Four operations are added to the mean. Each is an unweighted statistic of a
fixed geometric adjacency -- no kernel, no bandwidth, no fitted constant -- and
each says something the mean provably cannot.

**Second moment.** ``sd_r[x] = sqrt(C_r[x^2] - C_r[x]^2)``. Two residues can have
the same mean hydrophobicity around them, one because every neighbour is
moderately oily and one because half are oily and half are charged. A cryptic
site sits at the interface between such regions.

**Centred difference.** ``d_r[x] = x - C_r[x]``, a discrete Laplacian: large
where a residue is unlike its surroundings. As a real number this is a
difference of two wires already present and a linear functional gains nothing
from it. The field does not read real numbers: each wire is separately ranked
within its chain and cut into a quaternary digit, and the digit of a difference
is not a function of the digits of its terms.

**Local rank.** ``k_r[x]_i = |{j in N_r(i) : x_j < x_i}| / |N_r(i)|``. The only
statistic here that is not a moment. It asks where a residue sits in the order
of its neighbourhood rather than how far it is from the neighbourhood's centre,
so it is unchanged by any monotone rescaling of the wire, and it separates a
residue that is marginally the most hydrophobic of its neighbours from one that
is marginally the least -- which the centred difference reports as two small
numbers of opposite sign and comparable magnitude.

**More radii.** 6, 10, 14, 20, 26 A rather than 6, 14, 20. The original three
skip the ~10 A scale at which a side-chain cluster becomes a loop, and stop
below the size of a subdomain.

43 x (1 + 5 + 3 + 3 + 3) = 645 wires. On the official test fold this took the
field from 0.7952 to 0.8010 and past the ceiling of a continuous functional over
the same wires, so the tables are once again reading structure an additive
readout cannot express.

Cost
----
Distances are formed once per chain and shared by every radius. The original
three-radius transform rebuilt them for each radius, which is what made eleven
passes look unaffordable.
"""
from __future__ import annotations

import numpy as np

from pocket_bench.methods.expanded_descriptors import CHEM_NAMES, chemical_wires
from pocket_bench.methods.sequence_wires import apply_propensity

MEAN_RADII = (6.0, 10.0, 14.0, 20.0, 26.0)
VAR_RADII = (6.0, 14.0, 20.0)
DIFF_RADII = (6.0, 14.0, 20.0)
RANK_RADII = (6.0, 14.0, 20.0)
BLOCK = 768

N_STATISTIC_GROUPS = (1 + len(MEAN_RADII) + len(VAR_RADII) + len(DIFF_RADII)
                      + len(RANK_RADII))


def local_wires(F: np.ndarray, codes: np.ndarray,
                prop_table: np.ndarray) -> np.ndarray:
    """The 43 per-residue quantities the transform operates on.

    35 algebraic and topological invariants of atom positions, 7 published
    physicochemical constants of the residue type, and one residue frequency
    counted on the training fold. The propensity table is always supplied from
    the compiled artifact, so a test residue never contributes a count to it.
    """
    return np.concatenate(
        [np.asarray(F, dtype=np.float64), chemical_wires(codes),
         apply_propensity(codes, prop_table)[:, None]], axis=1)


def local_wire_names(base_names) -> tuple[str, ...]:
    return tuple(base_names) + CHEM_NAMES + ("propensity",)


def wire_names(local_names) -> tuple[str, ...]:
    names = list(local_names)
    names += [f"{nm}@{int(r)}" for r in MEAN_RADII for nm in local_names]
    names += [f"{nm}~sd{int(r)}" for r in VAR_RADII for nm in local_names]
    names += [f"{nm}~d{int(r)}" for r in DIFF_RADII for nm in local_names]
    names += [f"{nm}~k{int(r)}" for r in RANK_RADII for nm in local_names]
    return tuple(names)


def wide_transform(local: np.ndarray, ctr: np.ndarray, n_res_per) -> np.ndarray:
    """``(R, C * 15)``: every local wire under every statistic, blocked by chain.

    The adjacency is intra-chain by construction; a residue's neighbourhood
    cannot reach into a structure it is not part of, which is also what lets one
    chain be scored without reference to any other.
    """
    local = np.asarray(local, dtype=np.float64)
    ctr = np.asarray(ctr, dtype=np.float64)
    C = local.shape[1]
    out = np.empty((local.shape[0], C * N_STATISTIC_GROUPS), dtype=np.float64)
    radii = sorted(set(MEAN_RADII) | set(VAR_RADII) | set(DIFF_RADII)
                   | set(RANK_RADII))
    off = 0
    for n in n_res_per:
        n = int(n)
        c = ctr[off:off + n]
        blk = local[off:off + n]
        sq = blk * blk
        means, sds, ranks = {}, {}, {}
        for r in radii:
            r2 = r * r
            acc = np.empty((n, C))
            acc2 = np.empty((n, C))
            krk = np.empty((n, C)) if r in RANK_RADII else None
            for i in range(0, n, BLOCK):
                j = min(i + BLOCK, n)
                d2 = ((c[i:j, None, :] - c[None, :, :]) ** 2).sum(-1)
                a = (d2 <= r2).astype(np.float64)
                cnt = np.maximum(a.sum(1), 1.0)[:, None]
                acc[i:j] = (a @ blk) / cnt
                acc2[i:j] = (a @ sq) / cnt
                if krk is not None:
                    less = blk[i:j, None, :] > blk[None, :, :]
                    krk[i:j] = (less * a[:, :, None]).sum(1) / cnt
            means[r] = acc
            if r in VAR_RADII:
                sds[r] = np.sqrt(np.maximum(acc2 - acc * acc, 0.0))
            if krk is not None:
                ranks[r] = krk
        cols = [blk]
        cols += [means[r] for r in MEAN_RADII]
        cols += [sds[r] for r in VAR_RADII]
        cols += [blk - means[r] for r in DIFF_RADII]
        cols += [ranks[r] for r in RANK_RADII]
        out[off:off + n] = np.concatenate(cols, axis=1)
        off += n
    return out


def build_wide(F: np.ndarray, codes: np.ndarray, ctr: np.ndarray, n_res_per,
               base_names, prop_table: np.ndarray,
               ) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(X, names)`` for one or many chains, given a compiled propensity table."""
    local = local_wires(F, codes, prop_table)
    names = wire_names(local_wire_names(base_names))
    X = wide_transform(local, ctr, n_res_per)
    assert X.shape[1] == len(names), (X.shape, len(names))
    return X, names
