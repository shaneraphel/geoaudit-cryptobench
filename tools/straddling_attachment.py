#!/usr/bin/env python3.12
"""Tables that pair a deployed wire with a new column, which nothing here has built.

The measurement that asks for this
----------------------------------
COLLECTABILITY_SCREEN.json now carries the interaction of pairs that straddle the
deployed 645-wire bus and each appended family, and for every family with a
positive value it is larger than that family's interaction with itself:

    family                internal     cross with the bus     field lift
    asymmetry 129        -9.69e-06          -1.24e-06           +0.0010
    composition 76       +5.96e-06          +1.15e-05           -0.0009
    graph invariants 225 +6.42e-06          +1.23e-05           -0.0061
    graph invariants 15  +5.05e-06          +1.93e-05        not measured
    deployed bus         +1.06e-05                 --          (the bank)

Composition's synergy with the bus is nearly twice its synergy with itself and
sits above the deployed bank's own mean pairwise interaction -- the bank whose
tables support a counting field that beats a linear solve by +0.0053.

And no attachment in this repository forms one of those pairs. ``union`` is
``old + [[c + n_old for c in t] for t in new]``: the 5,152 deployed pairings held
exactly, plus tables over the new columns alone, and zero straddling. ``widened``
does form them, by redrawing every pairing at the new width, which keeps 281 of
the 5,152 old tables and throws the tuned bank away to reach them. So the
strongest collectible structure these families have is what both attachments
discard, one by construction and one by demolition.

What this tool does
-------------------
A third attachment. Keep all 5,152 deployed tables unchanged and add tables that
each pair one deployed wire with one new column, drawn as edge-disjoint rounds so
that no wire-column pair is used twice, at a table count matched to what ``union``
adds so the two differ in where the tables sit and not how many there are.

Arms, and why each is needed to read the result
-----------------------------------------------
``union``     the existing attachment, recomputed here rather than quoted, so
              that straddle-minus-union is a difference of two numbers produced
              by the same code on the same splits.
``straddle``  the new attachment.
``more_old``  the same number of extra tables, drawn over the deployed wires
              only, with the new family absent entirely. Without this arm a lift
              could be nothing but a larger bank: more tables mean more cells,
              more coverage of the existing wires, and a longer fan-out to
              decorrelate with. This is the arm that separates "the crossing
              carries something" from "more tables help".

The prediction, recorded before the run
---------------------------------------
Written into ``docs/AGENT_MEMORY.md`` 2d in the commit that measured the cross
term, and repeated in ``PREDICTION`` below so the artifact can be checked against
it mechanically rather than against a memory of what was expected.

* ``straddle`` beats ``union`` by more than the 0.0026 reseed floor, and
  composition's -0.0009 turns positive: the synergy is reachable and the
  attachment was the constraint.
* ``straddle`` does not beat ``union``: the cross term joins the internal one as
  a correlate. Two statistics will then have ordered the families and neither
  will have moved a number, and the honest conclusion is to stop screening
  families and say so.
* ``straddle`` beats ``union`` but not ``more_old``: the lift is bank size and
  not the crossing, which is a third outcome the first two would have hidden.

Nothing here reads the test fold or any external unit.
"""

from __future__ import annotations

import argparse
import json
import time
from math import comb
from pathlib import Path

import numpy as np

import digit_cache  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    addresses,
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.straddling_attachment.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
SCREEN = ROOT / "results/architecture_sweep/COLLECTABILITY_SCREEN.json"
OUT = ROOT / "results/architecture_sweep/STRADDLING_ATTACHMENT.json"

PREDICTION = {
    "recorded": "docs/AGENT_MEMORY.md 2d, in the commit that measured the cross "
                "term, before this tool existed",
    "reseed_floor": 0.0026,
    "supports_the_mechanism": "straddle minus union above the reseed floor, and "
                              "straddle minus narrow positive for composition 76",
    "ends_the_screening_programme": "straddle minus union at or below the reseed "
                                    "floor: two statistics will have ordered the "
                                    "families and neither moved a number",
    "third_outcome_the_control_exists_for": "straddle above union but not above "
                                            "more_old, which would make the lift "
                                            "bank size and not the crossing",
}


def load_family(name: str) -> tuple[np.ndarray, str]:
    """The same registry ``appended_family_lift.py`` uses, plus composition."""
    if name == "chemistry 42":
        from chemistry_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/chemistry_wires.py"
    if name in ("backbone 39", "backbone 132"):
        # 39 was the first thirteen quantities at three aggregations; the
        # expansion to forty-four leaves those columns bit-identical and appends
        # the rest, so BACKBONE_WIRES_LIFT.json remains a statement about a
        # subset of this family rather than about a different one.
        from backbone_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/backbone_wires.py"
    if name in ("backbone permuted 39", "backbone permuted 132"):
        # The family-specific control: identical columns, rows shuffled inside
        # each chain, so every marginal is the same multiset and only the
        # correspondence between a row and its residue is destroyed. Beating
        # more_old says the columns carry something; beating this says what they
        # carry is the backbone geometry of the residue being scored.
        from backbone_wires import build_or_load
        X, _n = build_or_load(permuted=True)
        return np.asarray(X), "tools/backbone_wires.py --permuted"
    if name == "sidechain 261":
        # 87 quantities at three aggregations. AGENT_MEMORY 2j-bis names this
        # family on 2i's rule -- a residue centroid does not determine chi1 --
        # and it is the second family that reads bytes the pipeline discards.
        from sidechain_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/sidechain_wires.py"
    if name == "sidechain permuted 261":
        # Same columns, rows shuffled inside each chain: every marginal is the
        # same multiset and only the correspondence between a row and its
        # residue is destroyed. Beating more_old says the columns carry
        # something; beating this says what they carry is the side-chain
        # conformation of the residue being scored.
        from sidechain_wires import build_or_load
        X, _n = build_or_load(permuted=True)
        return np.asarray(X), "tools/sidechain_wires.py --permuted"
    if name == "void 135":
        # 45 quantities at three aggregations. The third family to pass 2i's
        # screen and the first to pass it by reading something that is not a
        # function of atom positions at all: the connectivity of the empty
        # space between them. It is *not* built for the small-pocket tail, and
        # the module docstring records why that justification was withdrawn --
        # 2g measured the tail as noise-limited rather than dilution-limited
        # and 2h found a rival with no gate at all scoring 0.5985 there against
        # our 0.5958.
        from void_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/void_wires.py"
    if name == "void permuted 135":
        from void_wires import build_or_load
        X, _n = build_or_load(permuted=True)
        return np.asarray(X), "tools/void_wires.py --permuted"
    if name == "conformation 393":
        # Backbone 132 and side-chain 261 as one block. The two were each
        # measured against the same deployed baseline and each is worth about
        # +0.0045, and two lifts of that size measured separately are not
        # evidence that they add: they are both conformation, they are computed
        # from overlapping atom sets, and a residue in a strained rotamer is
        # often a residue in an irregular backbone. This arm is the only thing
        # that answers whether the sum survives, and it has to be measured
        # rather than added.
        import numpy as _np
        from backbone_wires import build_or_load as bb
        from sidechain_wires import build_or_load as sc
        A, _a = bb()
        B, _b = sc()
        return _np.concatenate([_np.asarray(A), _np.asarray(B)], axis=1), (
            "tools/backbone_wires.py + tools/sidechain_wires.py")
    if name == "displacement 144":
        # 48 quantities at three aggregations, and the fourth family on 2i's
        # screen. The first three all read coordinates; this one reads none of
        # them -- the temperature factor, the occupancy and the alternate-
        # location indicator, three fields of every ATOM record that pdb_io
        # never extracts or extracts only in order to discard. 770 of 770
        # training chains have a varying B-factor column and 332 carry an
        # alternate. The prediction, written before the run and recorded in the
        # module docstring and AGENT_MEMORY 2n: +0.001 to +0.003 raw, 30-50 %
        # overlap with geometry 528, because a B-factor is largely a function
        # of exposure and exposure is what the deployed wires already read.
        from displacement_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/displacement_wires.py"
    if name in ("displacement B 96", "displacement alt 48"):
        # The attribution arm, and it is run because a prediction failed rather
        # than because a number looked interesting. AGENT_MEMORY 2n predicted
        # +0.001 to +0.003 for the whole family on the argument that a B-factor
        # is largely a function of solvent exposure, which the deployed wires
        # already read; it measured +0.0070 on 12/12. The same section names the
        # falsification route in advance: if the lift exceeds +0.004 the place
        # to look is the alternate-conformer group, which is not a function of
        # exposure. Splitting the family at exactly that seam is what tests it.
        #
        # Quantities 0..31 are the B-factor groups A-D; 32..47 are the
        # alternate-conformer and occupancy groups E-F. The 144 columns are
        # laid out as aggregation-major, so a group is one slice per
        # aggregation and neither arm is a re-aggregation of the other.
        import numpy as _np
        from displacement_wires import build_or_load
        from pocket_bench.methods.displacement import N_COLUMNS as Q
        X, _n = build_or_load()
        lo, hi = (0, 32) if name == "displacement B 96" else (32, Q)
        take = [a * Q + q for a in range(3) for q in range(lo, hi)]
        return _np.asarray(X)[:, take], (
            f"tools/displacement_wires.py, quantities {lo}..{hi - 1}")
    if name == "displacement permuted 144":
        from displacement_wires import build_or_load
        X, _n = build_or_load(permuted=True)
        return np.asarray(X), "tools/displacement_wires.py --permuted"
    if name == "geometry 624":
        # The four live families, minus the part of the fourth that measured
        # null. 132 backbone + 261 side-chain + 135 void + 96 displacement-B.
        #
        # The 48 alternate-conformer and occupancy columns are left out because
        # they were measured and are not there: `displacement alt 48` scored
        # +0.00052 against its own `more_old` control at +0.00052 -- the same
        # number to five decimals, which is what cell budget looks like when the
        # separation is exact. Carrying them would spend 384 tables on nothing.
        #
        # It is also what makes this arm runnable. All 672 columns exceed the
        # 645 deployed wires, and the harness refuses that draw: a round pairs
        # each new column with a *distinct* wire, which needs the new family to
        # be the smaller side. Dropping the null sub-family is the right answer
        # to both constraints at once, and it is the attribution that says which
        # 48 to drop rather than the column budget.
        import numpy as _np
        from backbone_wires import build_or_load as bb
        from sidechain_wires import build_or_load as sc
        from void_wires import build_or_load as vd
        from displacement_wires import build_or_load as dp
        from pocket_bench.methods.displacement import N_COLUMNS as Q
        A, _a = bb()
        B, _b = sc()
        C, _c = vd()
        D, _d = dp()
        take = [a * Q + q for a in range(3) for q in range(32)]
        return _np.concatenate(
            [_np.asarray(A), _np.asarray(B), _np.asarray(C),
             _np.asarray(D)[:, take]],
            axis=1), ("tools/backbone_wires.py + tools/sidechain_wires.py + "
                      "tools/void_wires.py + tools/displacement_wires.py "
                      "quantities 0..31")
    if name == "geometry 528":
        # All three live families as one block: 132 backbone + 261 side-chain +
        # 135 void. Measured because the pairwise result does not settle the
        # triple. Backbone and side-chain recovered 78.8 % of their additive sum
        # over 12 splits, so they share about a fifth of what they carry; void
        # is the one family here that is not a function of atom positions in the
        # residue being scored, which is a reason to expect it to overlap less
        # and not a measurement that it does. If this arm lands near the sum of
        # the three it is the stack that ships; if it lands near conformation
        # 393 then void was buriedness arriving by a third route and the 1072
        # tables it costs buy nothing.
        import numpy as _np
        from backbone_wires import build_or_load as bb
        from sidechain_wires import build_or_load as sc
        from void_wires import build_or_load as vd
        A, _a = bb()
        B, _b = sc()
        C, _c = vd()
        return _np.concatenate(
            [_np.asarray(A), _np.asarray(B), _np.asarray(C)], axis=1), (
            "tools/backbone_wires.py + tools/sidechain_wires.py + "
            "tools/void_wires.py")
    if name == "composition 76":
        from composition_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/composition_wires.py"
    if name == "graph invariants 225":
        from graph_invariant_wires import build_wide_or_load
        X, _n = build_wide_or_load()
        return np.asarray(X), "tools/graph_invariant_wires.py"
    if name == "graph invariants 15":
        from graph_invariant_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X), "tools/graph_invariant_wires.py"
    if name == "asymmetry 129":
        from anisotropic_expansion_ceiling import build_or_load
        X, _d, _n = build_or_load(8)
        return np.asarray(X), "tools/anisotropic_expansion_ceiling.py"
    raise SystemExit(f"unknown family {name!r}")


def straddling_tables(n_old: int, n_new: int, n_tables: int, seed: int
                      ) -> list[list[int]]:
    """Edge-disjoint rounds of wire-to-column pairs, one column per round each.

    A round is a matching: every new column is paired with a distinct deployed
    wire, so no column is over-represented within a round and no wire is read
    twice in the same round. Rounds are accumulated until the requested table
    count is reached, and a pair already used is skipped rather than repeated,
    which is the same edge-disjointness ``partition_tables`` enforces within the
    bank. Columns of the new family are indexed from ``n_old``, matching the
    concatenation order used everywhere else here.
    """
    if n_new > n_old:
        raise SystemExit(
            f"{n_new} new columns against {n_old} deployed wires: a round can "
            f"pair each column with a distinct wire only while the new family "
            f"is the smaller side, and this draw assumes it")
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    out: list[list[int]] = []
    while len(out) < n_tables:
        wires = rng.permutation(n_old)[:n_new]
        cols = rng.permutation(n_new)
        progressed = False
        for w, c in zip(wires, cols):
            key = (int(w), int(c))
            if key in seen:
                continue
            seen.add(key)
            out.append([int(w), n_old + int(c)])
            progressed = True
            if len(out) == n_tables:
                break
        if not progressed:
            raise SystemExit(
                f"exhausted the {n_old * n_new} distinct straddling pairs at "
                f"{len(out)} tables")
    return out


def more_old_tables(n_old: int, n_tables: int, seed: int) -> list[list[int]]:
    """The control: extra tables over the deployed wires, no new column at all.

    Drawn by continuing ``partition_tables`` past the deployed 16 rounds with a
    different seed and taking pairs the bank does not already contain, so the
    added tables are the same kind of object as the bank's own and differ from
    the straddling ones only in what the second column is.
    """
    have = {tuple(sorted(t)) for t in
            partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS,
                             PARTITION_SEED)}
    out: list[list[int]] = []
    extra = 1
    while len(out) < n_tables:
        for t in partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS,
                                  PARTITION_SEED + 1000 * extra):
            k = tuple(sorted(t))
            if k in have:
                continue
            have.add(k)
            out.append(list(t))
            if len(out) == n_tables:
                break
        extra += 1
        if extra > 64:
            raise SystemExit(f"could not draw {n_tables} unused old pairings")
    return out


def digits_of(F: np.ndarray, n_res, deployed: np.ndarray) -> np.ndarray:
    """Concatenated digits, without ever concatenating the float columns.

    ``chain_digits`` ranks each column within each chain independently of every
    other column, so digitising two blocks separately and concatenating the int8
    results is the same array as digitising the concatenation -- 234,838 x 721
    int8 at 169 MB rather than a float64 concatenation at 1.35 GB. Same argument
    as the digit cache, and checked the same way below rather than assumed.
    """
    return np.concatenate([np.asarray(deployed), chain_digits(F, n_res)], axis=1)


def check_digit_concatenation(F: np.ndarray, n_res, deployed: np.ndarray,
                              n_chains: int = 24) -> dict:
    """Require the split digitisation to equal digitising the concatenation."""
    k = min(int(n_chains), len(n_res))
    n = int(np.sum(n_res[:k]))
    z = np.load(WIDE, allow_pickle=False)
    head = np.concatenate([np.asarray(z["X"][:n], dtype=np.float64),
                           np.asarray(F[:n], dtype=np.float64)], axis=1)
    z.close()
    want = chain_digits(head, n_res[:k])
    got = np.concatenate([np.asarray(deployed[:n]),
                          chain_digits(F[:n], n_res[:k])], axis=1)
    same = np.array_equal(want, got)
    if not same:
        raise SystemExit(
            f"digitising the two blocks separately does not reproduce "
            f"digitising the concatenation on the first {k} chains; the memory "
            f"saving is not free and the run is refused")
    return {"checked_on": f"the first {k} chains, {n} residues, "
                          f"{want.shape[1]} columns",
            "identical": bool(same)}


# Rows per pass through the fan-out's scatter accumulation. table_bank.BLOCK is
# 8,192, which at K = 5,760 tables makes each intermediate 377 MB and the
# centring step below allocate three of them at once.
LEAN_BLOCK = 2048


def lean_integer_fanout(D, y, tables, offsets, frac, ridge: float, cap: int
                        ) -> np.ndarray:
    """``integer_fanout`` with the scatter accumulated without three copies.

    Why a local copy of a deployed routine, which this repository normally
    forbids. ``table_bank.py`` is one of the eight files ``TABLE_FIELD.json``
    carries a ``code_sha256`` over, so editing it invalidates the compiled field
    for a reason unrelated to the field. The rule when that happens is to copy
    the function into the tool, say so, and say why -- this is that.

    What is different, and it is only the memory profile. The canonical
    ``scatter_and_means`` centres a block with
    ``np.where(p[:, None], v - mu1, v - mu0)``, which holds ``v - mu1``,
    ``v - mu0`` and the result simultaneously: at 8,192 rows and 5,760 tables
    that is three float64 arrays of 377 MB for one 377 MB answer, and with the
    scatter itself at 265 MB and the address block at another 377 MB the call
    peaks near 2.4 GB. Measured on this host, that is what the kernel kills.
    Here the block is centred in place, one array rather than three, over
    shorter passes.

    The arithmetic is the same arithmetic. Floating-point summation is not
    associative, so a different block length can move the scatter in the last
    bits, and the check below requires the rounded multiplicities -- which are
    integers in [-cap, cap] and are what actually reaches inference -- to come
    back identical at a table count where the canonical routine still fits.
    """
    K = len(tables)
    pos = np.asarray(y) == 1
    n = int(D.shape[0])
    n1 = int(pos.sum())
    n0 = int(n - n1)

    s1 = np.zeros(K)
    s0 = np.zeros(K)
    for a in range(0, n, LEAN_BLOCK):
        b = min(a + LEAN_BLOCK, n)
        v = frac[addresses(D, tables, offsets, a, b)]
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
        del v
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)

    S = np.zeros((K, K))
    for a in range(0, n, LEAN_BLOCK):
        b = min(a + LEAN_BLOCK, n)
        c = frac[addresses(D, tables, offsets, a, b)]
        p = pos[a:b]
        c[p] -= mu1
        c[~p] -= mu0
        S += c.T @ c
        del c
    S /= max(n - 2, 1)

    S.flat[::K + 1] += ridge * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(K, dtype=np.int64)
    return np.round(w / peak * cap).astype(np.int64)


def check_lean_fanout(D, y, tables, offsets, frac, ridge: float, cap: int,
                      n_tables: int = 512) -> dict:
    """Require the lean fan-out to return the canonical multiplicities exactly.

    Run at a table count small enough that the canonical routine fits here, on
    the same rows and the same compiled cells, because the claim being made is
    about the arithmetic and not about the size.
    """
    k = min(int(n_tables), len(tables))
    sub = tables[:k]
    offs = cell_offsets(sub)
    fr, _t = compile_cells(D, y, sub, offs)
    want = integer_fanout(D, y, sub, offs, fr, ridge, cap)
    got = lean_integer_fanout(D, y, sub, offs, fr, ridge, cap)
    same = np.array_equal(want, got)
    if not same:
        n_diff = int((want != got).sum())
        raise SystemExit(
            f"the lean fan-out returns {n_diff} of {k} multiplicities different "
            f"from integer_fanout on the same cells; the memory saving changes "
            f"the detector and the run is refused")
    return {
        "checked_on": f"{k} of {len(tables)} tables, {int(D.shape[0])} rows",
        "multiplicities_identical": bool(same),
        "why_only_a_subset": "the canonical routine peaks near 2.4 GB at the "
                             "full table count on this host, which is the "
                             "reason the lean one exists; the claim is about "
                             "the arithmetic and is testable where both run",
    }


def out_path(spec: str) -> Path:
    p = Path(spec)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="composition 76")
    ap.add_argument("--splits", type=int, default=0,
                    help="0 means every split the frozen baseline has")
    ap.add_argument("--arms", default="union,straddle,more_old")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    wanted = [s.strip() for s in a.arms.split(",") if s.strip()]

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}

    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    deployed = digit_cache.load(n_res)
    n_old = int(deployed.shape[1])
    if n_old not in frozen:
        raise SystemExit(f"the frozen artifact reports widths {sorted(frozen)}")
    n_splits = a.splits or int(cdoc["protocol"]["n_splits"])
    narrow = frozen[n_old][:n_splits]

    F, built_by = load_family(a.family)
    n_new = int(F.shape[1])

    t0 = time.perf_counter()
    concat_check = check_digit_concatenation(F, n_res, deployed)
    D = digits_of(F, n_res, deployed)
    print(f"digits {D.shape} {D.dtype} in {time.perf_counter() - t0:.0f}s; "
          f"separate digitisation matches the concatenated one on "
          f"{concat_check['checked_on']}", flush=True)

    old = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    new = partition_tables(n_new, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    n_added = len(new)
    union = old + [[c + n_old for c in t] for t in new]
    straddle = old + straddling_tables(n_old, n_new, n_added, PARTITION_SEED)
    more_old = old + more_old_tables(n_old, n_added, PARTITION_SEED)
    banks = {"union": union, "straddle": straddle, "more_old": more_old}
    banks = {k: (v, cell_offsets(v)) for k, v in banks.items() if k in wanted}

    n_str = sum(1 for t in banks["straddle"][0]
                if (t[0] < n_old) != (t[1] < n_old)) if "straddle" in banks else 0
    print(f"  {len(old)} deployed tables held by every arm; each arm adds "
          f"{n_added}", flush=True)
    print(f"  union: {n_added} over the new columns alone, 0 straddling", flush=True)
    print(f"  straddle: {n_str} straddling, 0 over the new columns alone",
          flush=True)
    print(f"  more_old: {n_added} more over the deployed wires, the family absent",
          flush=True)

    # Checkpoint each split as it completes. integer_fanout on 5,760 tables
    # holds a K x K scatter and a block of the same width, about 1.5 GB, and
    # this host has been killing processes at that size all evening: a
    # twelve-split run that loses everything on a kill in split eleven never
    # finishes. One line of JSON per split makes a kill cost one split, and a
    # relaunch picks up where it stopped. The checkpoint records the arm set and
    # the family so a resume cannot silently mix two configurations.
    ckpt = Path(str(out_path(a.out)) + ".splits.jsonl")
    done: dict[int, dict] = {}
    tag = {"family": a.family, "arms": wanted, "n_old": n_old,
           "n_added": n_added}
    if ckpt.is_file():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("tag") == tag:
                done[int(r["split"])] = r["auc"]
        if done:
            print(f"  resuming: {len(done)} of {n_splits} splits already in "
                  f"{ckpt.name}", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    fanout_check = None
    got: dict[str, list[float]] = {k: [] for k in wanted}
    for s in range(n_splits):
        if s in done:
            for arm in wanted:
                got[arm].append(float(done[s][arm]))
            print(f"  split {s + 1}/{n_splits}  (from checkpoint)  "
                  + "  ".join(f"{k} {got[k][-1]:.4f}" for k in wanted),
                  flush=True)
            continue
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        yfit, ypick, ctrpick = y[fit], y[pick], ctr[pick]
        t1 = time.perf_counter()
        here: dict[str, float] = {}
        for arm in wanted:
            tabs, offs = banks[arm]
            Dfit = D[fit]
            frac, _t = compile_cells(Dfit, yfit, tabs, offs)
            if fanout_check is None:
                fanout_check = check_lean_fanout(Dfit, yfit, tabs, offs, frac,
                                                 RIDGE, FAN_OUT_CAP)
                print(f"  lean fan-out returns the canonical multiplicities on "
                      f"{fanout_check['checked_on']}", flush=True)
            mult = lean_integer_fanout(Dfit, yfit, tabs, offs, frac, RIDGE,
                                       FAN_OUT_CAP)
            del Dfit
            Dpick = D[pick]
            sc = apply_gate(score(Dpick, tabs, offs, frac, mult),
                            ctrpick, n_pick)
            del Dpick, frac, mult
            here[arm] = float(per_unit_auc(sc, ypick, n_pick))
            got[arm].append(here[arm])
        with ckpt.open("a") as fh:
            fh.write(json.dumps({"tag": tag, "split": s, "auc": here}) + "\n")
        print(f"  split {s + 1}/{n_splits}  narrow {narrow[s]:.4f}  "
              + "  ".join(f"{k} {got[k][-1]:.4f}" for k in wanted)
              + f"  {time.perf_counter() - t1:.0f}s", flush=True)

    def summarise(v):
        v = np.asarray(v, dtype=float)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, base):
        """Paired difference with an interval, because a mean and a count of
        positive splits cannot say whether an effect crosses zero, and every
        number on this axis is small enough that it usually does."""
        d = np.asarray(v, dtype=float) - np.asarray(base, dtype=float)
        n = int(len(d))
        k = int((d > 0).sum())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        return {"mean": round(float(d.mean()), 6),
                "n_splits_positive": k,
                "n_splits": n,
                "paired_se": round(se, 6),
                "ci95": [round(float(d.mean() - 1.96 * se), 6),
                         round(float(d.mean() + 1.96 * se), 6)],
                "crosses_zero": bool(abs(d.mean()) < 1.96 * se),
                "sign_test_p_one_sided": round(float(p), 4)}

    minus_narrow = {k: compare(v, narrow) for k, v in got.items()}
    versus_union = ({k: compare(v, got["union"])
                     for k, v in got.items() if k != "union"}
                    if "union" in got else {})

    # A verdict is refused below the full split count. The first version of this
    # block printed "stop screening families and say so" from a two-split smoke
    # test, which is the exact failure AGENTS.md records: a monotone reading from
    # two splits had to be withdrawn at twelve, and the anisotropic radius sweep
    # reversed between two and twelve. A smoke test establishes that the code
    # runs. It does not establish which arm wins, and a tool that will state a
    # conclusion from one is not an instrument.
    full_splits = int(cdoc["protocol"]["n_splits"])
    verdict = None
    if n_splits < full_splits:
        verdict = {
            "withheld": True,
            "why": f"{n_splits} of {full_splits} splits. This is a smoke test; "
                   f"its ordering is not a finding and no verdict is computed "
                   f"from it",
            "straddle_minus_union_so_far": (
                versus_union.get("straddle", {}).get("mean")),
        }
    elif "straddle" in got and "union" in got:
        d = versus_union["straddle"]["mean"]
        floor = PREDICTION["reseed_floor"]
        beats_union = d > floor and not versus_union["straddle"]["crosses_zero"]
        beats_more_old = (
            None if "more_old" not in got
            else float(np.mean(np.asarray(got["straddle"], dtype=float)
                               - np.asarray(got["more_old"], dtype=float))) > 0)
        # The decisive comparison depends on what is being asked. For the family
        # this tool was written for, composition 76, the question was whether
        # straddling reaches an interaction the union attachment discards. For a
        # family measured here for the first time it is a different question --
        # does the family contribute anything at all -- and the arm that answers
        # it is more_old, which adds the same number of tables with the family
        # absent. The first version of this block printed the cross-term reading
        # for every family, so a null on a new family came back describing an
        # experiment it was not part of.
        union_vs_narrow = minus_narrow.get("union") or {}
        control_vs_narrow = minus_narrow.get("more_old") or {}
        family_adds_nothing = (
            "more_old" in got
            and union_vs_narrow.get("crosses_zero", True)
            and control_vs_narrow.get("mean", 0.0)
            >= union_vs_narrow.get("mean", 0.0))
        if family_adds_nothing:
            reading = (
                f"the family contributes nothing. Its union attachment moves "
                f"the detector by {union_vs_narrow.get('mean', 0.0):+.5f} with "
                f"an interval that crosses zero, and adding the same number of "
                f"tables over the deployed wires with the family absent "
                f"entirely moves it by {control_vs_narrow.get('mean', 0.0):+.5f}"
                f" -- as much or more. Whatever small positive there is, is bank "
                f"size and not these columns. The control arm is what makes this "
                f"unambiguous and a run without it would have reported a lift")
        elif not beats_union:
            reading = (
                "the cross term does not survive being acted on. It ordered the "
                "families and moving the attachment to use it changes nothing "
                "above the reseed floor, which makes it a correlate exactly as "
                "the internal statistic was. Two statistics have now ordered "
                "these families and neither has moved a number: stop screening "
                "families and say so")
        elif beats_more_old is False:
            reading = (
                "straddling beats the union attachment and does not beat the "
                "same number of extra tables drawn over the deployed wires "
                "alone. The lift is bank size, not the crossing, and the new "
                "columns are not carrying it. This is the outcome the control "
                "arm exists to catch and it would have been read as a success "
                "without it")
        else:
            reading = (
                "straddling beats both the union attachment and a bank grown by "
                "the same number of tables over the deployed wires. The synergy "
                "the cross term measured is reachable, and the attachment rather "
                "than the columns was the constraint")
        verdict = {
            "straddle_minus_union": versus_union["straddle"],
            "straddle_minus_the_deployed_detector": minus_narrow.get("straddle"),
            "beats_union_above_the_reseed_floor": bool(beats_union),
            "beats_the_more_old_control": beats_more_old,
            "prediction": PREDICTION,
            "the_reading": reading,
            "what_the_two_comparisons_say_together": (
                "straddle minus union is the mechanism question and straddle "
                "minus narrow is the one that decides whether anything ships. "
                "A positive first with a zero second means the attachment was "
                "costing something and the crossing recovers exactly that cost, "
                "which is a real effect and not a gain: the detector is no "
                "better than it was before the family was added at all"),
        }

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether tables that pair a deployed wire with a new column "
                    "collect the straddling interaction that both existing "
                    "attachments discard",
        "family": a.family,
        "built_by": built_by,
        "n_new_columns": n_new,
        "why_this_attachment": (
            "COLLECTABILITY_SCREEN.json measures the interaction of straddling "
            "pairs as larger than each family's interaction with itself, and "
            "above the deployed bank's own +1.06e-05 for composition. The union "
            "attachment forms no straddling tables and widening forms them only "
            "by redrawing every pairing, which keeps 281 of 5,152"),
        "held_fixed": {
            "n_old_wires": n_old,
            "old_bank": f"{PARTITION_ROUNDS} rounds at width {TABLE_WIDTH}, "
                        f"seed {PARTITION_SEED}, held identically by every arm",
            "gate": "as deployed",
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "banding": "within-chain rank quartiles, per column",
        },
        "bank": {
            "n_deployed_tables": len(old),
            "n_tables_added_by_each_arm": n_added,
            "n_straddling_in_the_straddle_arm": n_str,
            "why_matched": "the arms differ in where the added tables sit and "
                           "not in how many there are, so a difference between "
                           "them is not a difference in cell budget",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC, gate applied as deployed",
            "baseline_was_not_recomputed":
                "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json",
        },
        "checks": {
            "separate_digitisation_matches_the_concatenated_one": concat_check,
            "digit_cache": "tools/digit_cache.py, which requires equality with "
                           "chain_digits on the leading chains",
            "lean_fanout_matches_integer_fanout": fanout_check,
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "minus_narrow": minus_narrow,
        "versus_union": versus_union,
        "verdict": verdict,
        "per_split": {k: [round(x, 6) for x in v] for k, v in got.items()},
        "per_split_narrow_frozen": [round(float(x), 6) for x in narrow],
    }

    out = out_path(a.out)
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  narrow {narrow.mean():.4f}"
          + "".join(f"  {k} {np.mean(v):.4f}" for k, v in got.items()))
    for k, v in minus_narrow.items():
        print(f"    {k:10s} - narrow  {v['mean']:+.4f} on "
              f"{v['n_splits_positive']}/{v['n_splits']}")
    for k, v in versus_union.items():
        print(f"    {k:10s} - union   {v['mean']:+.4f} on "
              f"{v['n_splits_positive']}/{v['n_splits']}")
    if verdict is not None and verdict.get("withheld"):
        print(f"\n  no verdict: {verdict['why']}")
    elif verdict is not None:
        print(f"\n  {verdict['the_reading']}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
