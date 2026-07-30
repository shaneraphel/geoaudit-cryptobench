#!/usr/bin/env python3.12
"""Chemical composition of the neighbourhood, counted. Training folds only.

Why this axis and not another
-----------------------------
The deficit that matters is against pLM-NN: -0.0243 internally and -0.0340 on the
external set. pLM-NN reads ESM2-3B embeddings of the sequence, and
``sequence_wires.py`` says in its own docstring what this repository does not
read: "True evolutionary conservation (a PSSM / HHblits profile) is the strongest
known sequence signal for ligand-binding sites, and it is deliberately absent."
Four wires of the 645 carry sequence at all -- hydropathy, side-chain volume, a
neighbourhood hydropathy mean, and a training-fold propensity counter. Residue
identity enters as two scalars per residue and nothing else.

That is a gap in the input, not in the readout, and it is the one place where a
counting-only construction can meet a neural encoder on the encoder's own ground.
What a protein language model has that a geometric invariant does not is which
chemistry sits where. Counting which chemistry sits where is arithmetic.

No database is read. Every column below is a count over residues of the same
receptor, so nothing is fetched, no alignment runs, and the sequence-wire
docstring's claim of no profile database and no network is unaffected. This is
therefore not conservation and is not described as conservation: it is
composition, a strict subset of what a profile would carry, needing nothing
outside the structure being scored.

The three families
------------------
Eight chemical classes partition the twenty standard residues: aliphatic
(AVLIM), aromatic (FWY), polar (STNQ), positive (KRH), negative (DE), and
glycine, proline and cysteine alone, the last three because flexibility,
rigidity and covalent capability are properties no other residue shares. The
builder asserts the partition covers all twenty exactly once rather than trusting
the table to have been typed correctly.

``shell``  For each residue and each radius in 6, 10 and 14 A, how many residues
of each class have their centroid inside that ball. 8 x 3 = 24 columns. The
signature this is aimed at is a pocket lining of aromatic and aliphatic residues
with a charged rim.

``walk``  On the 8 A contact graph, the number of walks of length 2 and 3 from
the residue that end on each class. 8 x 2 = 16 columns. A walk count is an exact
integer -- an entry of a power of a symmetric 0/1 matrix -- and it reaches
chemistry two and three contacts away, which no ball around the residue
separates from chemistry that is adjacent.

``pair``  Among the residue's contacts at 8 A, how many unordered pairs of each
class combination occur. 36 columns. This states "an aromatic sits beside another
aromatic in my neighbourhood", which neither other family can: both are linear in
the neighbour multiset and a pair count is quadratic in it.

76 columns. Each is a non-negative integer before quantisation and each is
rank-quantised within its own chain by the same rule as every other wire, so no
constant crosses the fold boundary.

How they are attached, and why that is half the experiment
----------------------------------------------------------
UNION_BANK_COUNTING_FIELD.json established that attaching columns matters as much
as choosing them: widening the bus from 645 to 774 keeps only 296 of the 5,152
existing pairings, because ``partition_tables`` redraws every pairing at the new
width. Both attachments are measured here.

``widened``  ``partition_tables(645 + 76, ...)``. Every pairing redrawn.
``union``    the 645-wire pairings held exactly, plus tables over the 76 new
             columns alone.

Read the union-minus-widened difference against SELECTED_PAIRINGS.json before
attributing it to pairing survival. That artifact measured a single reseed of the
random construction at -0.0026, which is larger than the union-versus-widened
difference the earlier commit credited to pairings, so a difference of that size
between two banks cannot be separated from seed variation.

The Fisher arm, and the mistake it nearly caused
------------------------------------------------
A ridge Fisher solve over the same columns under the deployed gate runs first and
is cheap. It separates "the information is absent" from "the construction does
not collect it", which need different next steps. It is a correlate and not a
bound: IS_FISHER_A_CEILING.json measured the counting field above the solve by
+0.0053 on the same wires.

``fisher`` is imported from ``expand_invariant_bank`` rather than written here.
A local version was written first and standardised nothing, which cost the
645-wire arm 0.03 against the frozen one and reported these columns at -0.0096 on
0 of 12 splits. With the canonical solve, which standardises every column on the
fit half, the same columns give +0.0010. The negative was entirely an artefact of
the reimplementation and would have closed a live axis. The reproduction check
below exists so that cannot happen silently again.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/composition_wires.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED, fisher  # noqa: E402
from quantisation_ladder import cell_occupancy  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.sequence_wires import AA20
from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.composition_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CODES = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
FISHER_REFERENCE = ROOT / "results/architecture_sweep/IS_FISHER_A_CEILING.json"
CACHE = ROOT / "data/cryptobench_apo/_composition_cache_train.npz"
OUT = ROOT / "results/architecture_sweep/COMPOSITION_WIRES.json"

CLASSES: dict[str, tuple[str, ...]] = {
    "aliphatic": ("ALA", "VAL", "LEU", "ILE", "MET"),
    "aromatic": ("PHE", "TRP", "TYR"),
    "polar": ("SER", "THR", "ASN", "GLN"),
    "positive": ("LYS", "ARG", "HIS"),
    "negative": ("ASP", "GLU"),
    "glycine": ("GLY",),
    "proline": ("PRO",),
    "cysteine": ("CYS",),
}
SHELL_RADII = (6.0, 10.0, 14.0)
CONTACT_RADIUS = 8.0
WALK_LENGTHS = (2, 3)


def class_of_code() -> np.ndarray:
    """Map each residue-code index to its class index, checking the partition."""
    seen: dict[str, int] = {}
    for ci, members in enumerate(CLASSES.values()):
        for m in members:
            if m in seen:
                raise SystemExit(f"{m} appears in two classes")
            seen[m] = ci
    missing = [a for a in AA20 if a not in seen]
    if missing:
        raise SystemExit(f"classes do not cover {missing}")
    if len(seen) != len(AA20):
        raise SystemExit(f"classes name {len(seen)} residues, expected "
                         f"{len(AA20)}")
    return np.asarray([seen[a] for a in AA20], dtype=np.int64)


def column_names() -> tuple[str, ...]:
    cls = list(CLASSES)
    names = [f"shell{int(r)}~{c}" for r in SHELL_RADII for c in cls]
    names += [f"walk{k}~{c}" for k in WALK_LENGTHS for c in cls]
    names += [f"pair~{a}+{b}" for i, a in enumerate(cls) for b in cls[i:]]
    return tuple(names)


def build_chain(ctr: np.ndarray, cls: np.ndarray, n_classes: int) -> np.ndarray:
    """All three families for one chain, as integer counts."""
    n = len(cls)
    ind = np.zeros((n, n_classes), dtype=np.float64)
    ind[np.arange(n), cls] = 1.0
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)

    out = [(d <= r).astype(np.float64) @ ind for r in SHELL_RADII]

    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    powers = {1: adj}
    for k in range(2, max(WALK_LENGTHS) + 1):
        powers[k] = powers[k - 1] @ adj
    for k in WALK_LENGTHS:
        p = powers[k].copy()
        np.fill_diagonal(p, 0.0)
        out.append(p @ ind)

    # Unordered class pairs among the contacts of each residue. With v the
    # neighbour class counts, the number of pairs of classes (a, b) is v_a * v_b
    # for a < b and v_a choose 2 for a == b.
    v = adj @ ind
    pair = np.empty((n, n_classes * (n_classes + 1) // 2), dtype=np.float64)
    col = 0
    for a in range(n_classes):
        for b in range(a, n_classes):
            pair[:, col] = (v[:, a] * (v[:, a] - 1.0) / 2.0 if a == b
                            else v[:, a] * v[:, b])
            col += 1
    out.append(pair)
    return np.concatenate(out, axis=1)


def build_or_load(force: bool = False) -> tuple[np.ndarray, tuple[str, ...]]:
    names = column_names()
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=False)
        cached = tuple(str(s) for s in z["names"])
        if cached == names:
            return z["C"], names
        print(f"cache holds {len(cached)} columns and this build wants "
              f"{len(names)}; rebuilding", flush=True)

    w = np.load(WIDE, allow_pickle=False)
    e = np.load(CODES, allow_pickle=False)
    for key in ("units", "n_res_per", "y"):
        if not np.array_equal(w[key], e[key]):
            raise SystemExit(
                f"the wide cache and the expanded cache disagree about {key}; "
                f"their rows cannot be assumed to be the same residues")
    if not np.array_equal(w["ctr"], e["ctr"]):
        raise SystemExit("the two caches disagree about residue centroids")
    codes = e["codes"]
    if int(codes.min()) < 0:
        raise SystemExit(
            f"{int((codes < 0).sum())} residues have no standard residue code; "
            f"a composition count over them would silently drop them")

    cls_all = class_of_code()[codes]
    ctr, n_res = w["ctr"], w["n_res_per"]
    n_classes = len(CLASSES)
    t0 = time.perf_counter()
    blocks = []
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], cls_all[off:off + n],
                                  n_classes))
        off += n
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(n_res)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    C = np.concatenate(blocks, axis=0)
    if C.shape != (len(codes), len(names)):
        raise SystemExit(f"built {C.shape}, expected "
                         f"{(len(codes), len(names))}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, C=C.astype(np.float32),
                        names=np.asarray(names))
    print(f"built {C.shape[1]} composition columns in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return C.astype(np.float32), names


def union_tables(n_old: int, n_new: int):
    """The old bank untouched, plus a bank over the new columns alone."""
    old = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    new = partition_tables(n_new, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    shifted = [[c + n_old for c in t] for t in new]
    replaced = partition_tables(n_old + n_new, TABLE_WIDTH, PARTITION_ROUNDS,
                                PARTITION_SEED)
    kept = sum(1 for t in old if sorted(t) in [sorted(r) for r in replaced])
    return old + shifted, replaced, {
        "n_tables_over_the_old_wires": len(old),
        "n_tables_over_the_new_columns": len(shifted),
        "n_tables_union": len(old) + len(shifted),
        "n_tables_widened": len(replaced),
        "n_old_pairings_that_survive_widening": kept,
        "read_this_against": "results/architecture_sweep/SELECTED_PAIRINGS.json,"
                             " which measured one reseed of the random "
                             "construction at -0.0026; a difference of that "
                             "size between two banks cannot be attributed to "
                             "pairing survival",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--arms", type=str, default="fisher,widened,union")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]
    wanted = [s.strip() for s in a.arms.split(",") if s.strip()]

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    n_old = int(W.shape[1])
    if n_old not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}, not "
                         f"{n_old}")

    C, names = build_or_load(a.rebuild)
    n_new = int(C.shape[1])
    tables_u, tables_w, bank = union_tables(n_old, n_new)
    print(f"{n_new} composition columns; union bank {bank['n_tables_union']} "
          f"tables, widened {bank['n_tables_widened']}, of which "
          f"{bank['n_old_pairings_that_survive_widening']} old pairings survive "
          f"the widening", flush=True)

    t0 = time.perf_counter()
    full = np.asarray(np.concatenate([W, C], axis=1), dtype=np.float64)
    D = chain_digits(full, n_res)
    print(f"banded {n_old + n_new} columns in {time.perf_counter() - t0:.0f}s",
          flush=True)
    off_u, off_w = cell_offsets(tables_u), cell_offsets(tables_w)
    row = np.repeat(np.arange(len(n_res)), n_res)

    got: dict[str, list[float]] = {k: [] for k in wanted}
    fis_narrow: list[float] = []
    occupancy: dict[str, dict] = {}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        print(f"  split {s + 1}/{n_splits}  deployed(frozen) "
              f"{frozen[n_old][s]:.4f}", flush=True)

        if "fisher" in wanted:
            t1 = time.perf_counter()
            sc = apply_gate(fisher(full[fit], y[fit], full[pick]), ctr[pick],
                            n_pick)
            got["fisher"].append(float(per_unit_auc(sc, y[pick], n_pick)))
            sc = apply_gate(fisher(full[fit][:, :n_old], y[fit],
                                   full[pick][:, :n_old]), ctr[pick], n_pick)
            fis_narrow.append(float(per_unit_auc(sc, y[pick], n_pick)))
            print(f"      fisher {n_old}+{n_new} {got['fisher'][-1]:.4f}  "
                  f"fisher {n_old} {fis_narrow[-1]:.4f}  "
                  f"lift {got['fisher'][-1] - fis_narrow[-1]:+.4f}  "
                  f"{time.perf_counter() - t1:.0f}s", flush=True)

        for name, tb, off in (("widened", tables_w, off_w),
                              ("union", tables_u, off_u)):
            if name not in wanted:
                continue
            t1 = time.perf_counter()
            frac, tot = compile_cells(D[fit], y[fit], tb, off)
            mult = integer_fanout(D[fit], y[fit], tb, off, frac, RIDGE,
                                  FAN_OUT_CAP)
            sc = apply_gate(score(D[pick], tb, off, frac, mult), ctr[pick],
                            n_pick)
            got[name].append(float(per_unit_auc(sc, y[pick], n_pick)))
            if s == 0:
                occupancy[name] = cell_occupancy(tot)
            print(f"      {name:8s} {got[name][-1]:.4f}  "
                  f"{got[name][-1] - frozen[n_old][s]:+.4f}  "
                  f"{time.perf_counter() - t1:.0f}s", flush=True)

    base = frozen[n_old][:n_splits]

    def summarise(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref):
        d = np.asarray(v) - np.asarray(ref)
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    counting_arms = [k for k in ("widened", "union") if got.get(k)]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "pLM-NN leads by 0.0243 and reads a language model of the "
                    "sequence; four of this detector's 645 wires carry residue "
                    "identity at all. Does counting which chemistry sits in the "
                    "neighbourhood close any of that, and does it matter whether "
                    "the columns are attached by widening the bus or by "
                    "extending the bank",
        "what_this_is_not": "not conservation. A profile states how a position "
                            "varies across a family and needs a database; these "
                            "columns state what is nearby in this one structure "
                            "and need nothing. The sequence-wire docstring's "
                            "claim of no profile database and no network is "
                            "unaffected",
        "columns": {
            "n": n_new,
            "classes": {k: list(v) for k, v in CLASSES.items()},
            "shell_radii_angstrom": [float(r) for r in SHELL_RADII],
            "contact_radius_angstrom": CONTACT_RADIUS,
            "walk_lengths": list(WALK_LENGTHS),
            "families": {
                "shell": "residues of each class with centroid inside a ball",
                "walk": "walks of the given length on the 8 A contact graph "
                        "ending on each class; an exact entry of a power of a "
                        "0/1 matrix",
                "pair": "unordered class pairs among the residue's contacts; "
                        "quadratic in the neighbour multiset, which neither "
                        "other family can express",
            },
            "every_value_is_a_count_before_quantisation": True,
            "names": list(names),
        },
        "bank": bank,
        "held_fixed": {
            "n_old_wires": n_old,
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "quantisation": "within-chain quartiles, the deployed ladder",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "arms": {k: summarise(v) for k, v in got.items() if v},
        "deployed_arm_frozen": summarise(base),
        # The narrow Fisher arm is stored per split as well as summarised. The
        # lift is a paired difference taken inside one code path, so a
        # systematic offset against the frozen reference cancels from it to
        # first order -- but that is an argument, and storing both arms makes it
        # a checkable one instead.
        "fisher_narrow_per_split": [round(float(x), 6) for x in fis_narrow]
        if fis_narrow else None,
        "minus_deployed": {k: compare(got[k], base) for k in counting_arms},
        "cell_occupancy_on_split_1": occupancy,
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in got.items() if v},
        "per_split_deployed_frozen": [round(float(x), 6) for x in base],
        "n_units": int(len(n_res)),
        "n_residues": int(len(y)),
        "n_positive_residues": int(y.sum()),
    }
    if len(counting_arms) == 2:
        doc["union_minus_widened"] = compare(got["union"], got["widened"])

    if fis_narrow:
        ref = None
        if FISHER_REFERENCE.exists():
            rdoc = json.loads(FISHER_REFERENCE.read_text())
            key = f"fisher {n_old}"
            if key in rdoc.get("per_split", {}):
                rv = np.asarray(rdoc["per_split"][key][:n_splits], dtype=float)
                gap = float(np.abs(np.asarray(fis_narrow) - rv).max())
                ref = {
                    "source": str(FISHER_REFERENCE.relative_to(ROOT)),
                    "max_absolute_difference": round(gap, 9),
                    "reproduces_the_frozen_arm": bool(gap < 1e-4),
                    "tolerance": 1e-4,
                    "mean_offset": round(
                        float(np.mean(fis_narrow) - rv.mean()), 9),
                    "what_to_do_when_it_fails": (
                        "the gap is a systematic offset in one direction plus "
                        "per-split scatter, and the reported lift is a paired "
                        "difference between two arms computed in this same code "
                        "path, so the offset cancels from it to first order. "
                        "That is checkable rather than assertable: "
                        "fisher_narrow_per_split holds the 645-wire arm split "
                        "by split beside per_split.fisher, so a reader can form "
                        "the difference themselves and compare its scatter to "
                        "the gap. Do not quote the absolute 645-wire value from "
                        "this artifact; quote it from the reference"),
                    "why_the_tolerance_is_not_machine_epsilon": (
                        "the reference solves over the float32 cache directly "
                        "and this tool concatenates it with the composition "
                        "columns into float64 first, so column means and "
                        "standard deviations accumulate differently. The "
                        "residual is order 1e-5 in ROC-AUC, four orders below "
                        "the effect being measured, and it is recorded rather "
                        "than rounded away"),
                    "why_this_is_here": (
                        "the first version of this tool wrote its own solve, "
                        "did not standardise the columns, scored the 645-wire "
                        "arm 0.03 below the frozen one, and reported a negative "
                        "lift that was an artefact of the solve rather than of "
                        "the columns"),
                }
        doc["fisher_lift_from_composition"] = {
            **compare(got["fisher"], fis_narrow),
            "fisher_at_645_mean": round(float(np.mean(fis_narrow)), 6),
            "reproduction_check": ref,
            "why_it_is_a_correlate_and_not_a_bound":
                "IS_FISHER_A_CEILING.json measured the counting field above "
                "this solve by +0.0053 on the same wires, so a null here does "
                "not close the axis and a lift here does not promise one",
        }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  deployed (frozen): {base.mean():.6f}")
    for k in counting_arms:
        c = doc["minus_deployed"][k]
        print(f"  {k:8s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']} splits")
    if "union_minus_widened" in doc:
        u = doc["union_minus_widened"]
        print(f"  union - widened: {u['mean']:+.6f} on "
              f"{u['n_splits_positive']}/{u['n_splits']}")
    if fis_narrow:
        f = doc["fisher_lift_from_composition"]
        print(f"  fisher lift from the columns: {f['mean']:+.6f} on "
              f"{f['n_splits_positive']}/{f['n_splits']}")
        if f["reproduction_check"]:
            r = f["reproduction_check"]
            print(f"  fisher 645 reproduces the frozen arm: "
                  f"{r['reproduces_the_frozen_arm']} "
                  f"(max |diff| {r['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
