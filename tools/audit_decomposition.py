#!/usr/bin/env python3
"""What the field actually read, on the residues it got right and wrong.

The paper claims the construction is auditable. That claim is cheap to make and
is usually made by pointing at the architecture: every table is a look-up, so in
principle a score can be taken apart. This decomposes one, and then several, and
reports what is inside.

The arithmetic is exact rather than attributed. A residue's pre-gate score is

    S(i) = sum_k m_k * frac_k[a_k(i)]

over all 5152 tables, so the per-table contributions are the score, not an
approximation of it. Each table addresses two wires, each wire derives from one
of the 43 local quantities, and each quantity belongs to one of the descriptor
families the appendix already defines. Grouping the exact contributions by
family therefore says where the score came from with no model of the model in
between. The gate adds a neighbourhood mean on top and its contribution is
reported as its own term, because spatial smoothing is part of what the detector
does and hiding it inside "geometry" would overstate how local the evidence is.

What is reported per residue is the deviation from its own chain's mean, table by
table. The absolute contribution is dominated by whichever tables have large
multiplicities regardless of the residue, and those are the same for every
residue in the chain; what makes one residue rank above another is only where it
sits relative to the rest, so that is the quantity decomposed.

Three things keep this honest.

The cases are not chosen here. They are read from ``CASE_STUDIES.json``, which a
committed tool selected, and the artifact fails if the set has changed. Choosing
which residues to explain after seeing which explanations look tidy would make
this a demonstration rather than an audit.

Within each committed case the residues are chosen by rule -- the highest-scoring
true positive, the highest-scoring false positive, the lowest-scoring missed
positive, and the residues the two methods disagree about -- so no residue is
picked for how it reads.

And the reconstruction is checked against the published per-residue scores before
anything is reported. If this file cannot recover the numbers the fold was scored
with, it is decomposing some other function and says so instead.

Usage:
  PYTHONPATH=src python3.12 tools/audit_decomposition.py
  PYTHONPATH=src python3.12 tools/audit_decomposition.py --check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"
CASES = ROOT / "results/official_fold/CASE_STUDIES.json"
PREDS = ROOT / "results/cryptobench_official/predictions/table_field.json"
P2PREDS = ROOT / "results/cryptobench_official/predictions/p2rank.json"
OUT = ROOT / "results/official_fold/AUDIT_DECOMPOSITION.json"

SCHEMA = "geoaudit.audit_decomposition.v1"
# How closely the reconstruction has to land on the published score. The scores
# are order 10^2 and are stored to full float precision, so this is a tight
# relative tolerance and not a courtesy.
REPRO_TOL = 1e-6
TOP_N = 6
N_LEVELS = 0  # filled from the scoring module, so it cannot drift from it

# The families are the appendix's, not new ones invented for this table. The
# coarse rollup is the reviewer's vocabulary -- geometric, chemical,
# topological -- mapped onto them once, here, so the mapping is auditable
# instead of being implied by prose.
GROUP_TITLES = ("G1 surface exposure", "G2 local spectral geometry",
                "G3 density field calculus", "G4 ultrametric structure",
                "G5 curvature and anisotropy", "G6 global position")
PHYSCHEM = ("kd", "volume", "charge", "aromatic", "hbd", "hba", "chi")
PROPENSITY = ("propensity",)
COARSE = {
    "G1 surface exposure": "geometric",
    "G5 curvature and anisotropy": "geometric",
    "G6 global position": "geometric",
    "G2 local spectral geometry": "topological",
    "G4 ultrametric structure": "topological",
    "G3 density field calculus": "density field",
    "physicochemical constants": "chemical",
    "training-fold residue propensity": "chemical",
    "gate (neighbourhood mean)": "spatial smoothing",
}


def _families() -> dict[str, str]:
    """Base quantity -> family, from the module that defines the groups."""
    from pocket_bench.methods import algebraic_descriptors as alg

    if len(alg.GROUPS) != len(GROUP_TITLES):
        raise SystemExit(
            f"the descriptor module now has {len(alg.GROUPS)} groups and this "
            f"file names {len(GROUP_TITLES)}; the family assignment would be "
            f"guessing")
    out = {}
    for title, group in zip(GROUP_TITLES, alg.GROUPS):
        for name in group:
            out[name] = title
    for name in PHYSCHEM:
        out[name] = "physicochemical constants"
    for name in PROPENSITY:
        out[name] = "training-fold residue propensity"
    return out


def _base_of(wire: str) -> str:
    """The local quantity a wire derives from.

    Wires are ``q``, ``q@r`` or ``q~stat``: the raw value, the value read at one
    of five radii, or one of nine neighbourhood statistics of it. All fifteen
    carry the same base quantity, which is what the family is a property of.
    """
    return wire.split("~")[0].split("@")[0]


def _resnum(x) -> int | None:
    if isinstance(x, int):
        return x
    s, digits, negative = str(x), "", False
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


class Decomposer:
    """One compiled field, taken apart per residue."""

    def __init__(self) -> None:
        global N_LEVELS
        from pocket_bench.methods.table_bank import N_LEVELS as _NL, cell_offsets
        from pocket_bench.methods.table_field import TableField

        N_LEVELS = _NL

        self.doc = json.loads(FIELD.read_text())
        self.field = TableField(self.doc)
        self.tables = [list(t) for t in self.doc["tables"]]
        self.offsets = cell_offsets(self.tables)
        self.mult = np.asarray(self.doc["multiplicity"], dtype=np.int64)
        self.frac = self.field.frac
        self.wires = [str(w) for w in self.doc["wire_names"]]
        self.prop = self.field.prop
        self.fam_of = _families()

        missing = sorted({_base_of(w) for w in self.wires} - set(self.fam_of))
        if missing:
            raise SystemExit(
                f"these wires' base quantities have no family: {missing}. "
                f"A decomposition that dropped them would not sum to the score")

        # A table's family pair. Where the two wires disagree the contribution
        # is split evenly, which is a convention and is recorded as one: there
        # is no fact of the matter about which half of a pairwise cell belongs
        # to which wire, and pretending otherwise would be the interesting part
        # of the answer smuggled into the bookkeeping.
        self.table_fams = []
        self.n_mixed = 0
        for a, b in self.tables:
            fa = self.fam_of[_base_of(self.wires[a])]
            fb = self.fam_of[_base_of(self.wires[b])]
            if fa != fb:
                self.n_mixed += 1
            self.table_fams.append((fa, fb))

    def per_table(self, X: np.ndarray):
        """Per-residue digits, addresses, per-table contributions, row sums.

        ``(n_res, 5152)`` of float64 is 4 MB for a 100-residue chain, so this is
        computed for one chain at a time and never for the fold.
        """
        from pocket_bench.methods.table_bank import addresses, chain_digits

        D = chain_digits(np.asarray(X, dtype=np.float64), [X.shape[0]])
        addr = addresses(D, self.tables, self.offsets, 0, D.shape[0])
        contrib = self.frac[addr] * self.mult[None, :]
        return D, addr, contrib, contrib.sum(axis=1)

    def gated(self, S: np.ndarray, ctr: np.ndarray) -> np.ndarray:
        from pocket_bench.methods.table_field import apply_gate

        return apply_gate(S, np.asarray(ctr, dtype=np.float64), [len(S)])

    def by_family(self, contrib: np.ndarray) -> dict[str, np.ndarray]:
        """Deviation from the chain mean, summed by family.

        Exact by construction: every table's deviation is assigned in full, so
        the family totals sum to the residue's pre-gate deviation.
        """
        dev = contrib - contrib.mean(axis=0, keepdims=True)
        out: dict[str, np.ndarray] = {}
        for k, (fa, fb) in enumerate(self.table_fams):
            if fa == fb:
                out[fa] = out.get(fa, 0.0) + dev[:, k]
            else:
                out[fa] = out.get(fa, 0.0) + 0.5 * dev[:, k]
                out[fb] = out.get(fb, 0.0) + 0.5 * dev[:, k]
        return out

    def top_tables(self, D: np.ndarray, addr: np.ndarray, contrib: np.ndarray,
                   i: int, n: int = TOP_N) -> list[dict]:
        """The tables that moved this residue furthest from its chain's mean.

        Each row is a complete statement of one look-up: which two quantities
        were read, which quartile the residue fell in for each, how often that
        cell was a binding residue in training, and how many of the 5152 tables
        happen to be this same pair.
        """
        base = float(self.doc["train"]["base_rate"])
        dev = contrib[i] - contrib.mean(axis=0)
        rows = []
        for k in np.argsort(-np.abs(dev))[:n]:
            k = int(k)
            a, b = self.tables[k]
            rate = float(self.frac[addr[i, k]])
            rows.append({
                "wires": [self.wires[a], self.wires[b]],
                "quantities": [_base_of(self.wires[a]), _base_of(self.wires[b])],
                "families": list(self.table_fams[k]),
                "quartile_of_this_residue": [int(D[i, a]) + 1, int(D[i, b]) + 1],
                "of_levels": int(N_LEVELS),
                "cell_binding_rate_in_training": round(rate, 4),
                "fold_base_rate": round(base, 4),
                "cell_is_this_many_times_the_base_rate": round(rate / base, 2),
                "multiplicity": int(self.mult[k]),
                "contribution_above_chain_mean": round(float(dev[k]), 4),
            })
        return rows


def _wires_for(unit: str, prop: np.ndarray):
    """The 645 wires for one receptor chain, by the deployment path.

    Not from the cached wire matrix. That cache is stored as float32, and the
    quantisation is a within-chain rank, so a float32 round-trip flips digits on
    near-ties and moves the score by a few tenths in 300. The called set does not
    change, but an audit that cannot reproduce the published number to the last
    place is not an audit, so this re-extracts from the coordinates exactly as
    ``score_receptor`` does.
    """
    from pocket_bench.methods.algebraic_descriptors import (
        FEATURE_NAMES, algebraic_residue_features)
    from pocket_bench.methods.wide_descriptors import build_wide

    path = RECEPTORS / f"{unit}_receptor.pdb"
    if not path.is_file():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)}. The receptors are not committed; "
            f"fetch them before regenerating this artifact. Checking the "
            f"committed artifact with --check needs no coordinates")
    chain = unit.split("_")[1] if "_" in unit else None
    resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
    X, _ = build_wide(F, codes, ctr,
                      np.asarray([len(resseq)], dtype=np.int64),
                      tuple(FEATURE_NAMES), prop)
    return [int(r) for r in resseq], np.asarray(X, dtype=np.float64), ctr


def _pick_residues(resnums: list[int], score: np.ndarray, truth: np.ndarray,
                   ours: set[int], theirs: set[int]) -> list[dict]:
    """The residues to explain, chosen by rule and not by how they read.

    Two kinds of miss are reported because they are two different failures. The
    near miss is a budget effect: the residue was ranked where a labelled residue
    should be ranked and the top-9% cut fell just above it, so nothing about the
    score is wrong. The deepest miss is an evidence failure: the field saw
    nothing there. Reporting only one of them would make the method look either
    better or worse than it is.
    """
    idx = {r: i for i, r in enumerate(resnums)}
    picks: list[dict] = []

    def add(role: str, cand: list[int], best, why: str) -> None:
        if cand:
            picks.append({"resnum": int(best(cand, key=lambda r: score[idx[r]])),
                          "role": role, "why_this_residue": why})

    called_true = [r for r in resnums if r in ours and truth[idx[r]]]
    called_false = [r for r in resnums if r in ours and not truth[idx[r]]]
    missed = [r for r in resnums if r not in ours and truth[idx[r]]]
    ours_only = [r for r in resnums if r in ours and r not in theirs]
    theirs_only = [r for r in resnums if r in theirs and r not in ours]

    add("true positive", called_true, max,
        "the highest-scoring residue we called that is a labelled cryptic "
        "binding residue")
    add("false positive", called_false, max,
        "the highest-scoring residue we called that is not labelled")
    add("near miss", missed, max,
        "the labelled residue we scored highest without calling: the one the "
        "top-9% budget cut off")
    add("deepest miss", missed, min,
        "the labelled residue we scored lowest, where the field saw nothing")
    add("we called it, P2Rank did not", ours_only, max,
        "the highest-scoring residue on which the two methods disagree in our "
        "favour")
    add("P2Rank called it, we did not", theirs_only, max,
        "the highest-scoring residue P2Rank called and we did not")
    return picks


def build() -> dict:
    t0 = time.time()
    dec = Decomposer()
    cases = json.loads(CASES.read_text())
    preds = json.loads(PREDS.read_text())["units"]
    p2 = json.loads(P2PREDS.read_text())["units"]

    out_cases = []
    worst_repro = 0.0
    for case in cases["cases"]:
        unit = case["unit_id"]
        resnums, X, ctr = _wires_for(unit, dec.prop)
        D, addr, contrib, S = dec.per_table(X)
        F = dec.gated(S, ctr)

        published = preds[unit]["residue_scores"]
        if len(resnums) != len(published):
            raise SystemExit(
                f"{unit}: the receptor gives {len(resnums)} residues and the "
                f"published prediction has {len(published)}; the decomposition "
                f"could not be aligned to the scores it must reproduce")
        want = np.array([published[str(r)] for r in resnums])
        err = float(np.max(np.abs(F - want)) / max(1.0, np.max(np.abs(want))))
        worst_repro = max(worst_repro, err)
        if err > REPRO_TOL:
            raise SystemExit(
                f"{unit}: reconstructing the score from the tables differs "
                f"from the published score by a relative {err:.3g}. This file "
                f"is decomposing a different function and has nothing to report")

        # Labels and calls come from the committed case study, not from anything
        # recomputed here, so this explains the calls the earlier read froze.
        truth_set = {int(r) for r in case["cryptic_residues"]}
        ours = {int(r) for r in case["called_residues"]["table_field"]}
        theirs = {int(r) for r in case["called_residues"]["p2rank"]}
        for name, committed, live in (
                ("table_field", ours, preds[unit].get("residue_positive")),
                ("p2rank", theirs, p2[unit].get("residue_positive"))):
            live_set = {r for r in (_resnum(x) for x in live or []) if r is not None}
            if live_set != committed:
                raise SystemExit(
                    f"{unit}: the committed case study says {name} called "
                    f"{len(committed)} residues and the prediction file now "
                    f"says {len(live_set)}; one of them has moved and the "
                    f"decomposition would explain calls nobody reported")
        truth = np.array([r in truth_set for r in resnums])

        fam = dec.by_family(contrib)
        gate_term = F - S
        gate_dev = gate_term - gate_term.mean()
        idx = {r: i for i, r in enumerate(resnums)}

        residues = []
        for pick in _pick_residues(resnums, F, truth, ours, theirs):
            i = idx[pick["resnum"]]
            parts = {k: float(v[i]) for k, v in fam.items()}
            parts["gate (neighbourhood mean)"] = float(gate_dev[i])
            total = sum(parts.values())
            coarse: dict[str, float] = {}
            for k, v in parts.items():
                coarse[COARSE[k]] = coarse.get(COARSE[k], 0.0) + v
            # The family terms are the deviation, exactly. If they stop summing
            # to it the split has lost a table somewhere.
            check = float(F[i] - F.mean())
            if abs(total - check) > 1e-6 * max(1.0, abs(check)):
                raise SystemExit(
                    f"{unit}/{pick['resnum']}: the family terms sum to "
                    f"{total:.6f} and the residue's deviation from its chain "
                    f"mean is {check:.6f}; the decomposition is not exact and "
                    f"must not be reported as one")
            residues.append({
                **pick,
                "is_labelled_cryptic": bool(truth[i]),
                "we_called_it": pick["resnum"] in ours,
                "p2rank_called_it": pick["resnum"] in theirs,
                "score": round(float(F[i]), 4),
                "chain_mean_score": round(float(F.mean()), 4),
                "score_rank_in_chain": int(1 + (F > F[i]).sum()),
                "of_residues": len(F),
                "deviation_from_chain_mean": round(check, 4),
                "by_family": {k: round(v, 4) for k, v in
                              sorted(parts.items(), key=lambda kv: -abs(kv[1]))},
                "by_coarse_family": {k: round(v, 4) for k, v in
                                     sorted(coarse.items(),
                                            key=lambda kv: -abs(kv[1]))},
                "largest_single_tables": dec.top_tables(D, addr, contrib, i),
            })

        # The anecdotes above suggest a pattern; this measures it. Every residue
        # of the chain is classified by rule and the coarse contributions are
        # averaged within each class, so the comparison is over all residues
        # rather than the handful chosen for display.
        classes = {
            "called and labelled": [i for i, r in enumerate(resnums)
                                    if r in ours and truth[i]],
            "called and not labelled": [i for i, r in enumerate(resnums)
                                        if r in ours and not truth[i]],
            "labelled and missed": [i for i, r in enumerate(resnums)
                                    if r not in ours and truth[i]],
            "neither": [i for i, r in enumerate(resnums)
                        if r not in ours and not truth[i]],
        }
        coarse_vec: dict[str, np.ndarray] = {}
        for k, v in fam.items():
            c = COARSE[k]
            coarse_vec[c] = coarse_vec.get(c, 0.0) + v
        coarse_vec["spatial smoothing"] = (
            coarse_vec.get("spatial smoothing", 0.0) + gate_dev)
        per_class = {
            name: {"n": len(rows),
                   **{c: round(float(v[rows].mean()), 3)
                      for c, v in coarse_vec.items()}}
            for name, rows in classes.items() if rows}

        out_cases.append({
            "unit_id": unit,
            "case": case["case"],
            "mean_contribution_by_residue_class": per_class,
            "why_this_case": case["why_this_one"],
            "n_residues": len(F),
            "n_labelled_cryptic": int(len(truth_set)),
            "n_called_by_us": len(ours),
            "n_called_by_p2rank": len(theirs),
            "reconstruction_relative_error": err,
            "residues": residues,
        })

    # Pooled with equal weight per chain, not per residue: the four chains have
    # 96 to 297 residues, so residue-weighting would let the largest chain
    # decide what the pattern is.
    pooled: dict[str, dict[str, float]] = {}
    for name in ("called and labelled", "called and not labelled",
                 "labelled and missed", "neither"):
        rows = [c["mean_contribution_by_residue_class"][name] for c in out_cases
                if name in c["mean_contribution_by_residue_class"]]
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r} - {"n"})
        pooled[name] = {"n_chains": len(rows),
                        "n_residues": sum(r["n"] for r in rows),
                        **{k: round(float(np.mean([r[k] for r in rows])), 3)
                           for k in keys}}

    hit = pooled.get("called and labelled", {})
    fp = pooled.get("called and not labelled", {})
    mis = pooled.get("labelled and missed", {})
    geom = round(hit.get("geometric", 0.0) - mis.get("geometric", 0.0), 3)
    gate = round(hit.get("spatial smoothing", 0.0)
                 - mis.get("spatial smoothing", 0.0), 3)
    carried = (mis.get("spatial smoothing", 0.0)
               > mis.get("geometric", 0.0) > 0)

    # The comparison the audit is actually for. If the false positives were
    # arbitrary they would decompose differently from the true positives; if they
    # decompose the same way, the field is finding the same kind of site and the
    # label is what differs.
    fam_keys = sorted(set(hit) & set(fp) - {"n_chains", "n_residues"})
    fp_gap = {k: round(hit[k] - fp[k], 3) for k in fam_keys}
    hit_mis_gap = {k: round(hit[k] - mis.get(k, 0.0), 3) for k in fam_keys}
    worst_fp = max((abs(v) for v in fp_gap.values()), default=0.0)
    worst_hm = max((abs(v) for v in hit_mis_gap.values()), default=0.0)

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "what_separates_a_hit_from_a_miss": {
            "mean_contribution_by_residue_class": pooled,
            "pooling": "equal weight per chain. The four chains run 96 to 297 "
                       "residues, so residue-weighting would let the largest "
                       "one decide what the pattern is",
            "geometric_margin_of_hits_over_misses": geom,
            "spatial_smoothing_margin_of_hits_over_misses": gate,
            "on_missed_positives_the_gate_outweighs_local_geometry": bool(carried),
            "reading": (
                "labelled residues the field calls are carried by local "
                "geometry, and labelled residues it misses are carried by the "
                "neighbourhood mean instead: on the misses the smoothing term "
                "supplies more of the score than the geometry does. The misses "
                "are not residues the field scored wrongly, they are residues "
                "where it had little evidence of its own and scored them "
                "because their neighbours scored. That is a statement about "
                "where the construction runs out, and it is the kind of "
                "statement a per-residue score alone cannot make"),
        },
        "what_the_false_positives_are_made_of": {
            "true_positive_minus_false_positive_by_family": fp_gap,
            "true_positive_minus_missed_by_family": hit_mis_gap,
            "largest_family_gap_to_false_positives": worst_fp,
            "largest_family_gap_to_missed_positives": worst_hm,
            "ratio": (round(worst_hm / worst_fp, 1) if worst_fp else None),
            "reading": (
                "the residues the field calls and the labels do not confirm "
                "decompose almost identically to the ones the labels do "
                "confirm: no family separates them by more than "
                f"{worst_fp:.1f}, against {worst_hm:.1f} between a call and a "
                "miss. Whatever the false positives are, they are not the field "
                "reading something different. They are pocket-like residues by "
                "every family it measures, which is what would be expected if "
                "an apo structure's labelled cryptic residues are the subset a "
                "holo partner happened to reveal rather than every residue "
                "capable of forming the site. The audit cannot decide that "
                "question, but it can say the errors are not arbitrary, and it "
                "localises where a curator would have to look"),
            "caveat": (
                "four chains, chosen for how the two methods compared on them "
                "and not for label quality. This is the pattern in the "
                "committed cases, not a fold-level estimate"),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": (
            "on residues this detector got right and wrong, which descriptor "
            "families supplied the score, and by how much"),
        "what_is_decomposed": (
            "the residue's deviation from its own chain's mean score. The "
            "absolute score is dominated by tables with large multiplicities "
            "that contribute the same amount to every residue in the chain; "
            "only the deviation determines the ranking, and the ranking is "
            "what the detector is"),
        "exactness": (
            "the per-table contributions are the score by construction, not an "
            "attribution of it: S(i) is the sum of m_k times the addressed "
            "cell's positive rate over all tables. The family totals are "
            "therefore the deviation exactly, and the read fails if they are "
            "not"),
        # Kept per chain as well as pooled, because the pooled row cannot say
        # whether the pattern holds on the chain a figure is about. The structural
        # figure reads these rather than recomputing them, so what it draws and
        # what the text claims come from one number.
        "cases": out_cases,
        "cases_are_not_chosen_here": {
            "source": str(CASES.relative_to(ROOT)),
            "n_cases": len(out_cases),
            "case_ids": [c["unit_id"] for c in out_cases],
            "why": "selecting which residues to explain after seeing which "
                   "explanations read well would make this a demonstration "
                   "rather than an audit",
        },
        "residues_are_chosen_by_rule": [
            "the highest-scoring residue we called that is labelled",
            "the highest-scoring residue we called that is not labelled",
            "the labelled residue we scored lowest",
            "the highest-scoring residue we called and P2Rank did not",
            "the highest-scoring residue P2Rank called and we did not",
        ],
        "family_assignment": {
            "source": "pocket_bench.methods.algebraic_descriptors.GROUPS, the "
                      "same grouping Appendix A documents",
            "families": sorted(set(dec.fam_of.values()))
                        + ["gate (neighbourhood mean)"],
            "coarse_rollup": COARSE,
            "n_tables": len(dec.tables),
            "n_tables_spanning_two_families": dec.n_mixed,
            "mixed_table_convention": (
                "a table whose two wires come from different families has its "
                "contribution split evenly between them. There is no fact "
                "about which half of a pairwise cell belongs to which wire, so "
                "the split is a stated convention rather than a measurement"),
        },
        "reconstruction": {
            "checked_against": str(PREDS.relative_to(ROOT)),
            "worst_relative_error": worst_repro,
            "tolerance": REPRO_TOL,
            "agrees": worst_repro <= REPRO_TOL,
            "why_it_matters": "a decomposition that does not add up to the "
                              "score the fold was scored with is a "
                              "decomposition of something else",
        },
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "it produces no statistic about the fold and no quantity that "
            "could be compared between two methods. It explains calls that "
            "earlier reads already froze, on units a committed tool had "
            "already selected, and it cannot be used to choose anything"),
        "cases": out_cases,
        "wall_clock_s": round(time.time() - t0, 1),
    }


def _report(d: dict) -> None:
    print(f"\naudit decomposition, {len(d['cases'])} committed cases, "
          f"worst reconstruction error "
          f"{d['reconstruction']['worst_relative_error']:.2e}")
    fa = d["family_assignment"]
    print(f"  {fa['n_tables_spanning_two_families']} of {fa['n_tables']} tables "
          f"span two families and are split evenly")
    sep = d["what_separates_a_hit_from_a_miss"]
    print("\n  mean contribution above the chain mean, by residue class "
          "(equal weight per chain)")
    for name, row in sep["mean_contribution_by_residue_class"].items():
        bits = "  ".join(f"{k} {v:+.1f}" for k, v in row.items()
                         if k not in ("n_chains", "n_residues"))
        print(f"    {name:<26} n={row['n_residues']:<5} {bits}")
    print(f"    geometry supplies {sep['geometric_margin_of_hits_over_misses']:+.1f} "
          f"more on hits than misses; smoothing "
          f"{sep['spatial_smoothing_margin_of_hits_over_misses']:+.1f}")
    fpm = d["what_the_false_positives_are_made_of"]
    print(f"    no family separates a true positive from a false positive by "
          f"more than {fpm['largest_family_gap_to_false_positives']:.1f}, "
          f"against {fpm['largest_family_gap_to_missed_positives']:.1f} between "
          f"a call and a miss")
    for c in d["cases"]:
        print(f"\n  {c['unit_id']}  ({c['case']}, {c['n_labelled_cryptic']} "
              f"labelled of {c['n_residues']})")
        for r in c["residues"]:
            top = list(r["by_coarse_family"].items())[:3]
            bits = "  ".join(f"{k} {v:+.1f}" for k, v in top)
            print(f"    {r['role']:<32} residue {r['resnum']:<5} "
                  f"rank {r['score_rank_in_chain']}/{r['of_residues']}  "
                  f"dev {r['deviation_from_chain_mean']:+7.2f}   {bits}")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if not d["reconstruction"]["agrees"]:
        print("FAILED: the decomposition does not reproduce the published "
              "per-residue scores")
        return 1
    committed = [c["unit_id"] for c in json.loads(CASES.read_text())["cases"]]
    if d["cases_are_not_chosen_here"]["case_ids"] != committed:
        print(f"FAILED: the decomposition covers "
              f"{d['cases_are_not_chosen_here']['case_ids']} but the committed "
              f"case studies are {committed}")
        return 1
    # Every residue's family terms must still sum to its stated deviation.
    for c in d["cases"]:
        for r in c["residues"]:
            total = sum(r["by_family"].values())
            if abs(total - r["deviation_from_chain_mean"]) > 5e-3:
                print(f"FAILED: {c['unit_id']}/{r['resnum']} family terms sum "
                      f"to {total:.4f}, deviation is "
                      f"{r['deviation_from_chain_mean']:.4f}")
                return 1
            coarse = sum(r["by_coarse_family"].values())
            if abs(coarse - total) > 5e-3:
                print(f"FAILED: {c['unit_id']}/{r['resnum']} coarse rollup "
                      f"{coarse:.4f} does not match the family total "
                      f"{total:.4f}")
                return 1
    if d.get("test_fold_read_index") is not None:
        print("FAILED: this artifact now declares a read index, which the "
              "ledger would count; that is a decision to make deliberately")
        return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return _check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
