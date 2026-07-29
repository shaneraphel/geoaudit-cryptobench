#!/usr/bin/env python3
"""Appendix C: the compile, the boundary cases, and where the units came from.

Appendix A states the input contract -- what the 43 local quantities are and how
they become 645 wires. This states everything between those wires and a score:
how the table bank is drawn and from which seed, what a cell holds and what an
unvisited cell holds instead, the objective the integer fan-out solves and the
rule that rounds it, the gate, and the operating point. It then states the
boundary cases that the formulas do not, and the path from the published
CryptoBench folds to the 770 and 192 units this paper evaluates.

Everything is read from the shipped artifact and from the modules that produced
it. Two things are checked rather than described: that the recorded seed really
does regenerate the shipped bank pair for pair, and that the wire coverage the
main text claims is the coverage the bank has. The second of those turned out to
be wrong -- 645 is odd, so a partition into pairs leaves one wire over in every
round -- and the appendix now says so because the check made it impossible not to.

Usage:
  PYTHONPATH=src python3.12 tools/emit_compile_appendix.py
  PYTHONPATH=src python3.12 tools/emit_compile_appendix.py --check
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
PROVENANCE = ROOT / "data/cryptobench_apo/PROVENANCE.json"
OSF_FOLDS = ROOT / "data/cryptobench_apo/_osf/folds.json"
OSF_DATASET = ROOT / "data/cryptobench_apo/_osf/dataset.json"
OUT = ROOT / "paper/appendix_c_compile.tex"

TRAIN_FOLDS = ("train-0", "train-1", "train-2", "train-3")


def _tex(s: str) -> str:
    return s.replace("_", r"\_")


def _plural(n: int, noun: str) -> str:
    return f"one {noun}" if n == 1 else f"{n} {noun}s"


def _sample_flow() -> dict:
    """Where every unit went, counted from the deposit and not from ourselves.

    The manifests carry an ``n_fold_units`` that reads like the top of the
    funnel and is not: it is the count *after* the multi-chain apo entries are
    dropped. Subtracting the exclusion from it, as a first version of this
    appendix did, removes the same 123 units twice and prints a column that does
    not add up. The counts here are recomputed from the OSF deposit, so the
    published side of the flow is measured rather than restated, and the
    manifest figures become a check on it rather than its source.
    """
    folds = json.loads(OSF_FOLDS.read_text())
    dataset = json.loads(OSF_DATASET.read_text())

    def count(pdb_ids: list[str]) -> tuple[int, int, int]:
        pairs = {(p, rec["apo_chain"]) for p in pdb_ids for rec in dataset[p]}
        # CryptoBench names a compound apo chain by joining the chain letters
        # with hyphens, as in M-O-P; that is the whole test for a multi-chain
        # assembly and the reason those units cannot be labelled by residue
        # number alone.
        multi = sum(1 for _, ch in pairs if "-" in ch)
        return len(set(pdb_ids)), len(pairs), multi

    train_pdbs = [p for k in TRAIN_FOLDS for p in folds[k]]
    out = {}
    for side, pdbs, man, skip_key in (
            ("train", train_pdbs, json.loads(TRAIN_MANIFEST.read_text()),
             "n_skipped_fetch"),
            ("test", folds["test"], json.loads(TEST_MANIFEST.read_text()),
             "n_skipped")):
        n_pdbs, n_pairs, n_multi = count(pdbs)
        skipped = man.get(skip_key, man.get("n_skipped", 0))
        row = {"pdbs": n_pdbs, "units": n_pairs, "multichain": n_multi,
               "skipped": skipped, "evaluated": man["n_entries"]}
        # Three independent statements have to agree: the deposit we recount,
        # the manifest's own post-exclusion count, and the number of receptors
        # actually on disk. A column that does not close means one of the three
        # moved, and printing it anyway is how a supplement acquires a number
        # nobody can reproduce.
        if n_pairs - n_multi != man["n_fold_units"]:
            raise SystemExit(
                f"{side}: the deposit has {n_pairs} apo chain units of which "
                f"{n_multi} are multi-chain, leaving {n_pairs - n_multi}, but "
                f"the manifest records n_fold_units={man['n_fold_units']}")
        if n_pairs - n_multi - skipped != row["evaluated"]:
            raise SystemExit(
                f"{side}: {n_pairs} units less {n_multi} multi-chain less "
                f"{skipped} unfetchable is {n_pairs - n_multi - skipped}, but "
                f"{row['evaluated']} were evaluated")
        if len(man["entries"]) != row["evaluated"]:
            raise SystemExit(
                f"{side}: the manifest claims {row['evaluated']} entries and "
                f"lists {len(man['entries'])}")
        out[side] = row
    return out


def _verify_boundaries() -> None:
    """Run the boundary cases the appendix describes, rather than describing them.

    A residue placed far from every other one exercises the isolated case, and a
    chain on which a wire is constant exercises the degenerate-variance case.
    Both are one call to the shipped transform, and both are the kind of claim
    that stays in a supplement long after the code stopped honouring it.
    """
    import numpy as np

    from pocket_bench.methods.wide_descriptors import (
        DIFF_RADII, MEAN_RADII, RANK_RADII, VAR_RADII, wide_transform)

    # Residue 0 is a thousand angstrom away; 1 and 2 are neighbours.
    local = np.array([[5.0], [1.0], [3.0]])
    ctr = np.array([[1000.0, 0, 0], [0, 0, 0], [1.0, 0, 0]])
    X = wide_transform(local, ctr, [3])

    n_mean, n_var, n_diff = len(MEAN_RADII), len(VAR_RADII), len(DIFF_RADII)
    at = {"identity": 0}
    at["mean"] = 1
    at["dispersion"] = 1 + n_mean
    at["diff"] = at["dispersion"] + n_var
    at["rank"] = at["diff"] + n_diff
    if at["rank"] + len(RANK_RADII) != X.shape[1]:
        raise SystemExit(
            f"the statistic layout is not the one this appendix describes: "
            f"{X.shape[1]} columns for "
            f"1 + {n_mean} + {n_var} + {n_diff} + {len(RANK_RADII)}")

    def bad(what: str, got, want) -> None:
        raise SystemExit(
            f"the boundary case the appendix states for {what} is not what the "
            f"code does: got {got}, expected {want}. The supplement would be "
            f"documenting behaviour the field does not have")

    if X[0, at["identity"]] != 5.0:
        bad("the identity statistic", X[0, at["identity"]], 5.0)
    for k in range(n_mean):
        if abs(X[0, at["mean"] + k] - 5.0) > 1e-12:
            bad("an isolated residue's neighbourhood mean",
                X[0, at["mean"] + k], "its own value")
    for k in range(n_var):
        if abs(X[0, at["dispersion"] + k]) > 1e-12:
            bad("an isolated residue's dispersion", X[0, at["dispersion"] + k], 0.0)
    for k in range(n_diff):
        if abs(X[0, at["diff"] + k]) > 1e-12:
            bad("an isolated residue's centred difference",
                X[0, at["diff"] + k], 0.0)
    for k in range(len(RANK_RADII)):
        if abs(X[0, at["rank"] + k]) > 1e-12:
            bad("an isolated residue's local rank", X[0, at["rank"] + k], 0.0)
    # Self-inclusion: residue 2 is the larger of the pair {1, 2}, so its rank is
    # 1/2 and not 1. If the neighbourhood ever stops containing the residue, this
    # is what changes first.
    if abs(X[2, at["rank"]] - 0.5) > 1e-12:
        bad("the local rank of the largest residue in its neighbourhood "
            "(self-inclusion)", X[2, at["rank"]], 0.5)

    # A constant wire over a whole chain: variance must be exactly zero and not
    # a small negative number under a square root.
    flat = wide_transform(np.array([[2.0], [2.0], [2.0]]),
                          np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]), [3])
    if not np.all(np.isfinite(flat)):
        bad("a constant wire", "a non-finite value", "all finite")
    for k in range(n_var):
        if abs(flat[0, at["dispersion"] + k]) > 1e-12:
            bad("a constant wire's dispersion", flat[0, at["dispersion"] + k], 0.0)


def facts() -> dict:
    """Every number in the appendix, read from the artifacts that own it."""
    from pocket_bench.methods.table_bank import partition_tables

    d = json.loads(FIELD.read_text())
    tables = [list(t) for t in d["tables"]]

    # The seed is recorded in the artifact. Whether it is the seed that produced
    # the artifact is a different question, and the only way to answer it is to
    # draw the bank again and compare.
    redrawn = partition_tables(d["n_wires"], d["table_width"],
                               d["partition_rounds"], d["partition_seed"])
    if [list(t) for t in redrawn] != tables:
        raise SystemExit(
            f"the recorded partition seed {d['partition_seed']} does not "
            f"regenerate the shipped bank. The appendix would be documenting a "
            f"draw nobody can reproduce")

    _verify_boundaries()

    per_wire = collections.Counter(w for t in tables for w in t)
    counts = collections.Counter(per_wire.values())
    width, rounds = d["table_width"], d["partition_rounds"]
    full = d["n_wires"] // width
    leftover = d["n_wires"] - full * width

    train = json.loads(TRAIN_MANIFEST.read_text())
    test = json.loads(TEST_MANIFEST.read_text())
    prov = json.loads(PROVENANCE.read_text())
    scope = prov["evaluable_scope"]

    return {
        "field": d,
        "n_tables": len(tables),
        "tables_per_round": full,
        "leftover_per_round": leftover,
        "n_wires_covered": len(per_wire),
        "appearance_counts": sorted(counts.items()),
        "n_wires_short": counts.get(rounds - 1, 0),
        "train": train,
        "test": test,
        "prov": prov,
        "scope": scope,
        "flow": _sample_flow(),
    }


def build() -> str:
    f = facts()
    d = f["field"]
    gate, op, tr = d["gate"], d["operating_point"], d["train"]
    L: list[str] = []
    A = L.append

    A("% GENERATED by tools/emit_compile_appendix.py -- do not edit by hand.")
    A("% Every number is read from data/cryptobench_apo/TABLE_FIELD.json and the")
    A("% dataset manifests. Regenerate with `make compileapp`.")
    A("")
    A(r"\section{Appendix C: the compile, the boundary cases, and the units}")
    A(r"\label{app:compile}")
    A("")
    A(r"Appendix~\ref{app:wires} states what the \(43\) local quantities are and "
      r"how they expand into the \NTabWires{} wires. This states everything "
      r"between a wire and a score, and then the two things a set of formulas "
      r"cannot state: what happens at the boundaries, and which structures the "
      r"numbers were computed over.")
    A("")

    # ------------------------------------------------------------------ bank
    A(r"\subsection{The table bank and its seed}")
    A("")
    A(rf"The bank is {d['partition_rounds']} independent random partitions of "
      rf"the \NTabWires{{}} wires into groups of {d['table_width']}, drawn from "
      rf"\texttt{{numpy.random.default\_rng}} seeded with "
      rf"\({d['partition_seed']}\). The seed is recorded in the compiled "
      rf"artifact and the generator of this appendix redraws the bank from it "
      rf"and refuses to emit unless the redraw matches the shipped bank pair "
      rf"for pair.")
    A("")
    A(r"A partition rather than a draw of random tuples: within one round every "
      r"wire appears at most once, so no wire is over-represented, and across "
      r"rounds each wire meets a different set of partners. Coverage of the "
      r"digit-pair lattice comes from the number of rounds and not from the "
      r"width of any table.")
    A("")
    A(rf"\NTabWires{{}} is odd, so a partition into pairs leaves one wire over in "
      rf"every round, and a group of one addresses no pair and is dropped. Each "
      rf"round therefore contributes {f['tables_per_round']} tables and the bank "
      rf"holds \({f['tables_per_round']} \times {d['partition_rounds']} = "
      rf"\NTabTables{{}}\). The consequence is worth stating because it is easy "
      rf"to assume otherwise: {f['n_wires_covered'] - f['n_wires_short']} wires "
      rf"appear in {d['partition_rounds']} tables and "
      rf"{f['n_wires_short']} appear in {d['partition_rounds'] - 1}, one for "
      rf"each round, and which wire is left out is a property of the seed. "
      rf"Every wire is in the bank; not every wire is in it equally often.")
    A("")

    # ----------------------------------------------------------------- cells
    A(r"\subsection{What a cell holds}")
    A("")
    A(rf"Each table \(k\) of width {d['table_width']} over \(L = "
      rf"{d['n_levels']}\) levels has \(L^{{{d['table_width']}}} = "
      rf"{d['n_levels'] ** d['table_width']}\) addresses, for "
      rf"\NTabCells{{}} cells across the bank. A cell holds two integers counted "
      rf"on the training fold: \(\mathrm{{tot}}_k[a]\), the residues that "
      rf"addressed it, and \(\mathrm{{pos}}_k[a]\), those of them that are "
      rf"labelled cryptic binding residues. The value read at inference is")
    A(r"\begin{equation}")
    A(r"  \hat p_k[a] = \begin{cases}")
    A(r"    \mathrm{pos}_k[a] / \mathrm{tot}_k[a], & \mathrm{tot}_k[a] > 0,\\[2pt]")
    A(rf"    \bar y = {tr['base_rate']:.6f}, & \mathrm{{tot}}_k[a] = 0,")
    A(r"  \end{cases}")
    A(r"\end{equation}")
    A(rf"where \(\bar y\) is the training fold's base rate. An address no "
      rf"training residue reached takes that base rate because it is the only "
      rf"value that adds nothing to the score, and there are \NTabCellsEmpty{{}} "
      rf"such cells. No smoothing is applied to the cells that were reached: a "
      rf"cell visited once holds \(0\) or \(1\), and the integer fan-out is what "
      rf"decides how much such a cell is allowed to say.")
    A("")

    # ----------------------------------------------------------------- solve
    A(r"\subsection{The integer fan-out}")
    A("")
    A(r"Write \(v_k(i) = \hat p_k[a_k(i)]\) for the value table \(k\) returns on "
      r"residue \(i\), and \(v(i) \in \mathbb{R}^{\NTabTables{}}\) for the "
      r"stacked vector. Let \(\mu_1\) and \(\mu_0\) be its means over the "
      r"labelled and unlabelled training residues and \(S\) its pooled scatter. "
      r"The fan-out is the rounded, rescaled solution of one regularised "
      r"symmetric system:")
    A(r"\begin{align}")
    A(rf"  \tilde S &= S + \left( \lambda \, \frac{{\operatorname{{tr}} S}}"
      rf"{{K}} + \varepsilon \right) I, \qquad \lambda = \TabRidge{{}}, \quad "
      rf"\varepsilon = 10^{{-12}}, \quad K = \NTabTables{{}},\\")
    A(r"  w &= \tilde S^{-1} (\mu_1 - \mu_0),\\")
    A(r"  m_k &= \operatorname{round}\!\left( \frac{w_k}{\max_j |w_j|} \, "
      r"C \right), \qquad C = \TabCap{}.")
    A(r"\end{align}")
    A("")
    A(r"Three points. The ridge is proportional to the mean diagonal of the "
      r"scatter rather than absolute, so it means the same thing whatever scale "
      r"the cell rates happen to sit on, and it is not cosmetic: pairs drawn "
      r"from the lattice repeat as the pool grows, the scatter goes "
      r"near-singular, and without regularisation the direction chases the null "
      r"space. The normalisation is by the largest coordinate, not the norm, so "
      r"the cap is attained by exactly the tables the solve ranked highest and "
      r"the rounding grid is the same regardless of how many tables there are. "
      r"And the rounding is to nearest, which sends every table whose weight is "
      rf"below half a grid step to zero --- which is why \NTabUsedFullFold{{}} of "
      rf"\NTabTables{{}} tables carry a non-zero multiplicity and the rest are "
      rf"compiled but never consulted. The solve is closed form: no gradient, no "
      rf"iteration, and nothing selected on the held-out fold.")
    A("")

    # ------------------------------------------------------------------ gate
    A(r"\subsection{The gate and the operating point}")
    A("")
    A(rf"The raw score is \(S(i) = \sum_k m_k v_k(i)\). The gate adds the mean "
      rf"of \(S\) over the residues within "
      rf"\({gate['radius_angstrom']:g}\)~\AA\ of residue \(i\)'s centroid, "
      rf"inside its own chain, rescaled so that the added term has the same "
      rf"standard deviation as \(S\) before it is weighted:")
    A(r"\begin{equation}")
    A(rf"  F(i) = S(i) + {gate['weight']:g} \cdot \bar S_{{"
      rf"{gate['radius_angstrom']:g}}}(i) \cdot "
      rf"\frac{{\operatorname{{sd}}(S)}}{{\operatorname{{sd}}(\bar S_{{"
      rf"{gate['radius_angstrom']:g}}})}}.")
    A(r"\end{equation}")
    A(r"Matching the standard deviation rather than the maximum matters because "
      r"the maximum of a score field over a chain is an order statistic of a "
      r"handful of residues, so a max-normalised gate would mix in a different "
      r"amount on every structure. Both constants were chosen on a "
      rf"cluster-disjoint half of the training fold. If \(\operatorname{{sd}}"
      rf"(\bar S)\) is zero the gate is skipped, which can only happen on a "
      rf"chain whose residues all score alike.")
    A("")
    A(rf"The positive call is the top \({100 * op['q']:g}\%\) of residues by "
      rf"\(F\) within each chain, \(k = \max(1, \operatorname{{round}}(q n))\), "
      rf"ties broken by ascending residue index so the rule is deterministic. "
      rf"The threshold \(q = {op['q']}\) is the argmax of the pooled F1 curve on "
      rf"the training fold and was frozen before the held-out fold was "
      rf"binarised.")
    A("")

    # -------------------------------------------------------------- boundary
    A(r"\subsection{Boundary cases}")
    A("")
    A(r"These are the cases the formulas above do not determine. Each is a "
      r"decision in the code, and each is stated here because the alternative "
      r"decisions are all defensible and a reader cannot tell which was taken.")
    A("")
    A(r"\begin{description}")
    A(r"  \item[The neighbourhood is never empty] \(N_r(i)\) is "
      r"\(\{j : \|c_j - c_i\| \le r\}\), which contains \(i\) itself, so no "
      r"statistic ever divides by zero and there is no isolated-residue case to "
      r"define. A residue with no other residue inside \(r\) therefore takes its "
      r"own value as the mean, \(0\) for the dispersion, \(0\) for the centred "
      r"difference and \(0\) for the local rank. Self-inclusion also means the "
      r"local rank of the largest residue in its neighbourhood is "
      r"\((|N_r(i)|-1)/|N_r(i)|\) rather than \(1\), and that the dispersion is "
      r"a population standard deviation over a set that includes the point it "
      r"is centred on.")
    A(r"  \item[Zero variance] Dispersions are computed as \(\sqrt{\max(\langle "
      r"x^2 \rangle - \langle x \rangle^2, 0)}\), so floating-point cancellation "
      r"on a neighbourhood of near-identical values cannot produce a negative "
      r"variance or a NaN.")
    A(r"  \item[Ties in a rank] In the banding, tied values share the mid-rank "
      r"of the run, so the digit assignment does not depend on the order the "
      r"residues happen to appear in the file. A wire constant across a chain "
      r"gives every residue the same mid-rank and lands them all in one digit, "
      r"which addresses one cell per table and contributes the same amount to "
      r"every residue's score --- visible in the ranking not at all, which is why "
      r"the audit of Section~\ref{sec:audit} decomposes deviations from the "
      r"chain mean rather than absolute scores.")
    A(r"  \item[Missing atoms] The residue universe is the distinct residue "
      r"numbers among \texttt{ATOM} records whose element is not hydrogen, so a "
      r"residue enters if it has at least one non-hydrogen atom, and its "
      r"centroid is the centroid of the atoms actually present. A residue with "
      r"no such atom is not in the universe at all: it is neither scored nor "
      r"counted against recall. Side chains resolved only in part are therefore "
      r"scored from what was resolved, without imputation.")
    A(r"  \item[Non-standard residues] A residue type outside the twenty takes "
      r"the mean of the twenty standard values for each of the seven "
      r"physicochemical constants, and the neutral prior \(1/2\) for the "
      r"propensity. Both are neutral in a rank order rather than extreme in it, "
      r"which is what a sentinel value would have been.")
    A(r"  \item[Chain boundaries] No neighbourhood crosses a chain. Every "
      r"statistic, every rank and every quantisation is computed inside the "
      r"chain being scored, which is what allows a chain to be scored without "
      r"reference to any other structure.")
    A(r"  \item[Heteroatoms] A ligand-leak guard rejects any non-solvent "
      r"heteroatom before feature extraction, so a receptor that still contains "
      r"its ligand raises rather than scoring.")
    A(r"\end{description}")
    A("")

    # ------------------------------------------------------------------ flow
    A(r"\subsection{From the published folds to \NTrainUnits{} and \NUnits{} units}")
    A("")
    A(rf"The dataset is CryptoBench ({_tex(f['prov']['citation'])}), taken from "
      rf"its OSF deposit at \url{{{f['prov']['osf_node']}}}; the SHA-256 of every "
      rf"fetched file is checked against the digest OSF reports. The published "
      rf"split is MMseqs2 clustering at "
      rf"\({100 * f['test']['clustering']['sequence_identity_threshold']:g}\%\) "
      rf"sequence identity, "
      rf"\({100 * f['test']['clustering']['coverage']:g}\%\) coverage, split "
      rf"80:20. Receptors are fetched from RCSB, scoped to the chain the fold "
      rf"names, and stripped of ligands.")
    A("")
    A(r"\begin{center}")
    A(r"\begin{tabular}{@{}lrr@{}}")
    A(r"\toprule")
    A(r" & training & test \\")
    A(r"\midrule")
    fl = f["flow"]
    A(rf"apo structures named by the fold & {fl['train']['pdbs']} & "
      rf"{fl['test']['pdbs']} \\")
    A(rf"apo chains in those structures & {fl['train']['units']} & "
      rf"{fl['test']['units']} \\")
    A(rf"less multi-chain apo chains & $-{fl['train']['multichain']}$ & "
      rf"$-{fl['test']['multichain']}$ \\")
    A(rf"less chains RCSB would not serve & $-{fl['train']['skipped']}$ & "
      rf"$-{fl['test']['skipped']}$ \\")
    A(r"\midrule")
    A(rf"evaluated & \NTrainUnits{{}} & \NUnits{{}} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{center}")
    A("")
    A(rf"A unit is one apo chain, so a structure contributing two apo chains "
      rf"contributes two units; that is why the chain counts exceed the "
      rf"structure counts. The multi-chain exclusions are the substantive step. "
      rf"{fl['test']['multichain']} test and {fl['train']['multichain']} "
      rf"training apo chains are compound, named by joining chain letters as in "
      rf"\texttt{{M-O-P}}. The per-residue metric indexes residues by residue "
      rf"number, which is not unique across chains, so those units cannot be "
      rf"labelled unambiguously without a chain-aware index and are excluded "
      rf"rather than silently mis-scored. The consequence is that the fold "
      rf"reported here is a strict subset of the official test fold and the "
      rf"results do not transfer to multi-chain assemblies. A further "
      rf"{_plural(fl['test']['skipped'], 'test chain')} and "
      rf"{_plural(fl['train']['skipped'], 'training chain')} are absent because "
      rf"RCSB does not serve those entries in PDB format; they are listed by "
      rf"accession in the manifests, the test one being \texttt{{7nbc}} chain "
      rf"\texttt{{CCC}}.")
    A("")
    A(rf"The \NTrainUnits{{}} training units comprise \NTabTrainRes{{}} residues, "
      rf"of which \({100 * tr['base_rate']:.3f}\%\) are labelled cryptic binding "
      rf"residues; that base rate is the value an unaddressed cell returns. The "
      rf"training and test folds share no "
      rf"sequence cluster: the ledger in the compiled artifact records "
      rf"{d['cluster_ledger']['train_clusters']} training and "
      rf"{d['cluster_ledger']['test_clusters']} test clusters over "
      rf"{d['cluster_ledger']['train_units']} and "
      rf"{d['cluster_ledger']['test_units']} units.")
    A("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    want = build()
    if args.check:
        if not OUT.is_file():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        if OUT.read_text() != want:
            print(f"STALE {OUT.relative_to(ROOT)}: the compile appendix no "
                  f"longer matches the artifact and the modules it is generated "
                  f"from. Regenerate with `make compileapp`.")
            return 1
        f = facts()
        print(f"OK {OUT.relative_to(ROOT)}: seed {f['field']['partition_seed']} "
              f"regenerates the bank, {f['n_tables']} tables, "
              f"{f['n_wires_covered']} wires covered "
              f"({f['n_wires_short']} in one round fewer than the rest)")
        return 0
    OUT.write_text(want)
    f = facts()
    print(f"wrote {OUT.relative_to(ROOT)}: seed {f['field']['partition_seed']} "
          f"verified against the shipped bank, {f['n_tables']} tables, "
          f"appearance counts {f['appearance_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
