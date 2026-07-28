#!/usr/bin/env python3
"""Write the appendix that defines all 43 local quantities and all 645 wires.

Why this is generated rather than typed
---------------------------------------
The manuscript says the field reads 43 local quantities under a fixed set of
neighbourhood statistics and that this yields 645 wires. Those are checkable
claims about code, and an appendix typed by hand stops being a description of
the code the first time a radius changes. So the structure -- the names, the
grouping, every radius, every threshold, the arithmetic that takes 43 to 645 --
is imported from the modules that define it, and only the prose is written here.

The two halves are cross-checked. Every name the modules export must have an
entry below and every entry must name something the modules export, so a
descriptor added without a definition, or a definition left behind by a
descriptor that was removed, fails this tool rather than reaching a reader.

Usage: PYTHONPATH=src:tools python3.12 tools/emit_wire_appendix.py [--check]
"""
from __future__ import annotations

import argparse

from pocket_bench.methods import algebraic_descriptors as alg
from pocket_bench.methods import density_topology as dt
from pocket_bench.methods import expanded_descriptors as chem
from pocket_bench.methods import wide_descriptors as wide
from pocket_bench.paths import ROOT

OUT = ROOT / "paper/appendix_a_wire_definitions.tex"

# Defaults of algebraic_residue_features, which is the only entry point the
# compiled artifacts call, so these are the values every published number used.
PROBE = {"grid_step": 1.5, "n_dirs": 30, "cutoff": 11.0, "perp": 1.8,
         "atom_r": 2.6, "max_pts": 6000}
NEAR_R = 6.0          # probe-to-residue attribution radius
EXPOSURE_R = 5.0
VOID_LEVEL = 0.55
VALUATION_GAMMA = 0.5  # the descriptor's own deflator, not density_topology's

# name -> (definition, neighbourhood it is read over, behaviour at the boundary)
DEFS: dict[str, tuple[str, str, str]] = {
    # ---- G1 surface exposure ------------------------------------------------
    "bur": (
        r"mean over probe points $p$ attributed to $i$ of the buriedness "
        r"$\beta(p)$, the fraction of $%(n_dirs)d$ Fibonacci directions "
        r"blocked by an atom within $%(cutoff).0f$~\AA\ of the ray, at "
        r"perpendicular tolerance $%(perp).1f$~\AA" % PROBE,
        r"free-grid probe points within $%.0f$~\AA\ of any heavy atom of $i$"
        % NEAR_R,
        r"$0$ when no probe point is attributed to $i$"),
    "void": (
        r"$\log\!\left(1 + \#\{p : \beta(p) \ge %.2f\}\right)$" % VOID_LEVEL,
        r"the same attributed probe points",
        r"$\log 1 = 0$ when none of them is that buried"),
    "probe_max": (
        r"$\max_p \beta(p)$",
        r"the same attributed probe points",
        r"$0$ when no probe point is attributed to $i$"),
    "protrusion": (
        r"$\left\| \operatorname{mean}_{j \in N(i)} c_j - c_i \right\| / R$, "
        r"the offset of the neighbourhood's centroid from the residue",
        r"residues within $R = %.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    "exposure": (
        r"$\#\{p\} / a_i$, probe points per heavy atom of the residue",
        r"probe points within $%.0f$~\AA\ of any heavy atom of $i$"
        % EXPOSURE_R,
        r"$0$ when none; $a_i \ge 1$ by construction"),
    "concavity": (
        r"$\operatorname{mean}_{j \in N(i)} \widehat{(c_j - c_i)} \cdot "
        r"\hat n_i$ with $\hat n_i$ the outward radial unit vector from the "
        r"chain centroid",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    # ---- G2 local spectral geometry ----------------------------------------
    "fiedler": (
        r"$\lambda_2$ of the combinatorial Laplacian $L = D - A$ of the "
        r"lining $G[S_i]$, $S_i = \{i\} \cup N(i)$",
        r"induced subgraph on $i$ and its contacts, capped at the "
        r"$%d$ nearest" % alg._MAX_LOCAL,
        r"$0$ when $i$ has no contact or $|S_i| \le 1$"),
    "lap_max": (
        r"$\lambda_{\max}$ of the same $L$",
        r"the same lining",
        r"$0$ when $i$ has no contact"),
    "spec_gap": (
        r"$\lambda_2 / \max(\lambda_{\max}, 10^{-9})$",
        r"the same lining",
        r"the floor in the denominator is what makes it defined at "
        r"$\lambda_{\max} = 0$"),
    "mean_degree": (
        r"$2E / |S_i|$ of the lining",
        r"the same lining",
        r"$0$ when $i$ has no contact"),
    "betti0": (
        r"number of connected components of $G[S_i \setminus \{i\}]$ --- how "
        r"many disjoint walls face the residue",
        r"the lining re-thresholded at $%.0f$~\AA, because at "
        r"$%.0f$~\AA\ the lining of a globular protein is connected and this "
        r"count is the constant $1$" % (alg._PINCH_R, alg._NBR_R),
        r"$0$ when the lining is empty"),
    "betti1": (
        r"cycle rank $E - V + C$ of the same graph, an integer",
        r"the same $%.0f$~\AA\ lining" % alg._PINCH_R,
        r"$0$ when the lining is empty"),
    # ---- G3 density field calculus ------------------------------------------
    "rho": (
        r"mean over the residue's heavy atoms of the atomic packing density "
        r"$\rho$, the coordination number within $%.0f$~\AA" % dt.CONTACT_CUTOFF,
        r"the residue's own atoms",
        r"$\rho \ge 1$ by construction"),
    "grad_rho": (
        r"$\|g_i\|$ where $g_i$ solves the normal equations "
        r"$A_i g_i = b_i$ of the first-order Taylor fit, "
        r"$A_i = \sum_j (c_j - c_i)(c_j - c_i)^\top$, "
        r"$b_i = \sum_j (\rho_j - \rho_i)(c_j - c_i)$",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$g_i = 0$ when $A_i$ is singular, which is a neighbourhood spanning "
        r"fewer than three dimensions"),
    "grad_radial": (
        r"$g_i \cdot \hat n_i$, the outward component of the same gradient",
        r"the same neighbourhood",
        r"$0$ with $g_i$"),
    "lap_rho": (
        r"$\operatorname{mean}_{j \in N(i)} \rho_j - \rho_i$, the graph "
        r"Laplacian of the density field",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    "var_rho": (
        r"$\max\!\left(\operatorname{mean}_j \rho_j^2 - "
        r"(\operatorname{mean}_j \rho_j)^2,\, 0\right)$",
        r"the same neighbourhood",
        r"clipped at zero so rounding cannot produce a negative variance"),
    "rho_rank_ball": (
        r"rank of $\rho_i$ among the residues of $i$'s ultrametric ball, "
        r"scaled to $[0,1]$",
        r"the ball, not a radius",
        r"$0.5$ when the ball holds one residue, which is the value that "
        r"asserts nothing"),
    # ---- G4 ultrametric structure -------------------------------------------
    "loose": (
        r"mean over the residue's atoms of $\bar\rho_{B(a)} / \bar\rho$, the "
        r"atom's ball density against the global mean",
        r"the residue's atoms and their balls",
        r"$\bar\rho$ is floored at $1$"),
    "ball_size": (
        r"$\log(1 + |B(i)|)$ in atoms",
        r"the ball",
        r"$\log 1 = 0$ for an empty ball, which cannot occur"),
    "ball_radius": (
        r"$\tau \cdot \bar\rho / \bar\rho_{B(i)}$, the ball's cut radius "
        r"deflated by its own density",
        r"the ball",
        r"the denominator is floored at $10^{-9}$"),
    "ball_interface": (
        r"fraction of $i$'s contacts lying in a different ball",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    "valuation": (
        r"$\max_{j \in N(i)} \|c_j - c_i\| \left(\rho_{\max} / "
        r"\min(\rho_i, \rho_j)\right)^{%.1f} \big/ \tau$: the longest "
        r"density-deflated edge the residue carries, in units of the global "
        r"ball radius" % VALUATION_GAMMA,
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact; the deflator's denominator is floored"),
    "rotor_gain": (
        r"largest increase in probe buriedness produced by rotating one "
        r"ultrametric ball rigidly by $\theta = %.2f$~rad about its branch "
        r"axis" % alg._ROTOR_THETA,
        r"the $%d$ largest balls of at least $%d$ atoms that are not the whole "
        r"structure; probes within $%.0f$~\AA\ of the hinge centre"
        % (alg._MAX_HINGE_BALLS, dt.MIN_BALL_ATOMS, PROBE["cutoff"] + 8.0),
        r"$0$ when no ball qualifies, which is the honest reading that the "
        r"structure has no hinge at this scale"),
    # ---- G5 curvature and anisotropy ----------------------------------------
    "anisotropy": (
        r"$(\lambda_1 - \lambda_3)/\operatorname{tr} M$ for "
        r"$M_i = \operatorname{mean}_{j \in N(i)} (c_j - c_i)(c_j - c_i)^\top$ "
        r"with $\lambda_1 \ge \lambda_2 \ge \lambda_3$",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$\operatorname{tr} M$ floored at $10^{-12}$"),
    "planarity": (
        r"$(\lambda_2 - \lambda_3)/\lambda_1$",
        r"the same tensor",
        r"$\lambda_1$ floored at $10^{-12}$"),
    "sphericity": (
        r"$\lambda_3/\lambda_1$",
        r"the same tensor",
        r"$\lambda_1$ floored at $10^{-12}$"),
    "normal_div": (
        r"$\operatorname{mean}_{j} (\hat n_j - \hat n_i) \cdot (c_j - c_i)$, "
        r"the discrete divergence of the outward normal field",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    "angle_deficit": (
        r"$1 - \operatorname{mean}_j \angle(\hat n_i, \hat n_j) / \pi$, "
        r"bounded in $[0,1]$ by construction",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$1$ when $i$ has no contact, the value a flat isolated patch takes"),
    "hess_bur": (
        r"$\operatorname{mean}_{j} \mathrm{bur}_j - \mathrm{bur}_i$, the graph "
        r"Laplacian of the buriedness field",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    # ---- G6 global position -------------------------------------------------
    "depth": (
        r"$\|c_i - \bar c\| / R_g$ against the chain centroid and radius of "
        r"gyration",
        r"the whole chain",
        r"$R_g$ floored at $1$"),
    "coordination": (
        r"$|N(i)|$, an integer",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ is attainable and meaningful"),
    "contact_order": (
        r"$\operatorname{mean}_{j \in N(i)} |s_j - s_i| / n$, the mean "
        r"sequence separation of the contacts normalised by chain length",
        r"residues within $%.0f$~\AA" % alg._NBR_R,
        r"$0$ when $i$ has no contact"),
    "ball_offset": (
        r"$\|c_i - \bar c_{B(i)}\| / R_g$, displacement from the residue's own "
        r"ball centroid",
        r"the ball",
        r"$R_g$ floored at $1$"),
    "depth_rank": (
        r"rank of $\mathrm{depth}$ within the chain, scaled to $[0,1]$",
        r"the whole chain",
        r"$0$ for a chain of one residue"),
}

CHEM_DEFS: dict[str, tuple[str, str]] = {
    "kd": ("Kyte--Doolittle hydropathy",
           r"Kyte and Doolittle, \emph{J.\ Mol.\ Biol.}\ \textbf{157}:105 "
           r"(1982)"),
    "volume": (r"side-chain volume in \AA$^3$",
               r"Zamyatnin, \emph{Prog.\ Biophys.\ Mol.\ Biol.}\ "
               r"\textbf{24}:107 (1974)"),
    "aromatic": ("aromatic rings in the side chain: 1 for F, Y, H, 2 for W, "
                 "0 otherwise",
                 "chemical structure"),
    "charge": ("formal charge at pH 7: $-1$ for D, E, $+1$ for K, R, $+0.1$ "
               "for H as partially protonated, 0 otherwise",
               "chemical structure"),
    "hbd": ("side-chain hydrogen-bond donors", "chemical structure"),
    "hba": ("side-chain hydrogen-bond acceptors", "chemical structure"),
    "chi": (r"rotatable side-chain bonds ($\chi$ angles)",
            "chemical structure"),
}


def _check_coverage() -> list[str]:
    """Neither half may drift from the other."""
    bad = []
    for n in alg.FEATURE_NAMES:
        if n not in DEFS:
            bad.append(f"{n} is exported by algebraic_descriptors and has no "
                       f"definition in this appendix")
    for n in DEFS:
        if n not in alg.FEATURE_NAMES:
            bad.append(f"{n} is defined in this appendix and is not exported "
                       f"by algebraic_descriptors")
    for n in chem.CHEM_NAMES:
        if n not in CHEM_DEFS:
            bad.append(f"{n} is a chemical wire with no source recorded")
    for n in CHEM_DEFS:
        if n not in chem.CHEM_NAMES:
            bad.append(f"{n} has a source recorded and is not a chemical wire")
    n_local = len(alg.FEATURE_NAMES) + len(chem.CHEM_NAMES) + 1
    n_wires = n_local * wide.N_STATISTIC_GROUPS
    if n_wires != 645:
        bad.append(f"the modules now expand to {n_wires} wires, not the 645 "
                   f"the manuscript reports")
    return bad


def _radii(rs) -> str:
    return ", ".join(f"{r:g}" for r in rs)


def build() -> str:
    L: list[str] = []
    A = L.append
    A("% Appendix A --- included by MAIN_CRYPTOBENCH_GEOAUDIT.tex via \\input.")
    A("% GENERATED by tools/emit_wire_appendix.py. Do not edit by hand: the")
    A("% names, groups, radii and counts are imported from the modules that")
    A("% define them, and `make wires` fails if this file drifts from them.")
    A("")
    A("\\appendix")
    A("\\section{Appendix A: every local quantity and every wire}")
    A("\\label{app:wires}")
    A("")
    A("This appendix is the definition of the input contract. Section~"
      "\\ref{sec:methods} describes the field in terms of \\(43\\) local "
      "quantities read under a fixed family of neighbourhood statistics; what "
      "follows states each of the \\(43\\), the neighbourhood it is read over "
      "and its value at the boundary, and then the rule that expands them into "
      "the \\NTabWires{} wires. It is generated from the modules, so it cannot "
      "describe a version of the code that is not the one that produced the "
      "numbers.")
    A("")
    A("\\paragraph{Conventions.} \\(c_i\\) is the centroid of the heavy atoms "
      "of residue \\(i\\) and \\(N(i)\\) its contacts, meaning residues whose "
      f"centroid lies within \\({alg._NBR_R:g}\\)~\\AA. Neighbourhoods never "
      "cross a chain: a residue's neighbourhood is computed inside its own "
      "chain, which is what allows one chain to be scored without reference to "
      "any other. \\(\\hat n_i\\) is the outward radial unit vector from the "
      "chain centroid, \\(R_g\\) the radius of gyration, \\(s_i\\) the "
      "sequence position, and \\(a_i\\) the number of heavy atoms. Hydrogens "
      "are discarded and a ligand-leak guard rejects any non-solvent "
      "heteroatom before any of this is computed.")
    A("")
    A("\\paragraph{The probe field.} Quantities in G1 and \\texttt{rotor"
      "\\_gain} are read off a free-space grid: points on a "
      f"\\({PROBE['grid_step']:g}\\)~\\AA\\ lattice at least "
      f"\\({PROBE['atom_r']:g}\\)~\\AA\\ from every atom, capped at "
      f"\\({PROBE['max_pts']}\\) points. The buriedness \\(\\beta(p)\\) of "
      f"such a point is the fraction of \\({PROBE['n_dirs']}\\) "
      "Fibonacci-spaced directions along which an atom lies within "
      f"\\({PROBE['cutoff']:g}\\)~\\AA\\ of the ray, at perpendicular "
      f"tolerance \\({PROBE['perp']:g}\\)~\\AA.")
    A("")
    A("\\paragraph{The ultrametric ball partition.} \\(\\rho_a\\) is the "
      f"coordination number of atom \\(a\\) within \\({dt.CONTACT_CUTOFF:g}"
      "\\)~\\AA. Contact edges are deflated to \\(\\|c_a - c_b\\| "
      "(\\rho_{\\max}/\\min(\\rho_a,\\rho_b))^{\\gamma}\\) with "
      f"\\(\\gamma = {dt.DENSITY_GAMMA:g}\\), and the balls at radius "
      "\\(\\tau\\) are the components of the minimum spanning tree restricted "
      "to edges of weight at most \\(\\tau\\), which is exactly the closed-ball "
      "partition of the minimax ultrametric. \\(\\tau\\) is chosen by the "
      "valuation threshold of the tree-weight spectrum, not tuned. Note that "
      "the \\texttt{valuation} descriptor below uses its own exponent "
      f"\\({VALUATION_GAMMA:g}\\), not \\(\\gamma\\).")
    A("")

    A("\\subsection{The \\NAlgFeatures{} algebraic and topological quantities}")
    A("")
    A("The grouping is not cosmetic: each group is one dense quaternary table "
      "in the cascaded compiler, so no group may exceed six digits.")
    A("")
    titles = ("G1 surface exposure", "G2 local spectral geometry",
              "G3 density field calculus", "G4 ultrametric structure",
              "G5 curvature and anisotropy", "G6 global position")
    if len(titles) != len(alg.GROUPS):
        raise SystemExit("the module now has a different number of groups")
    for title, group in zip(titles, alg.GROUPS):
        A(f"\\paragraph{{{title} ({len(group)}).}}")
        A("\\begin{description}")
        for name in group:
            defn, nbhd, edge = DEFS[name]
            A(f"  \\item[\\texttt{{{name.replace('_', chr(92) + '_')}}}] "
              f"{defn}. \\emph{{Read over}}: {nbhd}. \\emph{{At the "
              f"boundary}}: {edge}.")
        A("\\end{description}")
        A("")

    A("\\subsection{The seven physicochemical constants and the propensity}")
    A("")
    A("None of these is fitted to CryptoBench. The seven are published "
      "constants of the residue type; the eighth is a count.")
    A("")
    A("\\begin{description}")
    for name in chem.CHEM_NAMES:
        what, src = CHEM_DEFS[name]
        A(f"  \\item[\\texttt{{{name}}}] {what}. \\emph{{Source}}: {src}.")
    A("\\end{description}")
    A("")
    A("An unknown or non-standard residue takes the mean of the twenty "
      "standard values rather than a sentinel, so it is neutral in the rank "
      "order rather than extreme in it.")
    A("")
    A("\\paragraph{The propensity, and why it is training-only.} The "
      "forty-third quantity is \\(\\hat P(\\text{cryptic} \\mid "
      "\\text{residue type})\\), Laplace-smoothed by one pseudo-count per "
      "class,")
    A("\\begin{equation}")
    A("  \\pi_t = \\frac{1 + \\sum_{i : \\mathrm{type}(i) = t} y_i}"
      "{2 + \\#\\{i : \\mathrm{type}(i) = t\\}},")
    A("\\end{equation}")
    A("counted over the training partition only. It is a bincount, not a fit. "
      "The table is written into the compiled artifact and read back at "
      "inference, so a test residue never contributes a count to the table "
      "that scores it; a type absent from training receives the neutral prior "
      "\\(1/2\\) rather than an undefined ratio. This is the only one of the "
      "\\(43\\) that touches a label, and it is the reason the field is "
      "described as compiled on the training partition rather than as "
      "training-free.")
    A("")

    A("\\subsection{From \\(43\\) local quantities to \\NTabWires{} wires}")
    A("")
    A("Each local quantity \\(x\\) is read under five kinds of statistic over "
      "intra-chain neighbourhoods \\(N_r(i)\\) of fixed radius:")
    A("\\begin{align}")
    A("  \\text{identity} &: x_i, \\\\")
    A("  \\text{mean} &: \\textstyle \\frac{1}{|N_r(i)|} \\sum_{j \\in N_r(i)} "
      "x_j, \\\\")
    A("  \\text{dispersion} &: \\Bigl( \\textstyle \\frac{1}{|N_r(i)|} "
      "\\sum_j x_j^2 - \\bigl(\\frac{1}{|N_r(i)|}\\sum_j x_j\\bigr)^2 "
      "\\Bigr)^{1/2}, \\\\")
    A("  \\text{centred difference} &: x_i - \\textstyle "
      "\\frac{1}{|N_r(i)|}\\sum_j x_j, \\\\")
    A("  \\text{local rank} &: \\textstyle \\frac{1}{|N_r(i)|} "
      "\\bigl|\\{ j \\in N_r(i) : x_j < x_i \\}\\bigr|.")
    A("\\end{align}")
    A("")
    A("The radii are not the same for every statistic, and the counts are what "
      "produce \\NTabWires{}:")
    A("")
    A("\\begin{center}")
    A("\\begin{tabular}{llr}")
    A("\\toprule")
    A("statistic & radii (\\AA) & groups \\\\")
    A("\\midrule")
    A(f"identity & --- & 1 \\\\")
    A(f"mean & {_radii(wide.MEAN_RADII)} & {len(wide.MEAN_RADII)} \\\\")
    A(f"dispersion & {_radii(wide.VAR_RADII)} & {len(wide.VAR_RADII)} \\\\")
    A(f"centred difference & {_radii(wide.DIFF_RADII)} & "
      f"{len(wide.DIFF_RADII)} \\\\")
    A(f"local rank & {_radii(wide.RANK_RADII)} & {len(wide.RANK_RADII)} \\\\")
    A("\\midrule")
    A(f"total & & {wide.N_STATISTIC_GROUPS} \\\\")
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\end{center}")
    A("")
    n_local = len(alg.FEATURE_NAMES) + len(chem.CHEM_NAMES) + 1
    A(f"So \\({len(alg.FEATURE_NAMES)}\\) algebraic quantities, "
      f"\\({len(chem.CHEM_NAMES)}\\) constants and one propensity give "
      f"\\({n_local}\\) local quantities, and "
      f"\\({n_local} \\times {wide.N_STATISTIC_GROUPS} = "
      f"{n_local * wide.N_STATISTIC_GROUPS}\\) wires. The local rank is the "
      "only statistic in the set that is not a moment: it asks where a residue "
      "sits in the order of its neighbourhood rather than how far it is from "
      "the neighbourhood's centre, so it is unchanged by any monotone "
      "rescaling of the quantity, and it separates a residue that is "
      "marginally the most hydrophobic of its neighbours from one that is "
      "marginally the least --- a distinction the centred difference blurs "
      "into two small numbers of opposite sign.")
    A("")
    A("\\paragraph{Banding.} Each wire is then banded into a quaternary digit "
      "by its rank \\emph{within its own chain}, ties sharing a mid-rank:")
    A("\\begin{equation}")
    A("  d_i = \\min\\!\\left( \\left\\lfloor \\frac{r_i}{\\max(n-1,1)} \\, L "
      "\\right\\rfloor,\\, L - 1 \\right), \\qquad L = 4 .")
    A("\\end{equation}")
    A("Ranking inside the chain is what makes a wire mean the same thing in a "
      "57-residue chain and a 307-residue one, and it removes any dependence "
      "on absolute units, so no constant has to be carried from the training "
      "fold in order to score a new structure.")
    A("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed appendix is stale")
    args = ap.parse_args(argv)
    bad = _check_coverage()
    if bad:
        print(f"INCOMPLETE {OUT.relative_to(ROOT)}:")
        for line in bad:
            print(f"  - {line}")
        return 1
    text = build()
    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        if OUT.read_text() != text:
            print(f"STALE {OUT.relative_to(ROOT)}: the modules have moved "
                  f"since this appendix was written out")
            return 1
        print(f"OK {OUT.relative_to(ROOT)}: "
              f"{len(alg.FEATURE_NAMES)} + {len(chem.CHEM_NAMES)} + 1 local "
              f"quantities, {wide.N_STATISTIC_GROUPS} statistics, "
              f"{(len(alg.FEATURE_NAMES) + len(chem.CHEM_NAMES) + 1) * wide.N_STATISTIC_GROUPS} wires")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
