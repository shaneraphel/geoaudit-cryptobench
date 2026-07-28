"""Generate the manuscript's numeric macros from the frozen result JSONs.

Every review of this repository has caught the same class of defect: a number in
prose that no longer matches the artifact it came from. Proof-reading does not
fix that class, because the prose and the artifact drift independently. Removing
the second copy does.

This tool emits ``paper/frozen_numbers.tex``, a file of ``\\newcommand`` macros
read straight out of ``results/cryptobench_official/BOOTSTRAP_CI.json``. The
manuscript cites ``\\AlgAuc`` and never a literal. ``make macros`` regenerates
the file and fails if it differs from the committed one, so a manuscript number
can only change when the artifact changes.

Regenerating and comparing is not sufficient on its own, which one refactor here
demonstrated: an edit that cut the emitter down to 36 of its 475 macros passed
the comparison, because the committed file had been rewritten by the same broken
code. The check that catches it reads the other end. Every macro the manuscript
cites must be defined here, so dropping a macro fails, and so does a macro name
LaTeX cannot accept -- an earlier revision emitted ``\\NLocP2``, and a control
sequence cannot contain a digit.

Usage: PYTHONPATH=src python3.12 tools/emit_frozen_numbers.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
FIELD = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD.json"
LINFIELD = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
WIDESEL = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE2.json"
WIDEPROBE = ROOT / "results/official_fold/COUNTERATTACK_WIDE_PROBE.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
CROSSVAL = ROOT / "results/architecture_sweep/REPEATED_TRAIN_SELECTION.json"
FIGPROV = ROOT / "results/official_fold/FIGURE_PROVENANCE.json"
CASES = ROOT / "results/official_fold/CASE_STUDIES.json"
QUOTIENT_SEL = ROOT / "results/architecture_sweep/COUNTERATTACK_QUOTIENT.json"
QUOTIENT_PROBE = ROOT / "results/official_fold/COUNTERATTACK_QUOTIENT_PROBE.json"
PREREG = ROOT / "results/architecture_sweep/PREREGISTERED_STATISTIC.json"
PREREG_READ = ROOT / "results/official_fold/PREREGISTERED_READ.json"
P2TRAIN = ROOT / "results/architecture_sweep/P2RANK_TRAIN_FOLD.json"

# Characters that are prose in a JSON caption and syntax in TeX. The captions
# are generated from the artifacts, so they are not free to avoid them.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\u00b1": r"$\pm$", "\u00c5": r"\AA{}",
    "\u2014": "---", "\u2013": "--",
}


def _tex(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in text)


def _roman(i: int) -> str:
    return ("One", "Two", "Three", "Four", "Five", "Six")[i - 1]
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
SELECTION = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"
CEILING = ROOT / "results/architecture_sweep/FEATURE_CEILING_DIAGNOSIS.json"
GAP = ROOT / "results/architecture_sweep/GAP_DECOMPOSITION.json"
READOUT = ROOT / "results/architecture_sweep/FINAL_READOUT_SELECTION.json"
PAIRWISE = ROOT / "results/architecture_sweep/PAIRWISE_READOUT_SELECTION.json"
SEEDPROBE = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
SEEDPROBE = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
OUT = ROOT / "paper/frozen_numbers.tex"
SRC = ROOT / "paper"

# macro stem -> (method key in the artifact)
METHODS = {
    "Tab": "table_field",
    "Alg": "algebraic_field",
    "AlgLin": "algebraic_field_linear",
    "PtwoR": "p2rank",
    "Geo": "geometric_foundation",
    "Qlut": "quaternary_lut",
    "QlutSeq": "quaternary_lut_seq",
    "Sstar": "sstar_pocket",
    "Fstar": "fstar_pocket",
    "Ultra": "ultrametric_shear_oracle",
    "Rand": "random_bbox",
}
METRICS = {"Auc": "residue_auc", "Pr": "residue_pr_auc",
           "Mcc": "residue_mcc", "FOne": "residue_f1"}


def _fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _signed(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:+.{nd}f}"


def build() -> str:
    boot = json.loads(BOOT.read_text())
    field = json.loads(FIELD.read_text())
    L: list[str] = [
        "% GENERATED FILE -- do not edit.",
        "% Regenerate with: PYTHONPATH=src python3.12 tools/emit_frozen_numbers.py",
        "% Source: " + str(BOOT.relative_to(ROOT)),
        "%         " + str(FIELD.relative_to(ROOT)),
        "",
    ]
    L.append(f"\\newcommand{{\\NUnits}}{{{boot['n_structures']}}}")
    L.append(f"\\newcommand{{\\NBoot}}{{{boot['n_boot']}}}")
    L.append(f"\\newcommand{{\\BootSeed}}{{{boot['seed']}}}")

    # Shape of the evaluation, counted from the telemetry rather than asserted:
    # a stale "1152 rows over 6 methods" outlived the run that produced it.
    rows = json.loads(TELEMETRY.read_text())["rows"]
    L.append(f"\\newcommand{{\\NRows}}{{{len(rows)}}}")
    L.append(f"\\newcommand{{\\NMethods}}{{{len({r['method'] for r in rows})}}}")

    if CEILING.exists():
        # The diagnostic that separates "the descriptors are weak" from "the
        # tabular estimator cannot spend them". Both figures are continuous
        # discriminants and are quoted only as an upper reference.
        ceil = json.loads(CEILING.read_text())
        by_label = {r["features"]: r["roc_auc"] for r in ceil["fisher"]}
        L.append(f"\\newcommand{{\\FisherSix}}"
                 f"{{{by_label['original 6 geometric']:.3f}}}")
        L.append(f"\\newcommand{{\\FisherAll}}"
                 f"{{{by_label['all 35 algebraic']:.3f}}}")

    if SELECTION.exists():
        sel = json.loads(SELECTION.read_text())
        L.append(f"\\newcommand{{\\NCandidates}}{{{len(sel['candidates'])}}}")
        L.append(f"\\newcommand{{\\NSelectFitUnits}}"
                 f"{{{sel['split']['n_fit_units']}}}")
        L.append(f"\\newcommand{{\\NSelectPickUnits}}"
                 f"{{{sel['split']['n_pick_units']}}}")
        L.append(f"\\newcommand{{\\SelectedArch}}"
                 f"{{{sel['selected']['architecture']}}}")
        L.append(f"\\newcommand{{\\SelectedArchAuc}}"
                 f"{{{sel['selected']['pick_half_roc_auc']:.4f}}}")
    if LINFIELD.exists():
        # The fitted readout. Its parameter count and its regulariser are the
        # two facts a reviewer needs in order to judge the honesty of the
        # "one closed-form solve" claim, so neither may be typed by hand.
        lin = json.loads(LINFIELD.read_text())
        L.append(f"\\newcommand{{\\NLinWires}}{{{lin['n_wires']}}}")
        L.append(f"\\newcommand{{\\LinRidge}}{{{lin['ridge']:g}}}")
        L.append(f"\\newcommand{{\\LinGateRadius}}"
                 f"{{{lin['gate']['radius_angstrom']:g}}}")
        L.append(f"\\newcommand{{\\LinGateWeight}}"
                 f"{{{lin['gate']['weight']:g}}}")
        L.append(f"\\newcommand{{\\LinOperatingQ}}"
                 f"{{{lin['operating_point']['q']:.2f}}}")

    if TABFIELD.exists():
        # The table field's shape has to come from the artifact, because every
        # one of these numbers is a claim about the circuit's size: how many
        # tables, how many cells, how full the cells are, and how much of the
        # fan-out is actually used. A hand-typed cell occupancy is exactly the
        # kind of figure that survives three architecture changes unnoticed.
        tf = json.loads(TABFIELD.read_text())
        tot = tf["cell_total"]
        occupied = [v for v in tot if v > 0]
        L.append(f"\\newcommand{{\\NTabWires}}{{{tf['n_wires']}}}")
        L.append(f"\\newcommand{{\\NTabTables}}{{{len(tf['tables'])}}}")
        L.append(f"\\newcommand{{\\TabWidth}}{{{tf['table_width']}}}")
        L.append(f"\\newcommand{{\\TabRounds}}{{{tf['partition_rounds']}}}")
        L.append(f"\\newcommand{{\\NTabCells}}{{{tf['n_cells']}}}")
        L.append(f"\\newcommand{{\\NTabCellsEmpty}}"
                 f"{{{tf['n_cells_never_addressed']}}}")
        L.append(f"\\newcommand{{\\TabCellsEmptyPct}}"
                 f"{{{100.0 * tf['n_cells_never_addressed'] / tf['n_cells']:.2f}}}")
        L.append(f"\\newcommand{{\\TabCellOccupancy}}"
                 f"{{{sum(occupied) / max(len(occupied), 1):.0f}}}")
        L.append(f"\\newcommand{{\\TabRidge}}{{{tf['ridge']:g}}}")
        L.append(f"\\newcommand{{\\TabCap}}{{{tf['fan_out_cap']}}}")
        L.append(f"\\newcommand{{\\NTabUsed}}"
                 f"{{{tf['n_tables_with_nonzero_fanout']}}}")
        L.append(f"\\newcommand{{\\TabFanOut}}{{{tf['total_fan_out']}}}")
        L.append(f"\\newcommand{{\\TabGateRadius}}"
                 f"{{{tf['gate']['radius_angstrom']:g}}}")
        L.append(f"\\newcommand{{\\TabGateWeight}}{{{tf['gate']['weight']:g}}}")
        L.append(f"\\newcommand{{\\TabOperatingQ}}"
                 f"{{{tf['operating_point']['q']:.2f}}}")

    if WIDESEL.exists():
        sel = json.loads(WIDESEL.read_text())
        L.append(f"\\newcommand{{\\TabPickAuc}}"
                 f"{{{sel['selected']['pick_half_roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\NTabCandidates}}"
                 f"{{{len(sel['candidates'])}}}")

    if CASES.exists():
        cs = json.loads(CASES.read_text())
        pop = cs["population"]
        both = pop["n_located_by_both"]
        L.append(f"\\newcommand{{\\CaseLocateF}}"
                 f"{{{cs['locate_threshold_f1']:g}}}")
        L.append(f"\\newcommand{{\\NLocBoth}}{{{both}}}")
        L.append(f"\\newcommand{{\\NLocOurs}}"
                 f"{{{pop['n_located_by_table_field'] - both}}}")
        # No digits in a command name: \NLocP2 is not a control sequence and
        # the manuscript would fail to compile on it.
        L.append(f"\\newcommand{{\\NLocPTwo}}"
                 f"{{{pop['n_located_by_p2rank'] - both}}}")
        L.append(f"\\newcommand{{\\NLocNeither}}"
                 f"{{{pop['n_located_by_neither']}}}")
        for case in cs["cases"]:
            tag = {"both_locate": "Both", "table_field_only": "Ours",
                   "p2rank_only": "Theirs",
                   "neither_locates": "Neither"}[case["case"]]
            L.append(f"\\newcommand{{\\Case{tag}}}"
                     f"{{{_tex(case['unit_id'])}}}")
            L.append(f"\\newcommand{{\\Case{tag}FOurs}}"
                     f"{{{case['f1']['table_field']:.2f}}}")
            L.append(f"\\newcommand{{\\Case{tag}FPTwo}}"
                     f"{{{case['f1']['p2rank']:.2f}}}")
            for method, mtag in (("table_field", "Ours"), ("p2rank", "PTwo")):
                g = (case.get("geometry", {}).get("methods", {})
                     .get(method) or {})
                if g.get("centroid_offset_angstrom") is not None:
                    L.append(f"\\newcommand{{\\Case{tag}Off{mtag}}}"
                             f"{{{g['centroid_offset_angstrom']:.0f}}}")
        for o in ((cs.get("burial") or {}).get("by_outcome") or []):
            tag = {"both_locate": "Both", "table_field_only": "Ours",
                   "p2rank_only": "Theirs",
                   "neither_locates": "Neither"}[o["outcome"]]
            L.append(f"\\newcommand{{\\Burial{tag}}}"
                     f"{{{o['pocket_excess_over_chain']:+.2f}}}")
        for m in ((cs.get("burial") or {}).get("calls_by_method") or []):
            if m["outcome"] == "neither_locates":
                tag = "Ours" if m["method"] == "table_field" else "PTwo"
                L.append(f"\\newcommand{{\\BurialCallsNeither{tag}}}"
                         f"{{{m['called_excess_over_chain']:+.2f}}}")

    if FIGPROV.exists():
        # The captions, so that the figure under a paragraph in the manuscript
        # and the figure under the same paragraph in the README say the same
        # sentence. They state structure counts, a seed and a standard error;
        # typed twice they would disagree within a revision, and the reader
        # would have no way to tell which copy was current.
        # Insertion order, not sorted: the generator draws them in the order
        # the README numbers them, and a caption macro numbered differently
        # from the image above it is worse than no macro.
        prov = json.loads(FIGPROV.read_text())
        for i, rec in enumerate((prov.get("figures") or {}).values(), 1):
            if rec.get("caption"):
                L.append(f"\\newcommand{{\\FigCaption{_roman(i)}}}"
                         f"{{{_tex(rec['caption'])}}}")

    if CROSSVAL.exists():
        # Whether the frozen architecture survives splits other than the one
        # that chose it. Two blocks, reported separately because they do not
        # guarantee the same thing: the four folds are CryptoBench's own, under
        # its MMseqs2 10% clustering; the 25 halves are disjoint by accession,
        # which is finer, so they resolve the ranking better and say less about
        # homology. The margin goes in beside the count on purpose -- the
        # ordering is stable, and it is stable by a few thousandths.
        xv = json.loads(CROSSVAL.read_text())
        for block, tag in (("cluster_level_cv", "Cv"), ("repeated_halves", "Rh")):
            b, v = xv[block], xv[block]["frozen_choice"]
            n = b.get("n_folds") or b.get("n_repeats")
            L.append(f"\\newcommand{{\\NSplits{tag}}}{{{n}}}")
            L.append(f"\\newcommand{{\\NFirst{tag}}}{{{v['n_first']}}}")
            L.append(f"\\newcommand{{\\WorstRank{tag}}}{{{v['worst_rank']}}}")
            L.append(f"\\newcommand{{\\MeanAuc{tag}}}"
                     f"{{{v['mean_roc_auc']:.4f}}}")
            L.append(f"\\newcommand{{\\SdAuc{tag}}}{{{v['sd_roc_auc']:.4f}}}")
            L.append(f"\\newcommand{{\\MeanMargin{tag}}}"
                     f"{{{v['mean_margin_over_runner_up']:+.4f}}}")
            L.append(f"\\newcommand{{\\WorstMargin{tag}}}"
                     f"{{{v['worst_margin_over_runner_up']:+.4f}}}")
        runner = next(r["architecture"]
                      for r in xv["cluster_level_cv"]["per_architecture"]
                      if r["architecture"] != xv["frozen_choice"]["architecture"])
        L.append(f"\\newcommand{{\\RunnerUpArch}}{{{runner}}}")

    if PREREG.exists() and PREREG_READ.exists():
        # The functional was chosen on the training partition and the fold read
        # under it afterwards. Both halves are emitted from their own artifacts,
        # and the commit that fixed the choice is a macro too, because the
        # ordering is the claim and a reader has to be able to check it.
        pg = json.loads(PREREG.read_text())
        rd = json.loads(PREREG_READ.read_text())
        cmp_ = pg["comparison"]
        cand = {c["statistic"]: c for c in pg["candidates"]}
        L.append(f"\\newcommand{{\\PreRegNPick}}{{{cmp_['n_paired_units']}}}")
        L.append(f"\\newcommand{{\\PreRegFieldPick}}"
                 f"{{{cmp_['mean_field']:.4f}}}")
        L.append(f"\\newcommand{{\\PreRegPTwoPick}}"
                 f"{{{cmp_['mean_p2rank']:.4f}}}")
        L.append(f"\\newcommand{{\\PreRegMeanPowerQuarter}}"
                 f"{{{cand['mean']['power_by_effect_shrink']['0.25']:.2f}}}")
        L.append(f"\\newcommand{{\\PreRegTrimPowerQuarter}}"
                 f"{{{cand['trimmed20']['power_by_effect_shrink']['0.25']:.2f}}}")
        L.append(f"\\newcommand{{\\PreRegStratPowerQuarter}}"
                 f"{{{cand['stratified_by_length']['power_by_effect_shrink']['0.25']:.2f}}}")
        L.append(f"\\newcommand{{\\PreRegChosen}}"
                 f"{{{pg['preregistered']['statistic'].replace('_', ' ')}}}")
        L.append(f"\\newcommand{{\\PreRegForecast}}"
                 f"{{{pg['forecast']['expected_power'] * 100:.0f}}}")
        L.append(f"\\newcommand{{\\PreRegCommit}}"
                 f"{{\\texttt{{{rd['provenance_of_the_choice']['committed_in'][:12]}}}}}")
        L.append(f"\\newcommand{{\\PreRegReadIndex}}"
                 f"{{{rd['test_fold_read_index']}}}")
        res = rd["preregistered_result"]
        L.append(f"\\newcommand{{\\PreRegTestDelta}}{{{res['point']:+.4f}}}")
        L.append(f"\\newcommand{{\\PreRegTestCI}}"
                 f"{{[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]}}")
        L.append(f"\\newcommand{{\\PreRegTestP}}"
                 f"{{{res['p_two_sided_bootstrap']:.3f}}}")
        for stem, key in (("Median", "median"), ("WinRate", "win_rate")):
            c = next(x for x in rd["candidates"] if x["statistic"] == key)
            L.append(f"\\newcommand{{\\PreRegTest{stem}}}{{{c['point']:+.4f}}}")
            L.append(f"\\newcommand{{\\PreRegTest{stem}P}}"
                     f"{{{c['p_two_sided_bootstrap']:.3f}}}")
        sh = rd["shape_of_the_differences"]
        L.append(f"\\newcommand{{\\PreRegNAhead}}{{{sh['n_field_ahead']}}}")
        L.append(f"\\newcommand{{\\PreRegNBehind}}{{{sh['n_baseline_ahead']}}}")
        L.append(f"\\newcommand{{\\PreRegNTrimmed}}"
                 f"{{{sh['n_trimmed_each_side']}}}")
        L.append(f"\\newcommand{{\\PreRegTailLoss}}"
                 f"{{{sh['worst_losses_mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\PreRegTailWin}}"
                 f"{{{sh['best_wins_mean']:+.4f}}}")
    if P2TRAIN.exists():
        p2t = json.loads(P2TRAIN.read_text())
        L.append(f"\\newcommand{{\\PTwoTrainAuc}}"
                 f"{{{p2t['residue_auc_mean']:.4f}}}")
        L.append(f"\\newcommand{{\\PTwoTrainUnits}}{{{p2t['n_ok']}}}")

    if QUOTIENT_SEL.exists() and QUOTIENT_PROBE.exists():
        # The quotient counterattack: a construction that beat the dense bank on
        # every training split and then did not beat it here. Both halves are
        # emitted from their own artifacts so the chapter cannot state a
        # training gain the selection did not measure or a test result the probe
        # did not produce.
        qs = json.loads(QUOTIENT_SEL.read_text())
        qp = json.loads(QUOTIENT_PROBE.read_text())
        cap, sel = qs["capacity"], qs["selected"]
        L.append(f"\\newcommand{{\\NTrainPositives}}"
                 f"{{{cap['n_train_positives']:,}}}")
        L.append(f"\\newcommand{{\\QuoDenseBound}}"
                 f"{{{cap['dense_width_bound_L4']:.2f}}}")
        L.append(f"\\newcommand{{\\QuoDenseWidth}}"
                 f"{{{cap['dense_widest_admissible_L4']}}}")
        L.append(f"\\newcommand{{\\QuoSymWidth}}"
                 f"{{{cap['quotient_widest_admissible_L4']}}}")
        L.append(f"\\newcommand{{\\QuoDenseCells}}{{{cap['dense_cells_d6_L4']:,}}}")
        L.append(f"\\newcommand{{\\QuoSymCells}}{{{cap['quotient_cells_d6_L4']}}}")
        L.append(f"\\newcommand{{\\QuoSymCellsEight}}"
                 f"{{{cap['quotient_cells_d6_L8']:,}}}")
        L.append(f"\\newcommand{{\\QuoSymCellsAll}}"
                 f"{{{cap['quotient_cells_d35_L4']:,}}}")
        L.append(f"\\newcommand{{\\QuoNSplits}}{{{qs['n_splits']}}}")
        L.append(f"\\newcommand{{\\QuoNCandidates}}{{{qs['n_candidates']}}}")
        L.append(f"\\newcommand{{\\QuoTrainGain}}"
                 f"{{{sel['mean_delta_vs_control']:+.4f}}}")
        L.append(f"\\newcommand{{\\QuoTrainWorst}}"
                 f"{{{sel['worst_delta_vs_control']:+.4f}}}")
        L.append(f"\\newcommand{{\\QuoNBeating}}"
                 f"{{{sel['n_splits_beating_control']}}}")
        L.append(f"\\newcommand{{\\QuoOneSplitGain}}"
                 f"{{{qs['selection_honesty']['delta_on_the_split_it_was_found_on']:+.4f}}}")
        worst = min(qs["summary"], key=lambda r: r["mean_delta_vs_control"])
        L.append(f"\\newcommand{{\\QuoWorstArchDelta}}"
                 f"{{{worst['mean_delta_vs_control']:+.4f}}}")
        L.append(f"\\newcommand{{\\QuoReadIndex}}"
                 f"{{{qp['test_fold_read_index']}}}")
        L.append(f"\\newcommand{{\\QuoTestAuc}}{{{qp['residue_auc_mean']:.4f}}}")
        L.append(f"\\newcommand{{\\QuoNTables}}"
                 f"{{{qp['architecture']['n_tables']}}}")
        L.append(f"\\newcommand{{\\QuoControlRepro}}"
                 f"{{{qp['reproduction_check']['residue_auc_mean']:.4f}}}")
        for method, tag in (("algebraic_field", "Af"), ("p2rank", "PTwo")):
            d = qp["paired_vs"][method]["residue_auc"]
            L.append(f"\\newcommand{{\\QuoD{tag}}}"
                     f"{{{d['paired_difference']:+.4f}}}")
            L.append(f"\\newcommand{{\\QuoD{tag}CI}}"
                     f"{{[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]}}")
            L.append(f"\\newcommand{{\\QuoD{tag}P}}{{{d['p_two_sided']:.4f}}}")

    if GAP.exists():
        # The manuscript attributes the counting field's deficit to capacity.
        # This artifact is what checked that attribution, and it is the reason
        # the attribution changed, so every number in that argument is read
        # from it rather than typed.
        gp = json.loads(GAP.read_text())
        dec, cells = gp["decomposition"], gp["decomposition"]["cells"]
        L.append(f"\\newcommand{{\\GapTablesSmall}}"
                 f"{{{cells['tables_35']['roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GapLinearSmall}}"
                 f"{{{cells['linear_35']['roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GapLinearWide}}"
                 f"{{{cells['linear_172']['roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GapTablesWide}}"
                 f"{{{cells['tables_172']['roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GapWideWires}}"
                 f"{{{cells['linear_172']['n_wires']}}}")
        L.append(f"\\newcommand{{\\GapExtraWires}}"
                 f"{{{cells['linear_172']['n_wires'] - cells['tables_35']['n_wires']}}}")
        L.append(f"\\newcommand{{\\GapWideTables}}"
                 f"{{{cells['tables_172']['n_tables']}}}")
        L.append(f"\\newcommand{{\\GapReadout}}{{{dec['readout_effect']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapInput}}{{{dec['input_effect']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapTotal}}{{{dec['total']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapReadoutShare}}"
                 f"{{{100 * dec['readout_share']:.0f}}}")
        L.append(f"\\newcommand{{\\GapInputShare}}"
                 f"{{{100 * dec['input_share']:.0f}}}")
        fan = gp["fanout_price"]["banks"]
        L.append(f"\\newcommand{{\\GapFanoutPrice}}"
                 f"{{{fan['dense_L4']['price']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapFanoutSolved}}"
                 f"{{{fan['dense_L4']['solved_integer_fanout']:.4f}}}")
        L.append(f"\\newcommand{{\\GapFanoutQuo}}"
                 f"{{{fan['quotient_L864']['price']:+.4f}}}")
        cap = gp["capacity_probes"]
        sc = cap["compile_scaling"]["rows"]
        L.append(f"\\newcommand{{\\GapScaleSmallPos}}{{{sc[0]['n_fit_positives']}}}")
        L.append(f"\\newcommand{{\\GapScaleSmallGain}}{{{sc[0]['gain']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapScaleBigPos}}{{{sc[-1]['n_fit_positives']}}}")
        L.append(f"\\newcommand{{\\GapScaleBigGain}}{{{sc[-1]['gain']:+.4f}}}")
        mg = max(cap["marginal_tables"]["rows"], key=lambda r: r["roc_auc"])
        L.append(f"\\newcommand{{\\GapMarginalLevels}}{{{mg['n_levels']}}}")
        L.append(f"\\newcommand{{\\GapMarginalAuc}}{{{mg['roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GapMarginalTables}}{{{mg['n_tables']}}}")
        L.append(f"\\newcommand{{\\GapUnseenPct}}"
                 f"{{{100 * cap['unseen_cells']['dense_L4_fraction']:.2f}}}")
        dif = gp["difficulty"]
        hard_tr = dif["train_pick_half"]["bins"][0]
        hard_te = dif["test_fold"]["bins"][0]
        L.append(f"\\newcommand{{\\GapHardCut}}{{{hard_tr['hi']:.2f}}}")
        L.append(f"\\newcommand{{\\GapHardTrainN}}{{{hard_tr['n']}}}")
        L.append(f"\\newcommand{{\\GapHardTrainGain}}{{{hard_tr['mean_gain']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapHardTestN}}{{{hard_te['n']}}}")
        L.append(f"\\newcommand{{\\GapHardTestGain}}{{{hard_te['mean_gain']:+.4f}}}")
        easy_tr = dif["train_pick_half"]["bins"][-2]
        easy_te = dif["test_fold"]["bins"][-2]
        L.append(f"\\newcommand{{\\GapEasyTrainGain}}{{{easy_tr['mean_gain']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapEasyTestGain}}{{{easy_te['mean_gain']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapReweighted}}"
                 f"{{{dif['reweighted_test_gain']:+.4f}}}")
        L.append(f"\\newcommand{{\\GapPickUnits}}{{{gp['split']['n_pick_units']}}}")

    if LEDGER.exists():
        # How often the held-out fold has been scored, and by how many
        # architectures. Hand-counting this is how the manuscript came to say
        # three when the artifacts say eleven, so it is read from the ledger.
        led = json.loads(LEDGER.read_text())
        L.append(f"\\newcommand{{\\NArchOnFold}}"
                 f"{{{led['n_distinct_architectures_evaluated']}}}")
        L.append(f"\\newcommand{{\\NOurDetectors}}{{{led['n_our_detectors']}}}")
        L.append(f"\\newcommand{{\\NStandaloneProbes}}"
                 f"{{{led['n_standalone_probes']}}}")

        # The selection arithmetic a hostile reviewer performs first, done here
        # instead. The standard error of a mean over the fold sets the scale
        # that any margin has to be read against.
        rows = json.loads(TELEMETRY.read_text())["rows"]
        per = [r["residue_auc"] for r in rows
               if r["method"] == "table_field" and r.get("residue_auc") is not None]
        if len(per) > 1:
            mu = sum(per) / len(per)
            var = sum((x - mu) ** 2 for x in per) / (len(per) - 1)
            se = (var / len(per)) ** 0.5
            L.append(f"\\newcommand{{\\FoldSE}}{{{se:.4f}}}")
            d = boot["metrics"]["residue_auc"]["paired_vs_baseline"]
            tab = (d.get("table_field") or {}).get("delta_point")
            if tab:
                L.append(f"\\newcommand{{\\LeadInSE}}{{{tab / se:.2f}}}")

    if WIDEPROBE.exists():
        # Three readings of the test fold, and the manuscript reports all
        # three. Their values are macros so that the disclosure cannot drift
        # apart from the artifacts that recorded them.
        pr = json.loads(WIDEPROBE.read_text())
        for e in pr["earlier_reads"]:
            L.append(f"\\newcommand{{\\ReadAuc{'I' * e['index']}}}"
                     f"{{{e['residue_auc_mean']:.4f}}}")
            L.append(f"\\newcommand{{\\ReadDelta{'I' * e['index']}}}"
                     f"{{{e['paired_vs_p2rank']:+.4f}}}")
        L.append(f"\\newcommand{{\\NTestReads}}"
                 f"{{{pr['test_fold_read_index']}}}")

    if SEEDPROBE.exists():
        # How often each verdict survives a change of resampling seed. The MCC
        # interval excludes zero at the pre-registered seed by four
        # ten-thousandths, so without these counts the manuscript would be
        # quoting one pseudo-random sequence as a result.
        sp = json.loads(SEEDPROBE.read_text())
        L.append(f"\\newcommand{{\\NSeedProbes}}{{{sp['n_seeds']}}}")
        for stem, key in (("TabAuc", "residue_auc"), ("TabPr", "residue_pr_auc"),
                          ("TabMcc", "residue_mcc"), ("TabFOne", "residue_f1")):
            blk = sp["metrics"].get(key)
            if blk:
                L.append(f"\\newcommand{{\\{stem}SeedSig}}"
                         f"{{{blk['n_seeds_excluding_zero']}}}")

    if READOUT.exists():
        # Readout comparison on the training split. These are the numbers that
        # justify choosing a first-order functional over the table bank, and
        # they must be quoted from the selection artifact, not from memory.
        ro = {c["readout"]: c for c in json.loads(READOUT.read_text())["candidates"]}
        for stem, key in (("Count", "count_square"), ("LinDigits", "lin_digits"),
                          ("LinOnehot", "lin_onehot"), ("LinTables", "lin_tables")):
            if key in ro:
                L.append(f"\\newcommand{{\\Pick{stem}}}"
                         f"{{{ro[key]['pick_half_roc_auc']:.4f}}}")

    if PAIRWISE.exists():
        pw = json.loads(PAIRWISE.read_text())
        best = max(pw["candidates"], key=lambda c: c["pick_half_roc_auc"])
        L.append(f"\\newcommand{{\\PickPairBest}}"
                 f"{{{best['pick_half_roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\PickPairBestName}}"
                 f"{{\\texttt{{{best['readout'].replace('_', r'\_')}}}}}")

    tr = field["train"]
    L.append(f"\\newcommand{{\\NTrainUnits}}{{{tr['n_units']}}}")
    L.append(f"\\newcommand{{\\NTrainResidues}}{{{tr['n_residues']}}}")
    L.append(f"\\newcommand{{\\TrainRate}}{{{tr['base_rate']:.4f}}}")
    L.append(f"\\newcommand{{\\NAlgFeatures}}{{{field['n_features']}}}")
    L.append(f"\\newcommand{{\\NAlgTables}}{{{len(field['tables'])}}}")
    L.append(f"\\newcommand{{\\AlgOperatingQ}}"
             f"{{{field['operating_point']['q']:.2f}}}")
    led = field.get("cluster_ledger", {})
    for k, v in sorted(led.items()):
        stem = "".join(p.capitalize() for p in k.split("_"))
        L.append(f"\\newcommand{{\\Ledger{stem}}}{{{v}}}")
    L.append("")

    for msuf, mkey in METRICS.items():
        blk = boot["metrics"][mkey]
        for stem, name in METHODS.items():
            pm = blk["per_method"].get(name)
            if pm is None:
                continue
            L.append(f"\\newcommand{{\\{stem}{msuf}}}{{{_fmt(pm['point'])}}}")
            L.append(f"\\newcommand{{\\{stem}{msuf}CI}}"
                     f"{{[{_fmt(pm['ci_low'])}, {_fmt(pm['ci_high'])}]}}")
        for stem, name in METHODS.items():
            d = (blk.get("paired_vs_baseline") or {}).get(name)
            if d is None:
                continue
            # Four decimals for the differences and their bounds, not
            # three: the MCC interval's lower bound is +0.0004, and rounding
            # that to +0.000 prints a bound touching zero in a row the same
            # table marks as resolved.
            L.append(f"\\newcommand{{\\{stem}{msuf}D}}"
                     f"{{{_signed(d['delta_point'], 4)}}}")
            L.append(f"\\newcommand{{\\{stem}{msuf}DCI}}"
                     f"{{[{_signed(d['delta_ci_low'], 4)}, "
                     f"{_signed(d['delta_ci_high'], 4)}]}}")
            p = d["p_two_sided_bootstrap"]
            L.append(f"\\newcommand{{\\{stem}{msuf}P}}"
                     f"{{{'n/a' if p is None else (f'{p:.3f}' if p >= 0.001 else '<0.001')}}}")
            # The structures the difference is actually defined on, and the two
            # matched means. Where this is below the fold size, the per-method
            # points above are over different sets and subtracting them would
            # be wrong; the manuscript quotes these instead.
            L.append(f"\\newcommand{{\\{stem}{msuf}NPaired}}"
                     f"{{{d.get('n_paired_structures', '')}}}")
            L.append(f"\\newcommand{{\\{stem}{msuf}Matched}}"
                     f"{{{_fmt(d.get('matched_point_method'))}}}")
            L.append(f"\\newcommand{{\\{stem}{msuf}MatchedBase}}"
                     f"{{{_fmt(d.get('matched_point_baseline'))}}}")
            cz = d.get("crosses_zero")
            L.append(f"\\newcommand{{\\{stem}{msuf}Verdict}}"
                     f"{{{'n/a' if cz is None else ('indistinguishable' if cz else 'separated')}}}")
            # Table-width variant of the same fact. Spelling "indistinguishable"
            # four times across a row overflows the text block.
            L.append(f"\\newcommand{{\\{stem}{msuf}NS}}"
                     f"{{{'' if cz is None else ('~ns' if cz else '~\\textbf{sig}')}}}")
        L.append("")
    return "\n".join(L) + "\n"


# \AA is TeX's own; every other \Name{} in the manuscript is ours to define.
_BUILTIN = {"AA"}


def _undefined(text: str) -> list[str]:
    """Macros the manuscript cites that this file does not define."""
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))
    out = []
    for tex in sorted(SRC.glob("*.tex")):
        if tex.name == OUT.name:
            continue
        cited = set(re.findall(r"\\([A-Z][A-Za-z]*)\{\}", tex.read_text()))
        for name in sorted(cited - defined - _BUILTIN):
            out.append(f"{tex.name} cites \\{name}{{}}, which is never defined")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed file is stale")
    args = ap.parse_args(argv)
    text = build()
    missing = _undefined(text)
    if missing:
        print(f"INCOMPLETE {OUT.relative_to(ROOT)}:")
        for line in missing:
            print(f"  - {line}")
        return 1
    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        if OUT.read_text() != text:
            print(f"STALE {OUT.relative_to(ROOT)}: manuscript numbers no longer "
                  f"match the frozen artifacts")
            return 1
        print(f"OK {OUT.relative_to(ROOT)} matches the frozen artifacts")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} "
          f"({text.count('newcommand')} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
