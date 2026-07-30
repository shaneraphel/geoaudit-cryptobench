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


_DIGIT_WORD = ("Zero", "One", "Two", "Three", "Four",
               "Five", "Six", "Seven", "Eight", "Nine")


def _caption_macro(filename: str) -> str:
    """``fig_case_studies.png`` -> ``FigCapCaseStudies``.

    Digits become words because a TeX macro name is letters only, and the word
    is the one the rest of this file already uses: ``p2rank`` reads ``PTwoRank``,
    matching \\PTwoRuntime and friends.
    """
    stem = filename.rsplit(".", 1)[0]
    if stem.startswith("fig_"):
        stem = stem[4:]
    out = []
    for part in stem.split("_"):
        for tok in re.findall(r"\d|[A-Za-z]+", part):
            out.append(_DIGIT_WORD[int(tok)] if tok.isdigit()
                       else tok.capitalize())
    return "FigCap" + "".join(out)
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
SELECTION = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"
CEILING = ROOT / "results/architecture_sweep/FEATURE_CEILING_DIAGNOSIS.json"
GAP = ROOT / "results/architecture_sweep/GAP_DECOMPOSITION.json"
READOUT = ROOT / "results/architecture_sweep/FINAL_READOUT_SELECTION.json"
PAIRWISE = ROOT / "results/architecture_sweep/PAIRWISE_READOUT_SELECTION.json"
SEEDPROBE = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
BANKS_CEILING = ROOT / "results/architecture_sweep/OPERATOR_BANK_CEILING.json"
WIDE3 = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3.json"
WIDE3_CONTROL = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE3_CONTROL.json"
SENS = ROOT / "results/architecture_sweep/SENSITIVITY_SWEEP.json"
P2TRAIN_OP = ROOT / "results/architecture_sweep/P2RANK_TRAIN_OPERATING_POINT.json"
MATCH_PREREG = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_OPERATING_POINT.json"
MATCH_READ = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"
FULL_READ = ROOT / "results/official_fold/MATCHED_FULL_READ.json"
AUDIT = ROOT / "results/official_fold/AUDIT_DECOMPOSITION.json"
COST = ROOT / "results/architecture_sweep/RUNTIME_COST.json"
INTERP = ROOT / "results/architecture_sweep/INTERPRETABLE_BASELINES.json"
ENDPOINT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
SUBGROUP = ROOT / "results/official_fold/SUBGROUP_READ.json"
POCKET = ROOT / "results/official_fold/POCKET_READ.json"
PLMNN = ROOT / "results/official_fold/PLMNN_READ.json"
PMREAD = ROOT / "results/official_fold/POCKETMINER_READ.json"
EXTREAD = ROOT / "results/external/EXTERNAL_READ.json"
EXTSET = ROOT / "results/external/EXTERNAL_SET.json"
PMSELF = ROOT / "results/baselines/POCKETMINER_SELFTEST.json"
CURVE = ROOT / "results/official_fold/THRESHOLD_CURVE.json"
PLMNN_SCORES = ROOT / "results/baselines/PLMNN_SCORES.json"
TRAIN_OP = ROOT / "results/architecture_sweep/TRAIN_OPERATING_POINTS.json"
# The five training-fold sweeps that measured the construction's own parameters.
# Grouped because the manuscript reports them as one finding: each varies a
# choice applied identically to all tables, and the ensemble is indifferent to
# every one of them except the quantisation.
LADDER = ROOT / "results/architecture_sweep/QUANTISATION_LADDER.json"
PAIRSEL = ROOT / "results/architecture_sweep/SELECTED_PAIRINGS.json"
COMPWIRE = ROOT / "results/architecture_sweep/COMPOSITION_WIRES.json"
GRAMCOND = ROOT / "results/architecture_sweep/GRAM_CONDITIONING.json"
TRUNC = ROOT / "results/architecture_sweep/BANK_TRUNCATION.json"
HIER = ROOT / "results/architecture_sweep/HIERARCHICAL_MULTIPLICITIES.json"
GATEROUTE = ROOT / "results/architecture_sweep/GATE_WEIGHT_ROUTING.json"
TABWIDTH = ROOT / "results/architecture_sweep/TABLE_WIDTH.json"
COMBMULT = ROOT / "results/architecture_sweep/COMBINATORIAL_MULTIPLICITIES.json"
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
        # How often a wire is actually in a table. 645 is odd, so a partition
        # into pairs leaves one wire over per round and the chapter used to say
        # every wire appeared exactly TabRounds times. It does not, and the
        # counts are emitted so the corrected sentence is checkable.
        import collections as _c

        _per_wire = _c.Counter(w for t in tf["tables"] for w in t)
        _counts = _c.Counter(_per_wire.values())
        _rounds = tf["partition_rounds"]
        if len(_per_wire) != tf["n_wires"]:
            raise SystemExit(
                f"only {len(_per_wire)} of {tf['n_wires']} wires appear in the "
                f"bank; a wire in no table is scored by nothing and the "
                f"appendix's coverage claim would be wrong")
        if set(_counts) - {_rounds, _rounds - 1}:
            raise SystemExit(
                f"wires now appear {sorted(_counts)} times; the chapter says "
                f"every wire is in {_rounds} tables or one fewer, which is what "
                f"an odd wire count under a pairing gives, and must be rewritten")
        L.append(f"\\newcommand{{\\NWireFull}}{{{_counts.get(_rounds, 0)}}}")
        L.append(f"\\newcommand{{\\NWireShort}}{{{_counts.get(_rounds - 1, 0)}}}")
        # Two different fits of the same configuration produce two different
        # non-zero table counts, and this repository reported both without
        # saying so: 4797 for the shipped compile over all 770 training units,
        # 4853 for the selection fit that used one cluster-disjoint half. A
        # reviewer reading both files sees a contradiction, so the fit set is
        # now part of the macro name and neither can be quoted by accident.
        L.append(f"\\newcommand{{\\NTabUsedFullFold}}"
                 f"{{{tf['n_tables_with_nonzero_fanout']}}}")
        L.append(f"\\newcommand{{\\TabFanOut}}{{{tf['total_fan_out']}}}")
        L.append(f"\\newcommand{{\\NTabTrainRes}}"
                 f"{{{tf['train']['n_residues']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\TabGateRadius}}"
                 f"{{{tf['gate']['radius_angstrom']:g}}}")
        L.append(f"\\newcommand{{\\TabGateWeight}}{{{tf['gate']['weight']:g}}}")
        L.append(f"\\newcommand{{\\TabOperatingQ}}"
                 f"{{{tf['operating_point']['q']:.2f}}}")
        # What the method costs to compile, to ship and to run. The median
        # rather than the mean, because one 900-residue chain would otherwise
        # decide the number.
        L.append(f"\\newcommand{{\\TabCompileSeconds}}"
                 f"{{{tf['compile_seconds']:.0f}}}")
        L.append(f"\\newcommand{{\\TabArtifactMB}}"
                 f"{{{TABFIELD.stat().st_size / 1e6:.2f}}}")
        if TELEMETRY.exists():
            trows = json.loads(TELEMETRY.read_text())["rows"]

            def _median_runtime(method: str) -> float | None:
                v = sorted(r["runtime_s"] for r in trows
                           if r["method"] == method
                           and r.get("runtime_s") is not None)
                if not v:
                    return None
                m = len(v) // 2
                return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])

            ours, theirs = _median_runtime("table_field"), _median_runtime("p2rank")
            if ours:
                L.append(f"\\newcommand{{\\TabRuntime}}{{{ours:.3f}}}")
            if theirs:
                L.append(f"\\newcommand{{\\PTwoRuntime}}{{{theirs:.3f}}}")
            if ours and theirs:
                L.append(f"\\newcommand{{\\TabSpeedup}}{{{theirs / ours:.1f}}}")

    if WIDESEL.exists():
        sel = json.loads(WIDESEL.read_text())
        L.append(f"\\newcommand{{\\TabPickAuc}}"
                 f"{{{sel['selected']['pick_half_roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\NTabCandidates}}"
                 f"{{{len(sel['candidates'])}}}")
        # The ridge curve of the shipped basis alone. The candidate list also
        # holds two bases that were not shipped, and averaging over them would
        # describe a model the paper does not contain.
        #
        # These are emitted from the sensitivity sweep rather than from here, so
        # that the one table a reader reads the sensitivity from contains all of
        # it. This artifact is kept as the cross-check: it measured the same
        # curve on the same half through a different digitiser, and if the two
        # ever disagree the sweep's re-implementation has drifted.
        chosen = sel["selected"]["basis"]
        curve = {c["ridge"]: c["pick_half_roc_auc"]
                 for c in sel["candidates"] if c["basis"] == chosen}
        ridge_words = {0.03: "Low", 0.1: "Mid", 0.3: "High"}
        sweep_curve = {}
        if SENS.exists():
            sw_rows = json.loads(SENS.read_text())["rows"]
            fz = json.loads(SENS.read_text())["frozen_configuration"]
            sweep_curve = {
                r["ridge"]: r["pick_half_roc_auc"] for r in sw_rows
                if (r["levels"], r["ranking"], r["cap"])
                == (fz["levels"], fz["ranking"], fz["cap"])}
        for r, word in ridge_words.items():
            value = sweep_curve.get(r, curve.get(r))
            if value is None:
                continue
            if r in curve and r in sweep_curve and abs(curve[r] - sweep_curve[r]) > 5e-5:
                raise SystemExit(
                    f"at ridge {r} the selection artifact reports "
                    f"{curve[r]:.6f} and the sensitivity sweep reports "
                    f"{sweep_curve[r]:.6f}. They measure the same fit on the "
                    f"same half through two digitisers and must agree")
            L.append(f"\\newcommand{{\\TabRidge{word}}}{{{value:.4f}}}")
            L.append(f"\\newcommand{{\\TabRidge{word}Value}}{{{r:g}}}")
        L.append(f"\\newcommand{{\\TabRidgeFromSweep}}"
                 f"{{{'yes' if len(sweep_curve) == len(ridge_words) else 'no'}}}")

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
        # Named after the image, not numbered by draw order. They used to be
        # numbered, and inserting a figure ahead of the case studies silently
        # moved that caption onto a different plot: the manuscript still said
        # FigCaptionThree while position three had become another figure. A name
        # derived from the filename cannot slide, and _figure_captions_match
        # below refuses a manuscript that pairs one with the wrong image.
        prov = json.loads(FIGPROV.read_text())
        for name, rec in (prov.get("figures") or {}).items():
            if rec.get("caption"):
                L.append(f"\\newcommand{{\\{_caption_macro(name)}}}"
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

    if P2TRAIN_OP.exists():
        # What P2Rank's threshold would have been, had it been tuned the way
        # ours was. Emitted separately from the read so that the plan's numbers
        # and the fold's numbers cannot be conflated in the manuscript.
        op = json.loads(P2TRAIN_OP.read_text())
        L.append(f"\\newcommand{{\\MatchPTwoQ}}{{{op['p2rank_selected_q']:.2f}}}")
        L.append(f"\\newcommand{{\\MatchPTwoTrainNative}}"
                 f"{{{op['p2rank_pooled_train_f1_at_native_call']:.4f}}}")
        L.append(f"\\newcommand{{\\MatchPTwoTrainTuned}}"
                 f"{{{op['p2rank_pooled_train_f1_at_selected_q']:.4f}}}")
        L.append(f"\\newcommand{{\\MatchTrainGain}}"
                 f"{{{op['tuning_is_worth_to_p2rank']:+.4f}}}")

    if MATCH_PREREG.exists() and MATCH_READ.exists():
        pg = json.loads(MATCH_PREREG.read_text())
        rd = json.loads(MATCH_READ.read_text())
        L.append(f"\\newcommand{{\\MatchForecast}}"
                 f"{{{pg['forecast']['expected_matched_delta_if_the_gain_transfers']:+.4f}}}")
        L.append(f"\\newcommand{{\\MatchCommit}}"
                 f"{{\\texttt{{{rd['ordering']['preregistration_commit'][:12]}}}}}")
        L.append(f"\\newcommand{{\\MatchReadIndex}}"
                 f"{{{rd['test_fold_read_index']}}}")
        L.append(f"\\newcommand{{\\MatchNUnits}}{{{rd['n_units']}}}")
        L.append(f"\\newcommand{{\\MatchOurQ}}"
                 f"{{{rd['matched']['A_common_q']['q_ours']:.2f}}}")
        nat = rd["native_call_reference"]
        L.append(f"\\newcommand{{\\MatchNativeOurs}}"
                 f"{{{nat['table_field_f1']:.4f}}}")
        L.append(f"\\newcommand{{\\MatchNativePTwo}}"
                 f"{{{nat['p2rank_f1']:.4f}}}")
        L.append(f"\\newcommand{{\\MatchNativeDelta}}"
                 f"{{{nat['published_delta']:+.4f}}}")
        for stem, rule in (("A", "A_common_q"), ("B", "B_each_tuned_on_train")):
            r = rd["matched"][rule]
            p = r["primary"]
            L.append(f"\\newcommand{{\\Match{stem}Ours}}"
                     f"{{{r['table_field_f1']:.4f}}}")
            L.append(f"\\newcommand{{\\Match{stem}PTwo}}"
                     f"{{{r['p2rank_f1']:.4f}}}")
            L.append(f"\\newcommand{{\\Match{stem}Delta}}"
                     f"{{{p['delta_point']:+.4f}}}")
            L.append(f"\\newcommand{{\\Match{stem}CI}}"
                     f"{{[{p['delta_ci_low']:+.4f}, {p['delta_ci_high']:+.4f}]}}")
            L.append(f"\\newcommand{{\\Match{stem}P}}"
                     f"{{{p['p_two_sided_bootstrap']:.3f}}}")
            t = r["secondary_trimmed_mean"]
            L.append(f"\\newcommand{{\\Match{stem}Trim}}"
                     f"{{{t['delta_point']:+.4f}}}")
            L.append(f"\\newcommand{{\\Match{stem}TrimCI}}"
                     f"{{[{t['delta_ci_low']:+.4f}, "
                     f"{t['delta_ci_high']:+.4f}]}}")
        o = rd["oracle"]
        g = o["our_shipped_q_vs_p2rank_oracle_q"]
        L.append(f"\\newcommand{{\\MatchOracleQ}}"
                 f"{{{o['p2rank_best_q_on_the_held_out_fold']:.2f}}}")
        L.append(f"\\newcommand{{\\MatchOraclePTwo}}"
                 f"{{{o['p2rank_f1_at_its_oracle_q']:.4f}}}")
        L.append(f"\\newcommand{{\\MatchOracleDelta}}"
                 f"{{{g['delta_point']:+.4f}}}")
        L.append(f"\\newcommand{{\\MatchOracleCI}}"
                 f"{{[{g['delta_ci_low']:+.4f}, {g['delta_ci_high']:+.4f}]}}")
        L.append(f"\\newcommand{{\\MatchOurOracleQ}}"
                 f"{{{o['table_field_best_q_on_the_held_out_fold']:.2f}}}")
        L.append(f"\\newcommand{{\\MatchOurOracleGap}}"
                 f"{{{o['our_shipped_q_is_within_of_our_oracle']:.4f}}}")

    if FULL_READ.exists() and TRAIN_OP.exists():
        # The seventh read. Four metrics on four conventions under three
        # resampling units is 48 intervals, and a chapter that quoted them by
        # hand would eventually quote one from the wrong cell. Every number the
        # chapter uses is named for its convention, its metric and its
        # resampling unit, so a mispaired citation is a missing macro rather
        # than a plausible wrong figure.
        fr = json.loads(FULL_READ.read_text())
        top = json.loads(TRAIN_OP.read_text())
        metric_stem = {"precision": "Prec", "recall": "Rec",
                       "positive_class_f1": "FOne", "mcc": "MCC"}
        conv_stem = {"D1_as_deployed": "Deployed",
                     "D2_common_budget": "Budget",
                     "D3_each_tuned_for_f1": "TunedF",
                     "D4_each_tuned_for_mcc": "TunedM"}
        for cid, cstem in conv_stem.items():
            e = fr["conventions"][cid]
            if e["q_ours"] is not None:
                L.append(f"\\newcommand{{\\Full{cstem}QOurs}}{{{e['q_ours']:.2f}}}")
                L.append(f"\\newcommand{{\\Full{cstem}QPTwo}}"
                         f"{{{e['q_p2rank']:.2f}}}")
            L.append(f"\\newcommand{{\\Full{cstem}CalledOurs}}"
                     f"{{{e['n_residues_called']['table_field']}}}")
            L.append(f"\\newcommand{{\\Full{cstem}CalledPTwo}}"
                     f"{{{e['n_residues_called']['p2rank']}}}")
            for met, mstem in metric_stem.items():
                p = e["paired"][met]["chain"]
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}Ours}}"
                         f"{{{e['table_field'][met]:.4f}}}")
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}PTwo}}"
                         f"{{{e['p2rank'][met]:.4f}}}")
                if p["delta_point"] is None:
                    continue
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}Delta}}"
                         f"{{{p['delta_point']:+.4f}}}")
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}CI}}"
                         f"{{[{p['delta_ci_low']:+.4f}, "
                         f"{p['delta_ci_high']:+.4f}]}}")
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}P}}"
                         f"{{{p['p_two_sided_bootstrap']:.3f}}}")
                L.append(f"\\newcommand{{\\Full{cstem}{mstem}N}}"
                         f"{{{p['n_paired_units']}}}")
        w = fr["where_the_deployment_rule_margin_came_from"]
        L.append(f"\\newcommand{{\\FullCallRatio}}"
                 f"{{{w['p2rank_calls_this_many_times_as_many']:.2f}}}")
        for unit, stem in (("chain", "Chain"), ("pdb_entry", "Pdb"),
                           ("uniprot_cluster", "Clu")):
            L.append(f"\\newcommand{{\\FullGroups{stem}}}"
                     f"{{{fr['resampling_units'][unit]}}}")
        # The widest any interval gets when the resampling unit is coarsened.
        # One number, because the claim is that none of them move, and a claim
        # about all of them is best supported by its own extreme.
        worst = max(
            (v[u]["ratio_to_chain"]
             for v in fr["ci_width_by_resampling_unit"].values()
             for u in ("pdb_entry", "uniprot_cluster")
             if v[u]["ratio_to_chain"] is not None), default=1.0)
        L.append(f"\\newcommand{{\\FullWidestClusterRatio}}{{{worst:.3f}}}")
        L.append(f"\\newcommand{{\\FullIntervals}}"
                 f"{{{fr['multiplicity']['intervals_examined']}}}")
        L.append(f"\\newcommand{{\\FullBonfAlpha}}"
                 f"{{{fr['multiplicity']['bonferroni_alpha_over_the_four_metrics_of_one_convention']:.4f}}}")
        L.append(f"\\newcommand{{\\FullReadIndex}}"
                 f"{{{fr['test_fold_read_index']}}}")
        for objective, stem in (("pooled_f1", "F"), ("pooled_mcc", "M")):
            g = top["what_tuning_p2rank_is_worth_on_the_training_fold"][objective]
            L.append(f"\\newcommand{{\\TrainGain{stem}}}"
                     f"{{{g['the_tuning_is_worth']:+.4f}}}")
        L.append(f"\\newcommand{{\\TrainQOurMCC}}"
                 f"{{{top['selected']['table_field/pooled_mcc/full_fold']['q']:.2f}}}")
        L.append(f"\\newcommand{{\\TrainQPTwoMCC}}"
                 f"{{{top['selected']['p2rank/pooled_mcc/full_fold']['q']:.2f}}}")
        # Whether choosing our threshold on a half our cells never counted
        # moves it. If it ever does, the sentence saying it does not has to go.
        shifts = [k for k, v in top["in_sample_optimism"].items() if not v["same"]]
        if shifts:
            raise SystemExit(
                f"the out-of-sample threshold now differs from the in-sample "
                f"one for {shifts}; the chapter says the choice is unaffected "
                f"by which half selected it and must be rewritten")
        L.append("\\newcommand{\\TrainQUnchangedOutOfSample}{yes}")
        fc = fr["forecast_vs_outcome"]
        L.append(f"\\newcommand{{\\FullForecastFOne}}"
                 f"{{{fc['f1']['forecast']:+.4f}}}")
        L.append(f"\\newcommand{{\\FullForecastMCC}}"
                 f"{{{fc['mcc']['forecast']:+.4f}}}")

    if AUDIT.exists():
        # The audit decomposition. The chapter's claim is that taking a score
        # apart says something a score alone does not, so the numbers that carry
        # the claim are the class means and the two gaps between them, not the
        # individual residues, which the chapter shows rather than quotes.
        ad = json.loads(AUDIT.read_text())
        sep = ad["what_separates_a_hit_from_a_miss"]
        cls = sep["mean_contribution_by_residue_class"]
        for name, stem in (("called and labelled", "Hit"),
                           ("called and not labelled", "FP"),
                           ("labelled and missed", "Miss"),
                           ("neither", "Rest")):
            row = cls[name]
            L.append(f"\\newcommand{{\\Aud{stem}N}}{{{row['n_residues']}}}")
            for fam, fstem in (("geometric", "Geom"), ("chemical", "Chem"),
                               ("topological", "Topo"),
                               ("density field", "Dens"),
                               ("spatial smoothing", "Gate")):
                L.append(f"\\newcommand{{\\Aud{stem}{fstem}}}"
                         f"{{{row[fam]:+.1f}}}")
        L.append(f"\\newcommand{{\\AudGeomMargin}}"
                 f"{{{sep['geometric_margin_of_hits_over_misses']:+.1f}}}")
        L.append(f"\\newcommand{{\\AudGateMargin}}"
                 f"{{{sep['spatial_smoothing_margin_of_hits_over_misses']:+.1f}}}")
        fpm = ad["what_the_false_positives_are_made_of"]
        L.append(f"\\newcommand{{\\AudGapToFP}}"
                 f"{{{fpm['largest_family_gap_to_false_positives']:.1f}}}")
        L.append(f"\\newcommand{{\\AudGapToMiss}}"
                 f"{{{fpm['largest_family_gap_to_missed_positives']:.1f}}}")
        L.append(f"\\newcommand{{\\AudGapRatio}}{{{fpm['ratio']:.0f}}}")
        fa = ad["family_assignment"]
        L.append(f"\\newcommand{{\\AudTablesMixed}}"
                 f"{{{fa['n_tables_spanning_two_families']:,}}}"
                 .replace(",", "{,}"))
        L.append(f"\\newcommand{{\\AudCases}}"
                 f"{{{ad['cases_are_not_chosen_here']['n_cases']}}}")
        # The exactness is the whole warrant for calling this a decomposition
        # rather than an attribution, so the manuscript states the residual.
        L.append(f"\\newcommand{{\\AudReproErr}}"
                 f"{{{ad['reconstruction']['worst_relative_error']:.0e}}}")
        # One worked residue, cited in the text. Named, so that editing the
        # artifact cannot leave a stale hand-typed quartile in the prose.
        worked = None
        for c in ad["cases"]:
            for r in c["residues"]:
                if c["unit_id"] == "2d05_A" and r["role"] == "true positive":
                    worked = (c, r)
        if worked is None:
            raise SystemExit(
                "the worked example the chapter cites (the true positive of "
                "2d05_A) is no longer in the audit artifact; the paragraph "
                "quoting its quartiles has to be rewritten around whatever the "
                "committed cases now contain")
        c, r = worked
        t = r["largest_single_tables"][0]
        L.append(f"\\newcommand{{\\AudEgUnit}}{{{c['unit_id'].replace('_', '\\_')}}}")
        L.append(f"\\newcommand{{\\AudEgRes}}{{{r['resnum']}}}")
        L.append(f"\\newcommand{{\\AudEgRank}}{{{r['score_rank_in_chain']}}}")
        L.append(f"\\newcommand{{\\AudEgOf}}{{{r['of_residues']}}}")
        L.append(f"\\newcommand{{\\AudEgQA}}{{{t['quantities'][0].replace('_', '\\_')}}}")
        L.append(f"\\newcommand{{\\AudEgQB}}{{{t['quantities'][1].replace('_', '\\_')}}}")
        L.append(f"\\newcommand{{\\AudEgLevA}}{{{t['quartile_of_this_residue'][0]}}}")
        L.append(f"\\newcommand{{\\AudEgLevB}}{{{t['quartile_of_this_residue'][1]}}}")
        L.append(f"\\newcommand{{\\AudEgRate}}"
                 f"{{{t['cell_binding_rate_in_training']:.3f}}}")
        L.append(f"\\newcommand{{\\AudEgTimesBase}}"
                 f"{{{t['cell_is_this_many_times_the_base_rate']:.1f}}}")
        L.append(f"\\newcommand{{\\AudEgMult}}{{{t['multiplicity']}}}")

    if COST.exists():
        # The controlled cost measurement. The earlier \TabSpeedup came from
        # telemetry taken with no pinned thread count and no common timing
        # boundary, and it said the wrong thing; these macros exist so the
        # paragraph that replaces it cannot be typed by hand.
        ct = json.loads(COST.read_text())
        warm, cold = ct["warm"], ct["cold"]
        wo, wp = warm["table_field"], warm["p2rank"]
        L.append(f"\\newcommand{{\\FairChains}}{{{ct['population']['chains']}}}")
        L.append(f"\\newcommand{{\\FairMachine}}{{{ct['controls']['processor']}}}")
        L.append(f"\\newcommand{{\\FairWarmOurs}}{{{wo['median_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairWarmOursIQR}}{{{wo['iqr_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairWarmPTwo}}"
                 f"{{{wp['amortised_per_chain_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairFeatures}}{{{wo['features_median_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairScoring}}{{{wo['scoring_median_s']:.4f}}}")
        L.append(f"\\newcommand{{\\FairFeatPct}}"
                 f"{{{100 * wo['fraction_of_the_median_spent_on_features']:.0f}}}")
        L.append(f"\\newcommand{{\\FairRSS}}{{{wo['peak_rss_mb']:.0f}}}")
        L.append(f"\\newcommand{{\\FairColdOurs}}"
                 f"{{{cold['table_field']['median_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairColdPTwo}}"
                 f"{{{cold['p2rank']['median_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairColdRatio}}{{{cold['ratio_of_medians']:.2f}}}")
        L.append(f"\\newcommand{{\\FairJVM}}{{{cold['jvm_start_median_s']:.3f}}}")
        L.append(f"\\newcommand{{\\FairJVMPct}}"
                 f"{{{100 * cold['jvm_start_as_fraction_of_p2ranks_cold_median']:.0f}}}")
        L.append(f"\\newcommand{{\\FairModelOurs}}"
                 f"{{{ct['model_size']['table_field_json_mb']:.2f}}}")
        L.append(f"\\newcommand{{\\FairModelOursGz}}"
                 f"{{{ct['model_size']['table_field_gzip_mb']:.2f}}}")
        L.append(f"\\newcommand{{\\FairModelPTwo}}"
                 f"{{{ct['model_size']['p2rank_default_model_mb']:.2f}}}")
        # Which way each ratio runs, written as a word, so the sentence around
        # it cannot say "faster" while the artifact says the opposite. The
        # margin is divided out of the two timings rather than out of the
        # artifact's rounded ratio, which would print 1.85 beside a verdict
        # sentence that says 1.84.
        def _margin(a: float, b: float) -> str:
            return f"{(b / a if b > a else a / b):.2f}"

        warm_ratio = warm["ratio_p2rank_over_table_field"]
        L.append(f"\\newcommand{{\\FairWarmRatio}}"
                 f"{{{_margin(wo['median_s'], wp['amortised_per_chain_s'])}}}")
        L.append(f"\\newcommand{{\\FairWarmWinner}}"
                 f"{{{'the counting field' if warm_ratio > 1 else 'P2Rank'}}}")
        par = ct["did_either_side_get_more_than_one_thread"]
        L.append(f"\\newcommand{{\\FairParOurs}}{{{par['table_field']:.2f}}}")
        L.append(f"\\newcommand{{\\FairParPTwo}}{{{par['p2rank_batch']:.2f}}}")
        if "cpu_seconds_per_chain" in wo:
            cpu_ratio = warm["ratio_p2rank_over_table_field_cpu"]
            L.append(f"\\newcommand{{\\FairCpuOurs}}"
                     f"{{{wo['cpu_seconds_per_chain']:.3f}}}")
            L.append(f"\\newcommand{{\\FairCpuPTwo}}"
                     f"{{{wp['cpu_seconds_per_chain']:.3f}}}")
            L.append(f"\\newcommand{{\\FairCpuRatio}}"
                     f"{{{_margin(wo['cpu_seconds_per_chain'], wp['cpu_seconds_per_chain'])}}}")
            L.append(f"\\newcommand{{\\FairCpuWinner}}"
                     f"{{{'the counting field' if cpu_ratio > 1 else 'P2Rank'}}}")
            # The two readings agreeing is what lets the chapter state one
            # conclusion. If they ever part, the paragraph needs rewriting
            # rather than a silently updated number.
            if (warm_ratio > 1) != (cpu_ratio > 1):
                raise SystemExit(
                    f"the wall-clock and CPU-second readings of the steady "
                    f"state disagree ({warm_ratio} against {cpu_ratio}); the "
                    f"cost paragraph asserts they agree and has to be rewritten "
                    f"to say which one it is reporting and why")
        # Section~\ref{sec:cost} is a withdrawal: it says in prose that the
        # steady state does not favour us. Macros alone cannot keep that honest,
        # because a reversal would leave the sentences intact and merely swap
        # the numbers inside them into a self-contradiction. Failing on the good
        # news is the cheaper error -- it costs one paragraph rewrite, where the
        # alternative costs a chapter that argues against its own figures.
        if warm_ratio > 1:
            raise SystemExit(
                f"the steady-state cost now favours the counting field "
                f"({warm_ratio}x), but Section 'What the method costs' is "
                f"written as a withdrawal of exactly that claim; rewrite the "
                f"paragraph before regenerating the macros")

    if ENDPOINT.exists():
        # Which endpoint is confirmatory. The manuscript used to lead with the
        # trimmed mean; these macros exist so that the sentence demoting it
        # carries the count that justifies the demotion, and so that the count
        # cannot drift from the commit graph it was read off.
        ep = json.loads(ENDPOINT.read_text())
        L.append(f"\\newcommand{{\\EndReadsBefore}}"
                 f"{{{ep['n_reads_before_the_preregistration']}}}")
        L.append(f"\\newcommand{{\\EndLineageBefore}}"
                 f"{{{sum(1 for r in ep['reads_before_the_preregistration']
                          if 'table field' in (r['method'] or ''))}}}")
        L.append(f"\\newcommand{{\\EndPriorBestAuc}}"
                 f"{{{max(r['mean_residue_auc'] for r in
                          ep['reads_before_the_preregistration']
                          if r['mean_residue_auc'] is not None):.4f}}}")
        c = ep["per_chain_outcome"]
        L.append(f"\\newcommand{{\\EndWins}}{{{c['n_field_ahead']}}}")
        L.append(f"\\newcommand{{\\EndLosses}}{{{c['n_baseline_ahead']}}}")
        L.append(f"\\newcommand{{\\EndMedianDelta}}"
                 f"{{{c['median_difference']:+.4f}}}")
        L.append(f"\\newcommand{{\\EndWorstLosses}}"
                 f"{{{c['worst_losses_mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\EndBestWins}}{{{c['best_wins_mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\EndNExploratory}}"
                 f"{{{len(ep['exploratory_endpoints'])}}}")
        L.append(f"\\newcommand{{\\EndNExploratoryResolving}}"
                 f"{{{sum(1 for e in ep['exploratory_endpoints']
                          if e['resolves'])}}}")
        # The manuscript states the primary endpoint by name and says it does
        # not resolve. Both have to remain true of the artifact.
        prim = ep["primary_endpoint"]
        if prim["statistic"] != "mean" or prim["resolves"]:
            raise SystemExit(
                f"the primary endpoint is now '{prim['statistic']}' and "
                f"{'resolves' if prim['resolves'] else 'does not resolve'}; the "
                f"abstract is written around an unresolved mean and has to be "
                f"rewritten before the macros are regenerated")

    if SUBGROUP.exists():
        # The subgroup read is exploratory by construction and the macros are
        # written so the manuscript cannot quote a band without also quoting the
        # number of bands examined and the trend that fails to support it.
        sg = json.loads(SUBGROUP.read_text())
        p = sg["per_chain_distribution"]
        L.append(f"\\newcommand{{\\SgTied}}{{{p['n_tied']}}}")
        L.append(f"\\newcommand{{\\SgSd}}{{{p['sd']:.4f}}}")
        L.append(f"\\newcommand{{\\SgBigLosses}}"
                 f"{{{p['n_losses_worse_than_5pct']}}}")
        L.append(f"\\newcommand{{\\SgBigWins}}"
                 f"{{{p['n_wins_better_than_5pct']}}}")
        L.append(f"\\newcommand{{\\SgQOhFive}}{{{p['quantiles']['0.05']:+.4f}}}")
        L.append(f"\\newcommand{{\\SgQNineFive}}{{{p['quantiles']['0.95']:+.4f}}}")
        L.append(f"\\newcommand{{\\SgNBandTests}}{{{sg['n_band_tests']}}}")
        L.append(f"\\newcommand{{\\SgNCovariates}}"
                 f"{{{len(sg['by_covariate'])}}}")
        L.append(f"\\newcommand{{\\SgBonf}}{{{sg['bonferroni_level']:.5f}}}")
        surv = sg["bands_excluding_zero_after_correction"]
        L.append(f"\\newcommand{{\\SgNSurviving}}{{{len(surv)}}}")
        L.append(f"\\newcommand{{\\SgNTrends}}"
                 f"{{{len(sg['trends_surviving_correction'])}}}")
        L.append(f"\\newcommand{{\\SgNSurvivingWithTrend}}"
                 f"{{{sg['n_surviving_bands_supported_by_a_trend']}}}")
        if len(surv) == 1:
            b = surv[0]
            # The artifact's covariate ids are field names; the manuscript needs
            # the thing they measure.
            names = {"prmsd": "apo/holo pocket RMSD",
                     "n_true": "pocket size",
                     "positive_rate": "positive rate",
                     "chain_length": "chain length",
                     "mean_bfactor": "mean pocket B-factor"}
            L.append(f"\\newcommand{{\\SgBandName}}"
                     f"{{{names[b['covariate']]}, {b['band']} third}}")
            L.append(f"\\newcommand{{\\SgBandDelta}}{{{b['mean']:+.4f}}}")
            L.append(f"\\newcommand{{\\SgBandCI}}"
                     f"{{[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]}}")
            L.append(f"\\newcommand{{\\SgBandN}}{{{b['n']}}}")
            L.append(f"\\newcommand{{\\SgBandRho}}"
                     f"{{{b['covariate_spearman_rho']:+.3f}}}")
            L.append(f"\\newcommand{{\\SgBandRhoP}}"
                     f"{{{b['covariate_spearman_p']:.2f}}}")
        # The manuscript describes the surviving band as unexplained by a trend.
        # If one ever acquires trend support that sentence becomes false, and it
        # should be rewritten deliberately rather than by regenerating macros.
        if sg["n_surviving_bands_supported_by_a_trend"]:
            raise SystemExit(
                f"{sg['n_surviving_bands_supported_by_a_trend']} surviving "
                f"band(s) are now supported by a monotone trend, but "
                f"Section 'Where the difference lives' is written around a band "
                f"that nothing explains; rewrite it before regenerating")
        if sg["status"] != "exploratory":
            raise SystemExit(
                f"the subgroup read now reports itself as {sg['status']}; the "
                f"manuscript presents it as exploratory throughout")

    if POCKET.exists():
        # The pocket stage. The macros are written so that the favourable
        # headline cannot be quoted without the candidate counts that bound it:
        # a hit rate at matched K is only meaningful beside how many candidates
        # each method offered.
        pk = json.loads(POCKET.read_text())
        pr, corr = pk["primary"], pk[
            "fairness_correction_p2rank_scored_at_its_own_residue_centroid"]
        c = pr["candidates_per_chain"]
        L.append(f"\\newcommand{{\\PkCut}}"
                 f"{{{pr['clustering_cutoff_angstrom']:.1f}}}")
        L.append(f"\\newcommand{{\\PkNPaired}}"
                 f"{{{pr['n_units_both_offer_a_candidate']}}}")
        L.append(f"\\newcommand{{\\PkNTheirsNone}}"
                 f"{{{pr['n_units_p2rank_offers_none']}}}")
        L.append(f"\\newcommand{{\\PkOursCand}}{{{c['ours_median']:.0f}}}")
        L.append(f"\\newcommand{{\\PkTheirsCand}}{{{c['p2rank_median']:.0f}}}")
        L.append(f"\\newcommand{{\\PkOursCandMean}}{{{c['ours_mean']:.3f}}}")
        L.append(f"\\newcommand{{\\PkTheirsCandMean}}{{{c['p2rank_mean']:.3f}}}")
        for key, stem in (("4.0A/top1", "OneFour"), ("4.0A/top3", "ThreeFour"),
                          ("6.0A/top1", "OneSix"), ("8.0A/top1", "OneEight")):
            h = pr["hit_rates"][key]
            L.append(f"\\newcommand{{\\PkOurs{stem}}}{{{h['ours']:.3f}}}")
            L.append(f"\\newcommand{{\\PkTheirs{stem}}}{{{h['p2rank']:.3f}}}")
            L.append(f"\\newcommand{{\\PkDelta{stem}}}"
                     f"{{{h['paired_95']['delta']:+.4f}}}")
            L.append(f"\\newcommand{{\\PkCI{stem}}}"
                     f"{{[{h['paired_95']['ci'][0]:+.4f}, "
                     f"{h['paired_95']['ci'][1]:+.4f}]}}")
            # Only the primary radius carries a corrected interval; the looser
            # radii are the sensitivity and the plan did not count them among the
            # tests, so there is no corrected macro to quote for them.
            if h.get("is_a_corrected_test"):
                L.append(f"\\newcommand{{\\PkBonf{stem}}}"
                         f"{{[{h['paired_bonferroni']['ci'][0]:+.4f}, "
                         f"{h['paired_bonferroni']['ci'][1]:+.4f}]}}")
        d = pr["top1_distance_to_labelled_site"]
        L.append(f"\\newcommand{{\\PkOursDist}}{{{d['ours_median']:.2f}}}")
        L.append(f"\\newcommand{{\\PkTheirsDist}}{{{d['p2rank_median']:.2f}}}")
        L.append(f"\\newcommand{{\\PkDistDelta}}"
                 f"{{{d['paired_95']['delta']:+.3f}}}")
        L.append(f"\\newcommand{{\\PkDistCI}}"
                 f"{{[{d['paired_95']['ci'][0]:+.3f}, "
                 f"{d['paired_95']['ci'][1]:+.3f}]}}")
        ch = corr["arm"]["hit_rates"]["4.0A/top1"]
        L.append(f"\\newcommand{{\\PkCorrTheirsOneFour}}{{{ch['p2rank']:.3f}}}")
        L.append(f"\\newcommand{{\\PkCorrDeltaOneFour}}"
                 f"{{{ch['paired_95']['delta']:+.4f}}}")
        L.append(f"\\newcommand{{\\PkCorrCIOneFour}}"
                 f"{{[{ch['paired_95']['ci'][0]:+.4f}, "
                 f"{ch['paired_95']['ci'][1]:+.4f}]}}")
        L.append(f"\\newcommand{{\\PkCorrDist}}"
                 f"{{{corr['arm']['top1_distance_to_labelled_site']['p2rank_median']:.2f}}}")
        for k, word in ((1, "One"), (3, "Three"), (5, "Five")):
            L.append(f"\\newcommand{{\\PkRecallOurs{word}}}"
                     f"{{{pr['recall'][f'top{k}']['ours']:.3f}}}")
            L.append(f"\\newcommand{{\\PkRecallTheirs{word}}}"
                     f"{{{pr['recall'][f'top{k}']['p2rank']:.3f}}}")
        # The manuscript reports this read as the one favourable result and as
        # exploratory twice over. Both halves have to keep being true.
        if pk["outcome_key"] != "top1_favours_the_field":
            raise SystemExit(
                f"the pocket read now returns {pk['outcome_key']}; the section "
                f"describing it is written around the top-1 advantage and has to "
                f"be rewritten rather than regenerated")
        if "exploratory" not in pk["status"].lower():
            raise SystemExit(
                f"the pocket read now reports itself as {pk['status']}; the "
                f"manuscript presents it as exploratory")
        if c["ours_mean"] > c["p2rank_mean"]:
            raise SystemExit(
                "the pocket stage now offers more candidates per chain than "
                "P2Rank, so the sentence conceding that it offers fewer is no "
                "longer true")

    if PLMNN.exists() and PLMNN_SCORES.exists():
        # The baseline that beats us. Every macro here is emitted whichever way
        # the read came out; the section is written around a deficit, and the
        # guard below fails if the outcome ever changes, because a section
        # explaining a loss cannot be regenerated into one explaining a win.
        pl = json.loads(PLMNN.read_text())
        sc = json.loads(PLMNN_SCORES.read_text())
        v = sc["validation_against_the_published_example"]
        pa = v["predicted_probability_agreement"]
        L.append(f"\\newcommand{{\\PlmReadIndex}}"
                 f"{{{pl['test_fold_read_index']}}}")
        L.append(f"\\newcommand{{\\PlmLayer}}{{{v['layer_recovered']}}}")
        L.append(f"\\newcommand{{\\PlmFidelity}}"
                 f"{{{v['mean_cosine_at_that_layer']:.4f}}}")
        L.append(f"\\newcommand{{\\PlmFidelityNext}}"
                 f"{{{v['second_best_mean_cosine']:.4f}}}")
        L.append(f"\\newcommand{{\\PlmRank}}{{{pa['spearman']:.4f}}}")
        L.append(f"\\newcommand{{\\PlmMaxDp}}"
                 f"{{{pa['max_absolute_difference']:.3f}}}")
        pc = pl["primary_comparison"]
        L.append(f"\\newcommand{{\\PlmLevel}}{{{pc['level_second']:.4f}}}")
        L.append(f"\\newcommand{{\\PlmAucDelta}}{{{pc['mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\PlmAucCI}}"
                 f"{{[{pc['ci'][0]:+.4f}, {pc['ci'][1]:+.4f}]}}")
        L.append(f"\\newcommand{{\\PlmAucBonf}}"
                 f"{{[{pc['ci_bonferroni'][0]:+.4f}, "
                 f"{pc['ci_bonferroni'][1]:+.4f}]}}")
        L.append(f"\\newcommand{{\\PlmWin}}{{{pc['n_first_ahead']}}}")
        L.append(f"\\newcommand{{\\PlmLoss}}{{{pc['n_second_ahead']}}}")
        cc = pl["context_comparison"]
        L.append(f"\\newcommand{{\\PlmAucCtxDelta}}{{{cc['mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\PlmAucCtxCI}}"
                 f"{{[{cc['ci'][0]:+.4f}, {cc['ci'][1]:+.4f}]}}")
        L.append(f"\\newcommand{{\\PlmNTests}}"
                 f"{{{pl['multiplicity']['n_paired_tests']}}}")
        for conv, cstem in (("common_budget", "Budget"),
                            ("each_methods_own_rule", "OwnRule")):
            for m, mstem in (("positive_class_f1", "F"), ("mcc", "M")):
                blk = pl["thresholded_metrics"][conv][m]
                dd = blk["paired_difference_ours_minus_plmnn"]
                L.append(f"\\newcommand{{\\Plm{cstem}{mstem}Theirs}}"
                         f"{{{blk['plmnn']:.4f}}}")
                L.append(f"\\newcommand{{\\Plm{cstem}{mstem}Delta}}"
                         f"{{{dd['mean']:+.4f}}}")
                L.append(f"\\newcommand{{\\Plm{cstem}{mstem}CI}}"
                         f"{{[{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}]}}")
        if pl["outcome"] != "the_baseline_is_ahead":
            raise SystemExit(
                f"the pLM-NN read now returns {pl['outcome']}; the abstract and "
                f"Section~\\ref{{sec:plmnn}} are written around the field being "
                f"behind the benchmark's own supervised baseline, and a different "
                f"outcome has to be rewritten rather than regenerated")
        if not pl["reproduction_gate"]["passes"]:
            raise SystemExit(
                "the pLM-NN reproduction no longer clears its own floor, so the "
                "comparison is against a baseline that may be broken and cannot "
                "be quoted in either direction")

    if PMREAD.exists():
        # The cryptic-specific baseline. Two macros here exist to stop the
        # headline being quoted alone: the P2Rank-minus-PocketMiner difference,
        # which shows the gap is not peculiar to this method, and the self-test
        # ROC-AUC, which shows the baseline was not rebuilt wrongly.
        pm = json.loads(PMREAD.read_text())
        L.append(f"\\newcommand{{\\PmReadIndex}}"
                 f"{{{pm['test_fold_read_index']}}}")
        L.append(f"\\newcommand{{\\PmLevel}}{{{pm['levels']['pocketminer']:.4f}}}")
        L.append(f"\\newcommand{{\\PmNUnits}}{{{pm['n_units']}}}")
        for key, stem in (("table_field_minus_pocketminer", "Ours"),
                          ("p2rank_minus_pocketminer", "PtwoR")):
            b = pm["primary"][key]
            L.append(f"\\newcommand{{\\PmAuc{stem}}}{{{b['mean']:+.4f}}}")
            L.append(f"\\newcommand{{\\PmAucCI{stem}}}"
                     f"{{[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]}}")
            L.append(f"\\newcommand{{\\PmAucBonf{stem}}}"
                     f"{{[{b['ci_bonferroni'][0]:+.4f}, "
                     f"{b['ci_bonferroni'][1]:+.4f}]}}")
            L.append(f"\\newcommand{{\\PmWin{stem}}}{{{b['n_first_ahead']}}}")
            L.append(f"\\newcommand{{\\PmLoss{stem}}}{{{b['n_second_ahead']}}}")
        ca = pm["contamination_arm"]
        L.append(f"\\newcommand{{\\PmDropped}}{{{len(ca['entries_removed'])}}}")
        L.append(f"\\newcommand{{\\PmCleanN}}{{{ca['n_units_left']}}}")
        cb = ca["table_field_minus_pocketminer"]
        L.append(f"\\newcommand{{\\PmCleanAuc}}{{{cb['mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\PmCleanAucCI}}"
                 f"{{[{cb['ci'][0]:+.4f}, {cb['ci'][1]:+.4f}]}}")
        for conv, cstem in (("common_budget", "Budget"),
                            ("their_trained_budget", "TheirBudget"),
                            ("their_trained_cut", "TheirCut")):
            if conv not in pm["thresholded"]:
                continue
            for m, mstem in (("positive_class_f1", "F"), ("mcc", "M")):
                blk = pm["thresholded"][conv][m]
                dd = blk["paired_difference_ours_minus_theirs"]
                L.append(f"\\newcommand{{\\Pm{cstem}{mstem}Theirs}}"
                         f"{{{blk['theirs']:.4f}}}")
                L.append(f"\\newcommand{{\\Pm{cstem}{mstem}Delta}}"
                         f"{{{dd['mean']:+.4f}}}")
                L.append(f"\\newcommand{{\\Pm{cstem}{mstem}CI}}"
                         f"{{[{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}]}}")
        rep = pm["reproduction_pinned_by_the_plan"]
        L.append(f"\\newcommand{{\\PmSelfAuc}}"
                 f"{{{rep['ours_on_their_test_set']:.4f}}}")
        L.append(f"\\newcommand{{\\PmSelfPublished}}"
                 f"{{{rep['published_roc_auc']:.2f}}}")
        L.append(f"\\newcommand{{\\PmBonfLevel}}"
                 f"{{{pm['multiplicity']['corrected_level']:.4f}}}")
        L.append(f"\\newcommand{{\\PmNTests}}"
                 f"{{{pm['multiplicity']['n_paired_tests']}}}")
        # The section is written around the field being ahead of a baseline that
        # was faithfully rebuilt. Both halves have to keep being true, and the
        # second is the one a reader cannot check by eye.
        if pm["outcome_key"] != "the_field_is_ahead":
            raise SystemExit(
                f"the PocketMiner read now returns {pm['outcome_key']}; the "
                f"section describing it is written around the field being ahead "
                f"and has to be rewritten rather than regenerated")
        if not rep["residue_counts_match_exactly"]:
            raise SystemExit(
                "the PocketMiner self-test no longer reproduces the published "
                "residue counts, so the comparison is against a baseline that "
                "may have been rebuilt wrongly and cannot be quoted")

    if EXTREAD.exists():
        # The only confirmatory comparison in the paper. Every macro here is
        # emitted in a group with the number that qualifies it, because each of
        # these figures is quotable alone in a way that would mislead: the P2Rank
        # advantage without the matched-budget arm that does not resolve, the
        # PocketMiner margin without P2Rank's own margin, the pLM-NN deficit
        # without the fact that it replicated rather than appeared.
        ex = json.loads(EXTREAD.read_text())
        if ex["status"] != "confirmatory":
            raise SystemExit(
                f"the external read now declares itself {ex['status']}; the "
                f"manuscript calls it the paper's one confirmatory comparison and "
                f"would have to be rewritten rather than regenerated")
        xs = json.loads(EXTSET.read_text())
        L.append(f"\\newcommand{{\\ExtN}}{{{ex['n_units_compared']}}}")
        L.append(f"\\newcommand{{\\ExtNPositives}}"
                 f"{{{sum(len(u['residues']) for u in xs['units'])}}}")
        L.append(f"\\newcommand{{\\ExtNPairs}}{{{xs['n_pairs_examined']}}}")
        L.append(f"\\newcommand{{\\ExtCutoff}}"
                 f"{{{xs['selection']['cutoff']}}}")
        for m, stem in (("table_field", "Ours"), ("p2rank", "PtwoR"),
                        ("plmnn", "Plm"), ("pocketminer", "Pm")):
            L.append(f"\\newcommand{{\\ExtLevel{stem}}}"
                     f"{{{ex['levels'][m]:.4f}}}")
        for key, stem in (("p2rank", "PtwoR"), ("plmnn", "Plm"),
                          ("pocketminer", "Pm")):
            b = ex["co_primary"][f"table_field_minus_{key}"]
            L.append(f"\\newcommand{{\\ExtAuc{stem}}}{{{b['mean']:+.4f}}}")
            L.append(f"\\newcommand{{\\ExtAucCI{stem}}}"
                     f"{{[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]}}")
            L.append(f"\\newcommand{{\\ExtAucBonf{stem}}}"
                     f"{{[{b['ci_bonferroni'][0]:+.4f}, "
                     f"{b['ci_bonferroni'][1]:+.4f}]}}")
            L.append(f"\\newcommand{{\\ExtWin{stem}}}{{{b['n_first_ahead']}}}")
            L.append(f"\\newcommand{{\\ExtLoss{stem}}}{{{b['n_second_ahead']}}}")
            L.append(f"\\newcommand{{\\ExtOld{stem}}}"
                     f"{{{b['cryptobench_predicted']['delta']:+.4f}}}")
            L.append(f"\\newcommand{{\\ExtVerdict{stem}}}"
                     f"{{{b['verdict'].replace('_', ' ')}}}")
        ctx = ex["co_primary"]["p2rank_minus_pocketminer_for_context"]
        L.append(f"\\newcommand{{\\ExtPtwoRMinusPm}}{{{ctx['mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\ExtPtwoRMinusPmCI}}"
                 f"{{[{ctx['ci'][0]:+.4f}, {ctx['ci'][1]:+.4f}]}}")
        th = ex["secondary_thresholded_against_p2rank"]
        L.append(f"\\newcommand{{\\ExtQPct}}{{{th['q'] * 100:.0f}}}")
        for arm, astem in (("as_deployed", "Deployed"),
                           ("common_budget", "Budget")):
            for m, mstem in (("positive_class_f1", "F"), ("mcc", "M")):
                dd = th[arm][m]["paired_difference_ours_minus_theirs"]
                L.append(f"\\newcommand{{\\Ext{astem}{mstem}Delta}}"
                         f"{{{dd['mean']:+.4f}}}")
                L.append(f"\\newcommand{{\\Ext{astem}{mstem}CI}}"
                         f"{{[{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}]}}")
        ab = th["p2rank_predicted_no_pocket_at_all"]
        L.append(f"\\newcommand{{\\ExtAbstain}}{{{ab['n']}}}")
        L.append(f"\\newcommand{{\\ExtAbstainUnit}}"
                 f"{{{_tex(ab['units'][0]['unit']) if ab['units'] else 'none'}}}")
        # The matched-budget arm is the one that keeps the advantage honest. If it
        # ever resolved, the sentence conceding it does not would be false, and a
        # regenerated macro would leave the false sentence in place.
        budget = th["common_budget"]["positive_class_f1"][
            "paired_difference_ours_minus_theirs"]
        if budget["excludes_zero"]:
            raise SystemExit(
                "the matched-budget F1 difference now excludes zero on the "
                "external set; the manuscript concedes that it does not and the "
                "concession has to be rewritten rather than regenerated")
        if ex["co_primary"]["table_field_minus_plmnn"]["mean"] > 0:
            raise SystemExit(
                "the external read now puts the field ahead of pLM-NN; every "
                "sentence in the manuscript about that deficit has to be "
                "rewritten rather than regenerated")

    if CURVE.exists():
        # The threshold axis. The only quantity the plan permits quoting is the
        # sign span, so that is the only quantity emitted -- no best point of any
        # curve reaches the manuscript through this file.
        cv = json.loads(CURVE.read_text())
        L.append(f"\\newcommand{{\\CurveReadIndex}}"
                 f"{{{cv['test_fold_read_index']}}}")
        L.append(f"\\newcommand{{\\CvN}}{{{cv['grid']['n']}}}")
        L.append(f"\\newcommand{{\\CvLowPct}}"
                 f"{{{cv['grid']['low'] * 100:.0f}}}")
        L.append(f"\\newcommand{{\\CvHighPct}}"
                 f"{{{cv['grid']['high'] * 100:.0f}}}")
        for arm, astem in (("p2rank", "PtwoR"), ("pocketminer", "Pm"),
                           ("plmnn", "Plm")):
            if arm not in cv["arms"]:
                continue
            for m, mstem in (("precision", "P"), ("recall", "R"),
                             ("positive_class_f1", "F"), ("mcc", "M")):
                sp = cv["arms"][arm]["sign_span"][m]
                L.append(f"\\newcommand{{\\Cv{astem}{mstem}Ahead}}"
                         f"{{{sp['n_where_ours_is_higher']}}}")
                L.append(f"\\newcommand{{\\Cv{astem}{mstem}Flips}}"
                         f"{{{sp['sign_changes']}}}")
                excl = [r["q"] for r in cv["arms"][arm]["curve"]
                        if r[m]["excludes_zero"]]
                L.append(f"\\newcommand{{\\Cv{astem}{mstem}Excl}}{{{len(excl)}}}")
        if cv["selects_nothing"]["consumed_by_any_configuration"]:
            raise SystemExit(
                "the threshold curve now feeds a configuration, which turns 39 "
                "cut points into 39 chances to pick one; the plan forbids it")

    if INTERP.exists():
        # The readout ladder. The chapter's claim here is a negative one -- the
        # architecture cannot be separated from a logistic regression on the
        # same wires -- so the interval that fails to exclude zero is as
        # load-bearing as the ones that do, and both are emitted.
        ib = json.loads(INTERP.read_text())
        stems = {"ridge direction": "Ridge",
                 "logistic regression": "Logit",
                 "additive over bins": "Additive",
                 "pairs, one round": "OneRound",
                 "pairs, sixteen rounds": "Field",
                 "pairs, sixteen rounds, unrounded": "Unrounded"}
        by_arm = {r["arm"]: r for r in ib["rows"]}
        for arm, stem in stems.items():
            r = by_arm[arm]
            L.append(f"\\newcommand{{\\Rd{stem}Auc}}"
                     f"{{{r['pick_half_roc_auc']:.4f}}}")
            L.append(f"\\newcommand{{\\Rd{stem}Raw}}"
                     f"{{{r['pick_half_roc_auc_raw']:.4f}}}")
            L.append(f"\\newcommand{{\\Rd{stem}Gain}}"
                     f"{{{r['spatial_smoothing_gain']:+.4f}}}")
            ci = ib["published_readout_against_each_other_arm"].get(arm)
            if ci:
                L.append(f"\\newcommand{{\\Rd{stem}Delta}}"
                         f"{{{ci['delta']:+.4f}}}")
                L.append(f"\\newcommand{{\\Rd{stem}CI}}"
                         f"{{[{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]}}")
        L.append(f"\\newcommand{{\\RdNTables}}"
                 f"{{{by_arm['pairs, sixteen rounds']['n_tables']:,}}}"
                 .replace(",", "{,}"))
        L.append(f"\\newcommand{{\\RdOneRoundTables}}"
                 f"{{{by_arm['pairs, one round']['n_tables']}}}")
        L.append(f"\\newcommand{{\\RdNPaired}}"
                 f"{{{ib['published_readout_against_each_other_arm']
                       ['logistic regression']['n_paired']}}}")
        L.append(f"\\newcommand{{\\RdBoot}}"
                 f"{{{ib['resampling']['draws']:,}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{\\RdNewtonSteps}}"
                 f"{{{by_arm['logistic regression']['newton_steps']}}}")
        # The negative result is the point of the section, so the manuscript
        # must not be able to keep its sentence if the interval moves off zero.
        unresolved = ib["arms_it_cannot_be_separated_from"]
        if unresolved != ["logistic regression"]:
            raise SystemExit(
                f"the readout comparison now cannot separate the field from "
                f"{unresolved or 'nothing'}, but the manuscript says the one "
                f"unresolved arm is the logistic regression; rewrite the "
                f"section before regenerating the macros")

    if BANKS_CEILING.exists() and WIDE3.exists() and WIDE3_CONTROL.exists():
        # The generated banks: a lift on the linear ceiling that the counting
        # field did not collect. The treatment and the control are emitted from
        # separate artifacts produced by the same tool on the same ridge grid,
        # so the chapter cannot quote a margin that came from searching one arm
        # harder than the other.
        bc = json.loads(BANKS_CEILING.read_text())
        gen, lift = bc["generator"], bc["lift"]
        base_key, top_key = bc["keys"]["baseline"], bc["keys"]["full"]
        L.append(f"\\newcommand{{\\GenOperator}}"
                 f"{{{gen['n_operator_descriptors']}}}")
        L.append(f"\\newcommand{{\\GenChain}}{{{gen['n_chain_descriptors']}}}")
        L.append(f"\\newcommand{{\\GenTotal}}"
                 f"{{{gen['n_operator_descriptors'] + gen['n_chain_descriptors']}}}")
        L.append(f"\\newcommand{{\\GenCeilBase}}"
                 f"{{{bc['ceilings'][base_key]['mean']:.4f}}}")
        L.append(f"\\newcommand{{\\GenCeilFull}}"
                 f"{{{bc['ceilings'][top_key]['mean']:.4f}}}")
        L.append(f"\\newcommand{{\\GenCeilLift}}{{{lift['mean']:+.4f}}}")
        L.append(f"\\newcommand{{\\GenCeilWorst}}{{{lift['min']:+.4f}}}")
        L.append(f"\\newcommand{{\\GenNSplits}}{{{lift['n_splits']}}}")
        L.append(f"\\newcommand{{\\GenNPositive}}"
                 f"{{{lift['n_splits_positive']}}}")
        fam = {f["family"]: f for f in bc["by_family"]}
        # Keyed by name, not by rank: the chapter names these four families and
        # says which were expected to matter, so a macro that silently followed
        # the ordering would let the text and the number drift apart.
        for stem, name in (("Trace", "spectral trace functional"),
                           ("Gyration", "gyration tensor"),
                           ("Diagonal", "diagonal functional at the centre"),
                           ("Hinge", "soft-mode hinge"),
                           ("Shape", "shape operator"),
                           ("Lag", "chain lag spectrum"),
                           ("Valuation", "valuation profile")):
            if name not in fam:
                raise SystemExit(f"the ceiling artifact has no family {name!r}; "
                                 f"the chapter names it")
            L.append(f"\\newcommand{{\\GenFam{stem}}}"
                     f"{{{fam[name]['delta_vs_algebraic_35']:+.4f}}}")
        ranked = sorted(fam.values(), key=lambda f: f["delta_vs_algebraic_35"])
        L.append(f"\\newcommand{{\\GenBestFamily}}"
                 f"{{{_tex(ranked[-1]['family'])}}}")
        L.append(f"\\newcommand{{\\GenWorstFamily}}"
                 f"{{{_tex(ranked[0]['family'])}}}")
        L.append(f"\\newcommand{{\\GenSecondWorstFamily}}"
                 f"{{{_tex(ranked[1]['family'])}}}")

        w3 = json.loads(WIDE3.read_text())
        w3c = json.loads(WIDE3_CONTROL.read_text())
        if w3["ridge_grid"] != w3c["ridge_grid"]:
            raise SystemExit("the two arms of the wide-wire comparison were "
                             "searched over different ridge grids, so their "
                             "difference is not attributable to the wires")
        L.append(f"\\newcommand{{\\GenWires}}{{{w3['n_wires']}}}")
        L.append(f"\\newcommand{{\\GenWiresBase}}{{{w3['n_wires_existing']}}}")
        L.append(f"\\newcommand{{\\GenWireAuc}}"
                 f"{{{w3['selected']['pick_half_roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GenWireControlAuc}}"
                 f"{{{w3c['selected']['pick_half_roc_auc']:.4f}}}")
        L.append(f"\\newcommand{{\\GenWireDelta}}"
                 f"{{{w3['selected']['pick_half_roc_auc'] - w3c['selected']['pick_half_roc_auc']:+.4f}}}")
        L.append(f"\\newcommand{{\\GenWireRidge}}"
                 f"{{{w3['selected']['ridge']:g}}}")
        L.append(f"\\newcommand{{\\GenWireControlRidge}}"
                 f"{{{w3c['selected']['ridge']:g}}}")
        pct = 100.0 * w3["selected"]["n_cells_never_addressed"] \
            / w3["selected"]["n_cells"]
        pctc = 100.0 * w3c["selected"]["n_cells_never_addressed"] \
            / w3c["selected"]["n_cells"]
        L.append(f"\\newcommand{{\\GenWireEmpty}}{{{pct:.2f}}}")
        L.append(f"\\newcommand{{\\GenWireControlEmpty}}{{{pctc:.2f}}}")

    if SENS.exists():
        sw = json.loads(SENS.read_text())
        if not sw["complete"]:
            raise SystemExit("the sensitivity sweep artifact is a checkpoint of "
                             "an unfinished run; the paper would quote a range "
                             "over settings that were never all measured")
        if sw["reads_test_fold"]:
            raise SystemExit("the sensitivity sweep claims to have read the "
                             "test fold, which the chapter says it did not")

        frozen_ridge = sw["frozen_configuration"]["ridge"]

        def _row(levels: int, ranking: str, cap: int,
                 ridge: float | None = None) -> dict:
            # The ridge has to be part of the key. Once the sweep varied it, a
            # lookup on levels, ranking and cap alone matched several rows and
            # returned whichever came first, which is a wrong number that looks
            # entirely plausible.
            want = frozen_ridge if ridge is None else ridge
            hits = [r for r in sw["rows"]
                    if (r["levels"], r["ranking"], r["cap"]) == (levels, ranking, cap)
                    and abs(r["ridge"] - want) < 1e-12]
            if len(hits) != 1:
                raise SystemExit(
                    f"the sweep has {len(hits)} rows for {levels} levels, "
                    f"{ranking} ranking, cap {cap}, ridge {want}; a macro "
                    f"cannot be emitted from an ambiguous or absent row")
            return hits[0]

        for lv in sw["swept"]["levels"]:
            stem = _roman(lv)
            L.append(f"\\newcommand{{\\SensLevels{stem}}}"
                     f"{{{_row(lv, 'within-chain', 32)['pick_half_roc_auc']:.4f}}}")
            L.append(f"\\newcommand{{\\SensLevels{stem}Pooled}}"
                     f"{{{_row(lv, 'pooled', 32)['pick_half_roc_auc']:.4f}}}")
            L.append(f"\\newcommand{{\\SensLevels{stem}Empty}}"
                     f"{{{100.0 * _row(lv, 'within-chain', 32)['fraction_never_addressed']:.2f}}}")
        # A control sequence is letters only, so the cap has to be spelled.
        cap_words = {16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}
        for cap in sw["swept"]["cap"]:
            if cap not in cap_words:
                raise SystemExit(f"no macro spelling for fan-out cap {cap}")
            L.append(f"\\newcommand{{\\SensCap{cap_words[cap]}}}"
                     f"{{{_row(4, 'within-chain', cap)['pick_half_roc_auc']:.4f}}}")
        four = _row(4, "within-chain", 32)["pick_half_roc_auc"]
        for lv in sw["swept"]["levels"]:
            if lv == 4:
                continue
            d = _row(lv, "within-chain", 32)["pick_half_roc_auc"] - four
            # The chapter reads these as losses against four levels. Emitting
            # the magnitude only is safe while that holds and a lie if it ever
            # stops, so it has to stop the build instead.
            if d >= 0:
                raise SystemExit(
                    f"{lv} levels now scores at or above four levels "
                    f"({d:+.4f}); Section 'What the constants are worth' calls "
                    f"it a loss and must be rewritten")
            L.append(f"\\newcommand{{\\SensLevels{_roman(lv)}Delta}}"
                     f"{{{abs(d):.4f}}}")
        # The published configuration fitted on the selection half, which is a
        # different object from the shipped compile over the whole training
        # fold and carries a different non-zero table count. Both are emitted,
        # each named for its fit set, and the two tools that measured the
        # selection half have to agree before either is quotable.
        pub = _row(4, "within-chain", 32)
        if WIDESEL.exists():
            wsel = json.loads(WIDESEL.read_text())["selected"]
            if wsel["n_tables_used"] != pub["n_tables_used"]:
                raise SystemExit(
                    f"the selection fit's non-zero table count disagrees "
                    f"between COUNTERATTACK_WIDE2 ({wsel['n_tables_used']}) "
                    f"and SENSITIVITY_SWEEP ({pub['n_tables_used']}); these "
                    f"are the same fit measured twice and must match")
        L.append(f"\\newcommand{{\\NTabUsedFitHalf}}{{{pub['n_tables_used']}}}")
        L.append(f"\\newcommand{{\\TabFanOutFitHalf}}{{{pub['total_fan_out']}}}")
        L.append(f"\\newcommand{{\\NTabCellsEmptyFitHalf}}"
                 f"{{{pub['n_cells_never_addressed']}}}")
        L.append(f"\\newcommand{{\\SensRange}}{{{sw['range_over_all_settings']:.4f}}}")
        L.append(f"\\newcommand{{\\SensSettings}}{{{len(sw['rows'])}}}")
        L.append(f"\\newcommand{{\\SensWorst}}"
                 f"{{{min(r['pick_half_roc_auc'] for r in sw['rows']):.4f}}}")
        L.append(f"\\newcommand{{\\SensBest}}"
                 f"{{{sw['best_pick_half_roc_auc']:.4f}}}")
        # The spread each constant is individually worth. The chapter's claim is
        # that three of the four are decorative and the fourth is the one the
        # method argues for, which is a claim about the relative sizes of these
        # four numbers, so they are computed here rather than being read off the
        # table by a reader who might read it wrong.
        def _spread(rows: list[dict]) -> float:
            vals = [r["pick_half_roc_auc"] for r in rows]
            return max(vals) - min(vals)

        spread = {
            "Levels": _spread([_row(lv, "within-chain", 32)
                               for lv in sw["swept"]["levels"]]),
            "Cap": _spread([_row(4, "within-chain", cap)
                            for cap in sw["swept"]["cap"]]),
            "Ridge": _spread([_row(4, "within-chain", 32, ridge=rg)
                              for rg in sw["swept"]["ridge"]]),
            "Ranking": _spread([_row(4, rk, 32)
                                for rk in sw["swept"]["ranking"]]),
        }
        for name, value in spread.items():
            L.append(f"\\newcommand{{\\Sens{name}Spread}}{{{value:.4f}}}")
        others = max(v for k, v in spread.items() if k != "Ranking")
        if spread["Ranking"] <= others:
            raise SystemExit(
                f"the ranking choice is now worth {spread['Ranking']:.4f}, no "
                f"more than the largest of the other constants ({others:.4f}); "
                f"the chapter says within-chain ranking is the one load-bearing "
                f"constant and must be rewritten")
        L.append(f"\\newcommand{{\\SensRankingOverNext}}"
                 f"{{{spread['Ranking'] / others:.1f}}}")
        # The chapter observes that the ridge sweep points one way before the
        # gate and the other way after it, which is the reason the sweep is
        # scored on the gated field. It is a sign comparison and would silently
        # invert if either curve flattened.
        lo_r, hi_r = min(sw["swept"]["ridge"]), max(sw["swept"]["ridge"])
        lo = _row(4, "within-chain", 32, ridge=lo_r)
        hi = _row(4, "within-chain", 32, ridge=hi_r)
        if not (hi["pick_half_roc_auc_raw"] > lo["pick_half_roc_auc_raw"]
                and hi["pick_half_roc_auc"] < lo["pick_half_roc_auc"]):
            raise SystemExit(
                f"the ridge no longer moves the raw and gated scores in "
                f"opposite directions (raw {lo['pick_half_roc_auc_raw']:.4f} to "
                f"{hi['pick_half_roc_auc_raw']:.4f}, gated "
                f"{lo['pick_half_roc_auc']:.4f} to {hi['pick_half_roc_auc']:.4f}"
                f"); the sentence saying an ungated sweep would have chosen the "
                f"opposite constant has to go")
        L.append(f"\\newcommand{{\\SensLevelsFourPooledEmpty}}"
                 f"{{{100.0 * _row(4, 'pooled', 32)['fraction_never_addressed']:.2f}}}")

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
    # The same budget as a percentage, because that is how the prose says it.
    L.append(f"\\newcommand{{\\AlgOperatingPct}}"
             f"{{{100 * field['operating_point']['q']:.0f}}}")
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

    L.append("% The five training-fold sweeps of the construction's own knobs.")
    L += _saturation_macros()
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


_FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)


def _figure_captions_match() -> list[str]:
    """Every figure must carry the caption written for the image it shows.

    The captions are generated from an artifact and the manuscript cites them
    by macro, which removes the risk of a caption drifting from its numbers but
    not the risk of it sitting under the wrong picture. That happened: the
    macros were numbered by draw order, a figure was inserted ahead of the case
    studies, and the case-study plot kept a caption written about a histogram of
    paired differences. Nothing in the pipeline noticed, because both halves
    were individually up to date.
    """
    bad = []
    for tex in sorted(SRC.glob("*.tex")):
        if tex.name == OUT.name:
            continue
        for body in _FIGURE_ENV.findall(tex.read_text()):
            imgs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body)
            caps = re.findall(r"\\caption\{\\([A-Za-z]+)\{\}\}", body)
            if len(imgs) != 1 or len(caps) != 1:
                continue
            want = _caption_macro(imgs[0])
            if caps[0] != want:
                bad.append(f"{tex.name}: {imgs[0]} carries \\{caps[0]}{{}}, "
                           f"but its caption is \\{want}{{}}")
    return bad


def _dangling_refs() -> list[str]:
    """Cross-references with no label anywhere in the manuscript.

    TeX renders these as ``??`` and exits zero, so a broken reference survives
    every check that only asks whether the document compiled. Three did: two
    sections pointed at a metrics table that had never been labelled, and the
    generated appendix pointed at an unlabelled Methods section. The labels are
    collected across all the .tex files because the appendices are separate
    files included into one document.
    """
    labels, refs = set(), {}
    for tex in sorted(SRC.glob("*.tex")):
        body = tex.read_text()
        labels |= set(re.findall(r"\\label\{([^}]+)\}", body))
        for name in re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", body):
            refs.setdefault(name, tex.name)
    return [f"{where} cites \\ref{{{name}}}, which is never labelled"
            for name, where in sorted(refs.items()) if name not in labels]


def _saturation_macros() -> list[str]:
    """Macros for the five sweeps that measured the construction's own knobs.

    Every value here is a training-fold quantity and none reads the test fold or
    an external unit; the artifacts each declare that and the emitter refuses to
    quote one that does not. The macros are grouped by artifact rather than by
    number so that a reader tracing a sentence back lands in one file.

    One rule is enforced here rather than left to prose. The reseed magnitude
    from SELECTED_PAIRINGS.json is the noise floor for any comparison between two
    banks, so ``\\SatReseed`` is emitted next to every bank-versus-bank
    difference the manuscript quotes, and the union-versus-widened difference is
    emitted with it deliberately: that difference is smaller than the floor and
    the sentence citing it has to say so.
    """
    L: list[str] = []
    for path in (LADDER, PAIRSEL, COMPWIRE, GRAMCOND, TRUNC, HIER, GATEROUTE,
                 TABWIDTH, COMBMULT):
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        if doc.get("reads_test_fold") is not False:
            raise SystemExit(
                f"{path.relative_to(ROOT)} does not declare "
                f"reads_test_fold: false; it may not feed the manuscript")
        if doc.get("reads_any_external_unit") is not False:
            raise SystemExit(
                f"{path.relative_to(ROOT)} does not declare "
                f"reads_any_external_unit: false")

    if LADDER.exists():
        d = json.loads(LADDER.read_text())
        md = d["minus_deployed"]
        occ = d["cell_occupancy_on_split_1"]
        dep = "uniform quartiles (deployed)"
        for stem, key in (("Ten", "tails at 10 %"), ("Five", "tails at 5 %"),
                          ("Two", "tails at 2 %")):
            L.append(f"\\newcommand{{\\LadderTail{stem}}}"
                     f"{{{_signed(md[key]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\LadderTail{stem}Splits}}"
                     f"{{{md[key]['n_splits_positive']}}}")
        L.append(f"\\newcommand{{\\LadderSplits}}"
                 f"{{{md[dep]['n_splits']}}}")
        L.append(f"\\newcommand{{\\LadderCellsDeployed}}"
                 f"{{{occ[dep]['median_count_of_addressed_cells']}}}")
        L.append(f"\\newcommand{{\\LadderCellsTwo}}"
                 f"{{{occ['tails at 2 %']['median_count_of_addressed_cells']}}}")
        L.append(f"\\newcommand{{\\LadderEmptyDeployed}}"
                 f"{{{100 * occ[dep]['fraction_never_addressed']:.2f}}}")
        L.append(f"\\newcommand{{\\LadderEmptyTwo}}"
                 f"{{{100 * occ['tails at 2 %']['fraction_never_addressed']:.2f}}}")
        rep = d.get("reproduction_check") or {}
        # The ladder artifact names this field differently from the later ones.
        # Reading the wrong key silently emitted 0.0e+00, which would have made
        # the manuscript claim a perfect reproduction it had not measured.
        gap = rep.get("max_absolute_difference_from_frozen")
        if gap is None:
            raise SystemExit(
                f"{LADDER.relative_to(ROOT)} carries no reproduction gap; the "
                f"manuscript may not quote one")
        L.append(f"\\newcommand{{\\LadderRepro}}{{{gap:.1e}}}")

    if PAIRSEL.exists():
        d = json.loads(PAIRSEL.read_text())
        md = d["minus_deployed"]
        for stem, key in (("Inter", "interaction"), ("Var", "pair variance"),
                          ("Reseed", "another seed"),
                          ("Anti", "anti-selected")):
            L.append(f"\\newcommand{{\\Pair{stem}}}"
                     f"{{{_signed(md[key]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\Pair{stem}Splits}}"
                     f"{{{md[key]['n_splits_positive']}}}")
        # The noise floor. Quoted as a magnitude because it is used as one.
        L.append(f"\\newcommand{{\\SatReseed}}"
                 f"{{{abs(md['another seed']['mean']):.4f}}}")
        surv = d["does_the_chosen_interaction_survive_on_split_1"]["banks"]
        L.append(f"\\newcommand{{\\PairSurviveSelected}}"
                 f"{{{surv['interaction']['interaction_surviving']:.2f}}}")
        L.append(f"\\newcommand{{\\PairSurviveRandom}}"
                 f"{{{surv['another seed']['interaction_surviving']:.2f}}}")
        fit = surv["interaction"]["on_the_fit_half_it_was_chosen_from"][
            "mean_interaction"]
        rnd = surv["another seed"]["on_the_fit_half_it_was_chosen_from"][
            "mean_interaction"]
        L.append(f"\\newcommand{{\\PairInteractionRatio}}{{{fit / rnd:.0f}}}")
        L.append(f"\\newcommand{{\\PairGreedyIdeal}}"
                 f"{{{d['greedy_matching_on_split_1']['interaction'][0]['fraction_of_ideal']:.2f}}}")

    if COMPWIRE.exists():
        d = json.loads(COMPWIRE.read_text())
        L.append(f"\\newcommand{{\\CompCols}}{{{d['columns']['n']}}}")
        L.append(f"\\newcommand{{\\CompClasses}}"
                 f"{{{len(d['columns']['classes'])}}}")
        for stem, key in (("Widened", "widened"), ("Union", "union")):
            md = d["minus_deployed"][key]
            L.append(f"\\newcommand{{\\Comp{stem}}}"
                     f"{{{_signed(md['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\Comp{stem}Splits}}"
                     f"{{{md['n_splits_positive']}}}")
        uw = d["union_minus_widened"]
        L.append(f"\\newcommand{{\\CompUnionMinusWidened}}"
                 f"{{{_signed(uw['mean'], 4)}}}")
        L.append(f"\\newcommand{{\\CompUnionMinusWidenedSplits}}"
                 f"{{{uw['n_splits_positive']}}}")
        f = d["fisher_lift_from_composition"]
        L.append(f"\\newcommand{{\\CompFisher}}{{{_signed(f['mean'], 4)}}}")
        L.append(f"\\newcommand{{\\CompFisherSplits}}"
                 f"{{{f['n_splits_positive']}}}")
        L.append(f"\\newcommand{{\\CompSurvived}}"
                 f"{{{d['bank']['n_old_pairings_that_survive_widening']}}}")
        L.append(f"\\newcommand{{\\CompTablesFull}}"
                 f"{{{d['bank']['n_tables_over_the_old_wires']}}}")

    if GRAMCOND.exists():
        d = json.loads(GRAMCOND.read_text())
        s = d["summary"]["deployed"]
        L.append(f"\\newcommand{{\\GramCos}}"
                 f"{{{s['cosine_solution_to_mean_difference']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\GramRound}}"
                 f"{{{s['cosine_rounded_to_real']['mean']:.4f}}}")
        L.append(f"\\newcommand{{\\GramRidgeShare}}"
                 f"{{{100 * s['lambda_share_of_diagonal']['mean']:.1f}}}")
        L.append(f"\\newcommand{{\\GramTraceTop}}"
                 f"{{{100 * s['trace_fraction_in_top_1_percent']['mean']:.1f}}}")
        L.append(f"\\newcommand{{\\GramEffRank}}"
                 f"{{{s['effective_rank_at_1e-6']['mean']:.0f}}}")
        sel = d["summary"]["interaction-selected"]
        L.append(f"\\newcommand{{\\GramEffRankSelected}}"
                 f"{{{sel['effective_rank_at_1e-6']['mean']:.0f}}}")
        L.append(f"\\newcommand{{\\GramSplits}}"
                 f"{{{d['protocol']['n_splits']}}}")
        o = d["orderings"]
        L.append("\\newcommand{\\GramAgrees}"
                 f"{{{'yes' if o['auc_agrees_with_conditioning'] else 'no'}}}")

    if TRUNC.exists():
        d = json.loads(TRUNC.read_text())
        # Sizes are spelled, not digits: a TeX control sequence is letters only,
        # so \TruncMult52 compiles as \TruncMult followed by a literal 52. That
        # happened once with \NLocP2 and tests/test_frozen_numbers.py forbids it.
        # The size itself is emitted as its own macro so the prose can cite the
        # number without typing it.
        for k, word in ((52, "Fifty"), (208, "TwoHundred"),
                        (1664, "SixteenHundred")):
            L.append(f"\\newcommand{{\\TruncK{word}}}{{{k}}}")
            for stem, rule in (("Mult", "by multiplicity"),
                               ("Gini", "by gini"), ("Rand", "random")):
                c = d["curves"][rule][str(k)]
                L.append(f"\\newcommand{{\\Trunc{stem}{word}}}"
                         f"{{{_signed(c['delta_mean'], 4)}}}")
        L.append(f"\\newcommand{{\\TruncFull}}"
                 f"{{{d['held_fixed']['n_tables_full']}}}")
        tol = d["smallest_bank_within_tolerance"]["0.001"]
        L.append(f"\\newcommand{{\\TruncWithinMilli}}"
                 f"{{{tol['by multiplicity']}}}")
        L.append(f"\\newcommand{{\\TruncWithinMilliRand}}"
                 f"{{{tol['random']}}}")
        rep = d.get("reproduction_check") or {}
        L.append(f"\\newcommand{{\\TruncRepro}}"
                 f"{{{rep.get('max_absolute_difference', 0):.1e}}}")

    if HIER.exists():
        d = json.loads(HIER.read_text())
        md, mr = d["minus_deployed"], d["minus_random_per_chain"]
        # Both baselines, emitted together and deliberately. This artifact holds
        # three of them and the answer differs by which is used: the best routed
        # arm is +0.0005 over the deployed solve and the same family is +0.0054
        # over a random router. A sentence quoting one while the reader supplies
        # the other is the failure this pairing is meant to make awkward.
        best = max(md, key=lambda k: md[k]["mean"] if k != "global" else -9)
        L.append(f"\\newcommand{{\\HierBestArm}}"
                 f"{{{best.replace('R=4, ', '').replace('d=', 'damping ')}}}")
        L.append(f"\\newcommand{{\\HierBestVsDeployed}}"
                 f"{{{_signed(md[best]['mean'], 4)}}}")
        L.append(f"\\newcommand{{\\HierBestVsDeployedSplits}}"
                 f"{{{md[best]['n_splits_positive']}}}")
        worst = min(md, key=lambda k: md[k]["mean"])
        L.append(f"\\newcommand{{\\HierWorstVsDeployed}}"
                 f"{{{_signed(md[worst]['mean'], 4)}}}")
        top = max(mr, key=lambda k: mr[k]["mean"])
        L.append(f"\\newcommand{{\\HierBestVsRandom}}"
                 f"{{{_signed(mr[top]['mean'], 4)}}}")
        L.append(f"\\newcommand{{\\HierBestVsRandomSplits}}"
                 f"{{{mr[top]['n_splits_positive']}}}")
        L.append(f"\\newcommand{{\\HierArms}}{{{len(md) - 1}}}")
        L.append(f"\\newcommand{{\\HierSplits}}{{{d['protocol']['n_splits']}}}")
        rep = d.get("reproduction_check") or {}
        L.append(f"\\newcommand{{\\HierRepro}}"
                 f"{{{rep.get('max_absolute_difference', 0):.1e}}}")

    if GATEROUTE.exists():
        d = json.loads(GATEROUTE.read_text())
        mg, mr = d["minus_global_fitted"], d["minus_random_per_chain"]
        for stem, key in (("Size", "chain size"), ("Polarity", "chain polarity")):
            L.append(f"\\newcommand{{\\GateRoute{stem}}}"
                     f"{{{_signed(mg[key]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\GateRoute{stem}Splits}}"
                     f"{{{mg[key]['n_splits_positive']}}}")
            L.append(f"\\newcommand{{\\GateRoute{stem}VsRandom}}"
                     f"{{{_signed(mr[key]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\GateRoute{stem}VsRandomSplits}}"
                     f"{{{mr[key]['n_splits_positive']}}}")
        L.append(f"\\newcommand{{\\GateRouteGlobalVsDeployed}}"
                 f"{{{_signed(d['global_fitted_minus_deployed']['mean'], 4)}}}")
        # The region count is read off a router's own weight vector rather than
        # parsed out of the prose in n_parameters, which is where it also appears.
        L.append(f"\\newcommand{{\\GateRouteRegions}}"
                 f"{{{len(d['weights_chosen']['chain size'][0]['weights'])}}}")
        rep = d.get("reproduction_check") or {}
        L.append(f"\\newcommand{{\\GateRouteRepro}}"
                 f"{{{rep.get('max_absolute_difference', 0):.1e}}}")

    if TABWIDTH.exists():
        d = json.loads(TABWIDTH.read_text())
        md, arms = d["minus_deployed"], d["arms"]
        occ = d["cell_occupancy_first_split"]
        # Named by width and matching rule, spelled out, because a control
        # sequence is letters only.
        for name, stem in (("width 1, matched tables", "OneTables"),
                           ("width 3, matched cells", "ThreeCells"),
                           ("width 3, matched rounds", "ThreeRounds"),
                           ("width 4, matched cells", "FourCells")):
            if name not in md:
                continue
            L.append(f"\\newcommand{{\\Width{stem}}}"
                     f"{{{_signed(md[name]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\Width{stem}Splits}}"
                     f"{{{md[name]['n_splits_positive']}}}")
            L.append(f"\\newcommand{{\\Width{stem}Cells}}"
                     f"{{{occ[name]['median_count_of_addressed_cells']}}}")
        dep = d["deployed_arm"]
        L.append(f"\\newcommand{{\\WidthTwoCells}}"
                 f"{{{occ[dep]['median_count_of_addressed_cells']}}}")
        L.append(f"\\newcommand{{\\WidthTwoTables}}{{{arms[dep]['n_tables']}}}")
        rep = d.get("reproduction_check") or {}
        L.append(f"\\newcommand{{\\WidthRepro}}"
                 f"{{{rep.get('max_absolute_difference', 0):.1e}}}")

    if COMBMULT.exists():
        d = json.loads(COMBMULT.read_text())
        md, cos, arms = (d["minus_deployed"], d["cosine_with_the_deployed_direction"],
                         d["arms"])
        for name, stem in (("diagonal solve", "Diag"),
                           ("standardised delta", "Std"),
                           ("rank bands", "Bands"),
                           ("sign only", "Sign"),
                           ("random signs", "RandSign")):
            L.append(f"\\newcommand{{\\Comb{stem}}}"
                     f"{{{_signed(md[name]['mean'], 4)}}}")
            L.append(f"\\newcommand{{\\Comb{stem}Splits}}"
                     f"{{{md[name]['n_splits_positive']}}}")
            L.append(f"\\newcommand{{\\Comb{stem}Cos}}{{{cos[name]:.3f}}}")
            L.append(f"\\newcommand{{\\Comb{stem}Auc}}"
                     f"{{{arms[name]['mean']:.4f}}}")
        L.append(f"\\newcommand{{\\CombDeployedAuc}}"
                 f"{{{arms['deployed ridge solve']['mean']:.4f}}}")
        # The share of the margin over chance that survives when the off-diagonal
        # is deleted. Computed here so the manuscript cannot round it by hand.
        dep, best = arms["deployed ridge solve"]["mean"], arms["sign only"]["mean"]
        L.append(f"\\newcommand{{\\CombSignShare}}"
                 f"{{{100 * (best - 0.5) / (dep - 0.5):.0f}}}")
        L.append(f"\\newcommand{{\\CombClosestRule}}{{{d['closest_rule']}}}")
        rep = d.get("reproduction_check") or {}
        L.append(f"\\newcommand{{\\CombRepro}}"
                 f"{{{rep.get('max_absolute_difference', 0):.1e}}}")
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed file is stale")
    args = ap.parse_args(argv)
    text = build()
    dangling = _dangling_refs()
    if dangling:
        print("DANGLING cross-references:")
        for line in dangling:
            print(f"  - {line}")
        return 1
    mispaired = _figure_captions_match()
    if mispaired:
        print("MISPAIRED figure captions:")
        for line in mispaired:
            print(f"  - {line}")
        return 1
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
