"""Generate the manuscript's numeric macros from the frozen result JSONs.

Every review of this repository has caught the same class of defect: a number in
prose that no longer matches the artifact it came from. Proof-reading does not
fix that class, because the prose and the artifact drift independently. Removing
the second copy does.

This tool emits ``paper/frozen_numbers.tex``, a file of ``\\newcommand`` macros
read straight out of ``results/cryptobench_official/BOOTSTRAP_CI.json``. The
manuscript cites ``\\AlgAuc`` and never a literal. ``tests/test_frozen_numbers``
regenerates the file and fails if it differs from the committed one, so a
manuscript number can only change when the artifact changes.

Usage: PYTHONPATH=src python3.12 tools/emit_frozen_numbers.py [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
FIELD = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD.json"
LINFIELD = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
WIDESEL = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE2.json"
WIDEPROBE = ROOT / "results/official_fold/COUNTERATTACK_WIDE_PROBE.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
SELECTION = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"
CEILING = ROOT / "results/architecture_sweep/FEATURE_CEILING_DIAGNOSIS.json"
READOUT = ROOT / "results/architecture_sweep/FINAL_READOUT_SELECTION.json"
PAIRWISE = ROOT / "results/architecture_sweep/PAIRWISE_READOUT_SELECTION.json"
SEEDPROBE = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
SEEDPROBE = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
OUT = ROOT / "paper/frozen_numbers.tex"

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed file is stale")
    args = ap.parse_args(argv)
    text = build()
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
