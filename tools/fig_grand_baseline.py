#!/usr/bin/env python3.12
"""Two figures for the three-baseline read, drawn so a parity cannot read as a win.

Panel A is the standing: every detector and every published baseline on the same
192 units, on both per-unit metrics, with the pooled residue read beside them.
Panel B is the thing a bar chart hides — the paired differences with their
intervals, where an interval crossing zero is drawn crossing zero and is not
recoloured to look decisive.

The ordering rule matters. Methods are drawn in the order they appear in the
artifact rather than sorted by score, because sorting by the metric shown is how
a figure comes to argue for the metric that flatters it. The one place ordering
is chosen is the difference panel, sorted by effect size, and that is stated in
the caption.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pocket_bench.paths import ROOT  # noqa: E402

READ = ROOT / "results/official_fold/GRAND_BASELINE_READ.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
# `tools/classify_artifacts.py` finds the images a generator claims by matching
# the literal `FIGDIR / "name"`, so the two paths are spelled that way rather
# than inlined into one string. A figure whose name the gate cannot read is a
# figure with no provenance, which is the state these two were committed in.
FIGDIR = ROOT / "figures"
OUT_A = FIGDIR / "fig_grand_baseline_standing.png"
OUT_B = FIGDIR / "fig_grand_baseline_paired.png"
PROV = ROOT / "results/official_fold/GRAND_BASELINE_FIGURE_PROVENANCE.json"

OURS = ("table_field", "geometry_field", "seam_geometry_field",
        "geo_seam_equalz", "geometry_field_r14", "seam_geometry_field_r14",
        "geo_seam_equalz_r14")
BASE = ("plmnn", "p2rank", "pocketminer")
PRETTY = {
    "table_field": "table_field\n(deployed, 645 wires)",
    "geometry_field": "geometry_field\n(645 + 624)",
    "seam_geometry_field": "seam_geometry_field\n(+129 nonlocal seam)",
    "geo_seam_equalz": "geo ⊕ seam\n(equal-z, unfitted)",
    "geometry_field_r14": "geometry_field\ngate r=14",
    "seam_geometry_field_r14": "seam_geometry_field\ngate r=14",
    "geo_seam_equalz_r14": "geo ⊕ seam, gate r=14\n(best standing)",
    "plmnn": "pLM-NN\n(CryptoBench)",
    "p2rank": "P2Rank 2.5.1",
    "pocketminer": "PocketMiner",
}
OURS_C, BASE_C = "#1f4e79", "#b0483a"

# Axis labels may carry a line break and a glyph; a caption may not. These
# strings are emitted into `paper/frozen_numbers.tex` and compiled by pdfLaTeX,
# where a bare U+2295 is an error, and they are also written into README.md,
# where a line break inside a caption would split the paragraph.
PROSE = {
    "geo_seam_equalz": "geometry + seam, equal-z",
    "geo_seam_equalz_r14": "geometry + seam, equal-z, gate r=14",
    "geometry_field_r14": "geometry_field, gate r=14",
    "seam_geometry_field_r14": "seam_geometry_field, gate r=14",
    "plmnn": "pLM-NN (CryptoBench's own)",
}


def panel_a(d: dict, out: Path) -> None:
    order = list(OURS) + list(BASE)
    s = d["summary"]
    metrics = [("mean_per_unit_roc_auc", "per-unit ROC-AUC"),
               ("mean_per_unit_pr_auc", "per-unit PR-AUC"),
               ("pooled_residue_roc_auc_on_rank_fractions",
                "pooled residue ROC-AUC\n(within-chain rank fractions)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.6))
    for ax, (key, title) in zip(axes, metrics):
        vals = [s[m][key] for m in order]
        cols = [OURS_C if m in OURS else BASE_C for m in order]
        ypos = np.arange(len(order))[::-1]
        ax.barh(ypos, vals, color=cols, height=0.66)
        for y, v in zip(ypos, vals):
            ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([PRETTY[m] for m in order], fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    n = d["summary"]["plmnn"]["n_units_scored"]
    fig.suptitle(
        f"CryptoBench official test fold, {n} single-chain apo units, one "
        f"residue universe.  Blue: this repository (no learned encoder, no "
        f"floating-point discriminant).  Red: published baselines, each "
        f"rebuilt locally.\nclinical_grade = false; development read on a "
        f"fold that has been read many times.",
        fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def panel_b(d: dict, out: Path) -> None:
    rows = []
    for key, v in d["paired"].items():
        if not key.startswith(OURS):
            continue
        for metric, label in (("per_unit_roc_auc", "ROC-AUC"),
                              ("per_unit_pr_auc", "PR-AUC")):
            p = v.get(metric)
            if not p or p.get("mean_delta") is None:
                continue
            ours, _, base = key.partition("_minus_")
            rows.append({
                "label": f"{PRETTY[ours].splitlines()[0]}  −  "
                         f"{PRETTY[base].splitlines()[0]}",
                "metric": label, "d": p["mean_delta"], "ci": p["ci95"],
                "res": p["excludes_zero"], "ahead": p["n_ahead"],
                "behind": p["n_behind"], "n": p["n_paired"],
            })
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 10.5), sharey=False)
    for ax, metric in zip(axes, ("ROC-AUC", "PR-AUC")):
        rs = sorted([r for r in rows if r["metric"] == metric],
                    key=lambda r: r["d"])
        y = np.arange(len(rs))
        for i, r in enumerate(rs):
            lo, hi = r["ci"]
            col = "#1f4e79" if (r["res"] and r["d"] > 0) else (
                "#b0483a" if (r["res"] and r["d"] < 0) else "#8a8a8a")
            ax.plot([lo, hi], [i, i], color=col, linewidth=2.4,
                    solid_capstyle="butt")
            ax.plot([r["d"]], [i], "o", color=col, markersize=6.5)
            ax.text(hi + 0.004, i,
                    f"{r['d']:+.4f}  {r['ahead']}/{r['behind']}"
                    + ("" if r["res"] else "   (crosses 0)"),
                    va="center", fontsize=8.2, color=col)
        ax.axvline(0, color="black", linewidth=1.0, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in rs], fontsize=8.2)
        ax.set_xlabel(f"paired difference in per-unit {metric}  "
                      f"(95 % bootstrap, 10 000 resamples)", fontsize=9)
        ax.set_xlim(min(r["ci"][0] for r in rs) - 0.012,
                    max(r["ci"][1] for r in rs) + 0.055)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(
        "Paired differences on the intersection where both methods are "
        "defined (n = 192 for every row).  Grey means the interval crosses "
        "zero: that is parity, not a win.\nNumbers beside each bar are the "
        "point estimate and the units ahead / behind.  Rows sorted by effect "
        "size within each panel.  clinical_grade = false.",
        fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=170)
    plt.close(fig)


README = ROOT / "README.md"
# Its own pair of markers rather than an entry in the block
# `tools/make_official_figures.py` owns. Two generators writing one block would
# make the figure numbering depend on which ran last, and the numbering is what
# the manuscript cites.
BEGIN = "<!-- BEGIN AUTOGENERATED: grand baseline figures -->"
END = "<!-- END AUTOGENERATED: grand baseline figures -->"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_readme(figs: dict) -> bool:
    """Put each caption under its image, verbatim.

    `tools/classify_artifacts.py` requires the caption in README.md to be
    byte-identical to the one recorded beside the image's digest, so this is
    generated from the same strings rather than copied by hand.
    """
    text = README.read_text()
    if BEGIN not in text or END not in text:
        print(f"README.md is missing the markers:\n  {BEGIN}\n  {END}")
        return False
    out = [BEGIN, ""]
    for i, (name, rec) in enumerate(figs.items(), 1):
        out += [f"![Three-baseline figure {i}](figures/{name})", "",
                f"**Three-baseline figure {i}.** {rec['caption']}", ""]
    out.append(END)
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    README.write_text(head + "\n".join(out) + tail)
    return True


def _pretty(m: str) -> str:
    """The display name as prose: ASCII, single line, LaTeX-safe."""
    return " ".join(PROSE.get(m, PRETTY.get(m, m)).split())


def _n_arch() -> int:
    """How many of our architectures have taken a number off this fold.

    Read from the ledger rather than typed, because it is a count of our own
    process and those are the numbers with the strongest incentive to drift.
    """
    if not LEDGER.exists():
        return 0
    return json.loads(LEDGER.read_text())["n_distinct_architectures_evaluated"]


def _caption_a(d: dict) -> str:
    s = d["summary"]
    best = max(OURS, key=lambda m: s[m]["mean_per_unit_roc_auc"])
    b, p = s[best], s["plmnn"]
    return (
        f"Per-residue detection on the official CryptoBench apo test fold "
        f"({b['n_units_scored']} single-chain units, MMseqs2 10% "
        f"cluster-disjoint), every method on the same residue universe in one "
        f"pass. Left, mean per-unit ROC-AUC; middle, mean per-unit PR-AUC; "
        f"right, the pooled residue read. The two per-unit metrics do not "
        f"order the methods the same way and both are shown: our best "
        f"architecture ({_pretty(best)}) leads pLM-NN on ROC-AUC "
        f"({b['mean_per_unit_roc_auc']:.4f} against "
        f"{p['mean_per_unit_roc_auc']:.4f}) and trails it on PR-AUC "
        f"({b['mean_per_unit_pr_auc']:.4f} against "
        f"{p['mean_per_unit_pr_auc']:.4f}). Neither margin is resolved; see "
        f"the paired panel. Against P2Rank 2.5.1 and PocketMiner the lead is "
        f"resolved on both metrics. Methods are drawn in the artifact's order "
        f"and not sorted by any metric shown, because sorting by the metric "
        f"displayed is how a figure comes to argue for the metric that "
        f"flatters it. {_n_arch()} of our architectures have been scored on "
        f"this fold, which is the multiplicity any single margin here should "
        f"be read against.")


def _caption_b(d: dict) -> str:
    s = d["summary"]
    best = max(OURS, key=lambda m: s[m]["mean_per_unit_roc_auc"])
    rows = []
    for base in BASE:
        r = d["paired"].get(f"{best}_minus_{base}", {}).get("per_unit_roc_auc")
        if r:
            rows.append(
                f"{_pretty(base)} {r['mean_delta']:+.4f} "
                f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                f"on {r['n_ahead']}/{r['n_paired']} units")
    return (
        f"Paired per-unit differences, {_pretty(best)} minus each "
        f"baseline, on the {d['summary'][best]['n_units_scored']} units where "
        f"both sides are defined -- every row here has the same n_paired, so "
        f"no difference in this panel mixes coverage into the comparison. "
        f"Whiskers are 95% bootstrap intervals over units "
        f"({d['n_boot']} resamples, seed {d['seed']}). A bar whose interval "
        f"crosses zero is drawn grey whatever the sign of its point estimate, "
        f"so a parity cannot be read as a win. On ROC-AUC: {'; '.join(rows)}. "
        f"The pLM-NN row is the one that does not resolve, in both directions "
        f"and on both metrics, and it is reported as parity rather than as a "
        f"lead. Rows are sorted by effect size.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--read", type=Path, default=READ)
    a = ap.parse_args(argv)
    d = json.loads(a.read.read_text())
    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    panel_a(d, OUT_A)
    panel_b(d, OUT_B)
    prov = {
        "schema": "geoaudit.grand_baseline_figure_provenance.v2",
        "clinical_grade": False,
        "reads_test_fold": True,
        # Keyed by bare filename, with the sha of the image and the caption that
        # must appear verbatim in README.md. v1 stored a list of prefixed paths
        # and no digest, so the gate could not tell whether the committed image
        # was the one this tool drew.
        "figures": {
            OUT_A.name: {"sha256": _sha(OUT_A), "caption": _caption_a(d)},
            OUT_B.name: {"sha256": _sha(OUT_B), "caption": _caption_b(d)},
        },
        # The gate re-hashes each source and fails when one has moved on, which
        # is what makes "the images show the current numbers" checkable.
        "sources": {str(p.relative_to(ROOT)): _sha(p)
                    for p in (a.read, LEDGER) if p.exists()},
        "drawn_from": str(a.read.relative_to(ROOT)),
        "drawn_from_seconds": d.get("seconds"),
        "n_units": d["summary"]["plmnn"]["n_units_scored"],
        "n_units_definition": (
            "units where every method returned a finite per-residue vector and "
            "ROC-AUC was defined; identical across all seven methods here, so "
            "every paired row has n_paired = this number"),
        "n_boot": d["n_boot"],
        "colour_rule": (
            "an interval crossing zero is drawn grey regardless of the sign of "
            "its point estimate, so that a parity cannot be read as a win"),
        "ordering_rule": (
            "panel A keeps the artifact's order; panel B sorts by effect size, "
            "which is stated in its caption"),
    }
    PROV.write_text(json.dumps(prov, indent=2) + "\n")
    print("WROTE", OUT_A, "\nWROTE", OUT_B, "\nWROTE", PROV)
    if not _write_readme(prov["figures"]):
        return 1
    print("WROTE", README)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
