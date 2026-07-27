#!/usr/bin/env python3
"""Draw the official-fold comparison from the frozen artifacts, and nothing else.

The repository already carried a file called fig_baseline_comparison.png. It
plotted the 14-structure ESR1 pilot, on a split its own summary records as not
cluster-disjoint, with method labels ("geometric manifold prior", "exact-form
filter") that appear nowhere in the manuscript, showing every one of our
detectors at zero. A reader opening the repository saw that image and formed a
picture unrelated to the result. A wrong figure is worse than no figure, because
it is read faster than the text that would correct it.

These two are generated from the same frozen JSON the paper's macros read, so a
number in a figure cannot disagree with a number in a sentence.

Figure 1, the four metrics side by side with bootstrap intervals. Ordered by
ROC-AUC, P2Rank marked, so the eye gets the comparison the benchmark exists for
without a caption having to assert it.

Figure 2, the paired differences against P2Rank as a forest plot, one row per
metric, with the interval and the shared-structure count. This is the honest
figure: three intervals straddle zero and the drawing says so at a glance, where
a bar chart of point estimates would quietly imply an ordering the fold cannot
support. The ROC-AUC row also carries the standard error of a fold mean, because
a margin has to be read against the scale of the measurement rather than against
zero.

Usage: PYTHONPATH=src python3.12 tools/make_official_figures.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
SEEDS = ROOT / "results/official_fold/SEED_SENSITIVITY.json"
FIGDIR = ROOT / "figures"
PROVENANCE = ROOT / "results/official_fold/FIGURE_PROVENANCE.json"
SOURCES = (BOOT, LEDGER, TELEMETRY, SEEDS)

METRICS = [("residue_auc", "ROC-AUC"), ("residue_pr_auc", "PR-AUC"),
           ("residue_mcc", "MCC"), ("residue_f1", "F1")]
LABEL = {
    "table_field": "Table field (ours)",
    "algebraic_field_linear": "Algebraic field, fitted readout",
    "algebraic_field": "Algebraic field",
    "quaternary_lut_seq": "Quaternary LUT + sequence",
    "quaternary_lut": "Quaternary LUT",
    "geometric_foundation": "Rigid geometric",
    "sstar_pocket": "Anisotropic shear",
    "ultrametric_shear_oracle": "Ultrametric shear",
    "fstar_pocket": "Isotropic ablation",
    "random_bbox": "Chance",
    "p2rank": "P2Rank 2.5.1 (baseline)",
}


def _fold_se() -> float:
    rows = json.loads(TELEMETRY.read_text())["rows"]
    v = [r["residue_auc"] for r in rows
         if r["method"] == "table_field" and r.get("residue_auc") is not None]
    mu = sum(v) / len(v)
    var = sum((x - mu) ** 2 for x in v) / (len(v) - 1)
    return (var / len(v)) ** 0.5


def figure_metrics(boot: dict) -> Path:
    order = sorted(
        boot["metrics"]["residue_auc"]["per_method"].items(),
        key=lambda kv: -(kv[1]["point"] or 0))
    names = [m for m, _ in order]

    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        pm = boot["metrics"][key]["per_method"]
        ys = list(range(len(names)))[::-1]
        for y, m in zip(ys, names):
            rec = pm.get(m) or {}
            pt = rec.get("point")
            if pt is None:
                # MCC is undefined for a detector that calls nothing positive.
                # Saying so beats an absent bar the reader has to interpret.
                ax.text(0.02, y, "undefined", va="center", fontsize=8,
                        color="#888888", style="italic")
                continue
            lo, hi = rec.get("ci_low"), rec.get("ci_high")
            colour = ("#d94801" if m == "table_field"
                      else "#2c7fb8" if m == "p2rank" else "#bdbdbd")
            ax.barh(y, pt, color=colour, edgecolor="black", linewidth=0.5,
                    height=0.68)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color="black", lw=1.1)
                ax.plot([lo, lo], [y - 0.16, y + 0.16], color="black", lw=1.1)
                ax.plot([hi, hi], [y - 0.16, y + 0.16], color="black", lw=1.1)
            ax.text(pt, y + 0.34, f"{pt:.3f}", fontsize=7.5, ha="center")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1.0 if key == "residue_auc" else None)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(list(range(len(names)))[::-1])
    axes[0].set_yticklabels([LABEL.get(m, m) for m in names], fontsize=9)

    n = boot["n_structures"]
    fig.suptitle(
        f"Per-residue detection on the official CryptoBench apo test fold "
        f"({n} single-chain units, MMseqs2 10% cluster-disjoint)\n"
        f"bars are means over structures, whiskers are 95% bootstrap intervals "
        f"({boot['n_boot']:,} resamples, seed {boot['seed']})",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    out = FIGDIR / "fig_official_fold_metrics.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _seed_survival() -> dict[str, tuple[int, int]]:
    """How many resampling seeds each verdict survives.

    Without this the figure would colour MCC as resolved while the manuscript
    calls it unresolved, because its interval clears zero at the frozen seed and
    at few others. A figure that contradicts the text is read first and believed
    longest, so the drawing reads the same artifact the prose does.
    """
    if not SEEDS.exists():
        return {}
    d = json.loads(SEEDS.read_text())
    n = d.get("n_seeds")
    out = {}
    for block in ("per_metric", "metrics"):
        for k, v in (d.get(block) or {}).items():
            if isinstance(v, dict):
                sig = (v.get("n_excluding_zero")
                       if v.get("n_excluding_zero") is not None
                       else v.get("n_seeds_excluding_zero"))
                if sig is not None and n:
                    out[k] = (int(sig), int(n))
    return out


def figure_paired(boot: dict, led: dict) -> Path:
    se = _fold_se()
    survival = _seed_survival()
    rows = []
    for key, title in METRICS:
        d = (boot["metrics"][key].get("paired_vs_baseline") or {}).get("table_field")
        if d and d.get("delta_point") is not None:
            rows.append((key, title, d))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ys = list(range(len(rows)))[::-1]
    for y, (key, title, d) in zip(ys, rows):
        lo, hi, pt = d["delta_ci_low"], d["delta_ci_high"], d["delta_point"]
        clears = not d.get("crosses_zero", True)
        sig, n_seeds = survival.get(key, (None, None))
        # Reported as resolved only if the interval clears zero AND the verdict
        # does not depend on which resample happened to be drawn.
        robust = clears and (sig is None or n_seeds is None or sig == n_seeds)
        colour = "#d94801" if robust else "#8c8c8c"
        ax.plot([lo, hi], [y, y], color=colour, lw=2.4, solid_capstyle="butt")
        ax.plot(pt, y, "o", color=colour, ms=7, zorder=3)
        if robust:
            verdict = "resolved"
        elif clears:
            verdict = f"clears zero at {sig}/{n_seeds} seeds \u2014 unresolved"
        else:
            verdict = "contains zero"
        seeds = "" if sig is None else f"   seeds {sig}/{n_seeds}"
        ax.text(hi + 0.004, y,
                f"{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]   "
                f"n={d['n_paired_structures']}{seeds}   {verdict}",
                va="center", fontsize=8.8,
                color="black" if robust else "#555555")
    ax.axvline(0, color="black", lw=1.0)
    ax.axvspan(-se, se, color="#2c7fb8", alpha=0.10, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([t for _, t, _ in rows], fontsize=10)
    ax.set_xlabel("paired difference, table field minus P2Rank "
                  "(positive favours the table field)", fontsize=10)
    ax.set_xlim(-0.05, 0.19)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        "Paired differences against P2Rank, on the structures where both are defined",
        fontsize=12)
    fig.text(0.5, 0.925,
             f"shaded band is \u00b11 standard error of a fold mean ({se:.4f}); "
             f"{led['n_distinct_architectures_evaluated']} of our architectures "
             f"have been scored on this fold, so a margin inside it is not an "
             f"ordering",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGDIR / "fig_paired_vs_p2rank.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    FIGDIR.mkdir(exist_ok=True)
    boot = json.loads(BOOT.read_text())
    led = json.loads(LEDGER.read_text())
    images = [figure_metrics(boot), figure_paired(boot, led)]
    for out in images:
        print(f"wrote {out.relative_to(ROOT)} "
              f"({out.stat().st_size / 1e3:.0f} kB)")

    # Naming the generator is not enough: an image keeps its filename while the
    # numbers underneath it move, and a stale plot passes a name check silently.
    # Recording what each image was drawn from lets the gate fail on the source
    # digest, which is the drift that matters and does not need a plotting
    # library in CI to detect.
    PROVENANCE.write_text(json.dumps({
        "schema": "geoaudit.figure_provenance.v1",
        "clinical_grade": False,
        "purpose": "tie every committed image to the exact bytes it was drawn "
                   "from, so a figure cannot outlive the numbers it shows",
        "generator": "tools/make_official_figures.py",
        "sources": {str(p.relative_to(ROOT)): _sha(p)
                    for p in SOURCES if p.exists()},
        "figures": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size}
                    for p in images},
    }, indent=2) + "\n")
    print(f"wrote {PROVENANCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
