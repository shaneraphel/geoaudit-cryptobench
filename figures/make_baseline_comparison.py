#!/usr/bin/env python3
"""Render the benchmark + ablation comparison figure from BENCHMARK_SUMMARY.json.

Reproducible from committed data only (no local/private inputs). Method labels
are intentionally abstract (a "geometric manifold prior" / "exact-form filter"),
per project policy; underlying components are kept out of the public tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "BENCHMARK_SUMMARY.json").read_text())

labels = [
    "P2Rank",
    "fpocket",
    "random (bbox)",
    "Foliation\n(burial)",
    "Foliation\n+ manifold prior",
    "Foliation\n+ prior + exact-form",
]
keys = [
    "p2rank",
    "fpocket",
    "random_bbox",
    "foliation_burial",
    "foliation_plus_geometric_manifold_prior",
    "foliation_plus_manifold_prior_and_exact_form_filter",
]
hits = [S["methods_top1_hits"][k] for k in keys]
n = S["n_structures"]
oracle = S["oracle_ceiling_top1_hits"]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2c7fb8", "#7fcdbb", "#bdbdbd", "#fdae6b", "#f16913", "#d94801"]
bars = ax.bar(labels, hits, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(oracle, ls="--", color="crimson", lw=1.2,
           label=f"oracle ceiling on candidate pool = {oracle}/{n}")
ax.set_ylabel(f"Top-1 DCA <= 4 Angstrom hits (of {n})")
ax.set_ylim(0, n + 1)
ax.set_title("ESR1 pocket recovery on corrected chain-scoped labels\n"
             "(splits NOT cluster-disjoint -> pocket recovery, not superiority)")
for b, h in zip(bars, hits):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.15, str(h),
            ha="center", va="bottom", fontsize=10)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out = HERE / "fig_baseline_comparison.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
