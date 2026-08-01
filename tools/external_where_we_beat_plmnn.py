#!/usr/bin/env python3
"""Name the Set A units where the counting field already beat pLM-NN.

This is not a new confirmatory read. ``EXTERNAL_READ.json`` already reports that
the counting field leads pLM-NN on 27 of 57 units and trails on 30, and that the
mean deficit is −0.0340. What that artifact does not carry is *which* units, or
what proteins they are. A reviewer who asks "show me a case you have and they do
not" cannot answer from the mean alone.

So this file opens the same frozen per-residue scores the confirmatory read used,
recomputes the same per-unit ROC-AUCs through the same harness, and lists every
unit with its four AUCs, the paired delta, and the UniProt accession the set
already records. Nothing is re-scored; nothing is selected by a new threshold.

Two summaries sit on top of that full table, both fixed before the names are
printed:

* **ahead**: ``ours > plmnn`` — the same count EXTERNAL_READ already published as
  27/30. This tool must reproduce that split exactly, or it is measuring a
  different quantity.
* **clear_margin**: ``ours − plmnn ≥ 0.10`` and ``ours ≥ 0.85``. Ten points is a
  round number above the half-width of the confirmatory CI on the mean deficit
  (±0.035), so a unit that clears it is not one the mean could have been moved
  by. The mirror — they clear the same bar against us — is reported beside it.

The proteins are real depositions released after CryptoBench's cutoff. Two of the
clear-margin units are named oncology GTPases (RhoA ``P61586``, RAC1 ``P63000``);
that is a fact about the UniProt accessions in the frozen set, not a claim that
anything here is clinical.

Usage: PYTHONPATH=src:tools python3.12 tools/external_where_we_beat_plmnn.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "results/external/EXTERNAL_SET.json"
READ = ROOT / "results/external/EXTERNAL_READ.json"
OUT = ROOT / "results/external/WHERE_WE_BEAT_PLMNN.json"
FIG = ROOT / "results/architecture_sweep/figures/fig_external_beat_plmnn.png"
SCHEMA = "geoaudit.where_we_beat_plmnn.v1"

CLEAR = 0.10
FLOOR = 0.85


def build() -> dict:
    import external_read as er

    shared = er._shared()
    auc = er._per_unit_auc(shared["units"])
    units = sorted(set(auc["table_field"]) & set(auc["plmnn"])
                   & set(auc["p2rank"]) & set(auc["pocketminer"]))
    meta = {f"{u['apo_pdb']}_{u['apo_chain']}": u
            for u in json.loads(SET.read_text())["units"]}
    read = json.loads(READ.read_text())
    published = read["co_primary"]["table_field_minus_plmnn"]

    rows = []
    for uid in units:
        m = meta[uid]
        o, p, r, k = (auc["table_field"][uid], auc["plmnn"][uid],
                      auc["p2rank"][uid], auc["pocketminer"][uid])
        rows.append({
            "unit": uid,
            "uniprot": m["uniprot"],
            "cluster": m["cluster"],
            "resolution": m["resolution"],
            "released": m["released"],
            "n_cryptic_residues": len(m["residues"]),
            "table_field": round(o, 6),
            "plmnn": round(p, 6),
            "p2rank": round(r, 6),
            "pocketminer": round(k, 6),
            "ours_minus_plmnn": round(o - p, 6),
            "ours_minus_p2rank": round(o - r, 6),
        })
    rows.sort(key=lambda r: -r["ours_minus_plmnn"])

    ahead = [r for r in rows if r["ours_minus_plmnn"] > 0]
    behind = [r for r in rows if r["ours_minus_plmnn"] < 0]
    clear = [r for r in rows
             if r["ours_minus_plmnn"] >= CLEAR and r["table_field"] >= FLOOR]
    clear_mirror = [r for r in rows
                    if (-r["ours_minus_plmnn"]) >= CLEAR and r["plmnn"] >= FLOOR]

    # Must reproduce the confirmatory read's per-unit split.
    if (len(ahead) != published["n_first_ahead"]
            or len(behind) != published["n_second_ahead"]):
        raise SystemExit(
            f"ahead/behind {len(ahead)}/{len(behind)} does not match the "
            f"confirmatory read's {published['n_first_ahead']}/"
            f"{published['n_second_ahead']}; this tool is measuring a different "
            f"quantity than the one it claims to name")

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_the_external_set": True,
        "why_this_is_not_a_second_confirmatory_read": (
            "no model is re-run and no threshold is re-fitted. The confirmatory "
            "read already published that 27 units favour us and 30 favour pLM-NN; "
            "this file names them and attaches the UniProt accessions the set "
            "already carries"),
        "source_scores": "the same frozen archives EXTERNAL_READ.json used",
        "reproduces_the_confirmatory_split": {
            "n_ahead": len(ahead),
            "n_behind": len(behind),
            "n_tied": len(rows) - len(ahead) - len(behind),
            "published_n_first_ahead": published["n_first_ahead"],
            "published_n_second_ahead": published["n_second_ahead"],
            "published_mean_delta": published["mean"],
        },
        "clear_margin_rule": {
            "ours_minus_plmnn_at_least": CLEAR,
            "ours_at_least": FLOOR,
            "why_point_one": (
                "a round number above the half-width of the confirmatory CI on the "
                "mean deficit (±0.035), so a unit that clears it is not one the "
                "mean alone could have been moved by"),
            "n_clear_margin": len(clear),
            "n_clear_margin_mirror": len(clear_mirror),
            "why_the_mirror_is_here": (
                "on any two methods of similar average accuracy, large wins exist "
                "in both directions; only the asymmetry, and the proteins, say "
                "something"),
        },
        "clear_margin_units": clear,
        "clear_margin_mirror_units": clear_mirror,
        "named_oncology_gtpases_in_the_clear_margin": [
            r for r in clear if r["uniprot"] in ("P61586", "P63000")
        ],
        "what_those_accessions_are": {
            "P61586": "RhoA, a Ras-family GTPase; PDB 9n4c chain A",
            "P63000": "RAC1, a Ras-family GTPase; PDB 9ifk chain A",
        },
        "what_this_cannot_show": [
            "that the counting field beats pLM-NN on average — the confirmatory "
            "read says it does not (−0.0340)",
            "anything about binding affinity, druggability, or a therapeutic "
            "decision",
            "that these pockets were unknown; the labels come from deposited "
            "holo partners under the recovered CryptoBench rule",
        ],
        "all_units_sorted_by_ours_minus_plmnn": rows,
        "external_read_sha256": hashlib.sha256(READ.read_bytes()).hexdigest(),
        "external_set_sha256": hashlib.sha256(SET.read_bytes()).hexdigest(),
    }


def figure(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = d["all_units_sorted_by_ours_minus_plmnn"]
    deltas = np.array([r["ours_minus_plmnn"] for r in rows])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#1b6b4a" if v >= CLEAR else ("#8a2b2b" if v <= -CLEAR else "#7a7a7a")
              for v in deltas]
    ax.barh(range(len(deltas)), deltas[::-1], color=colors[::-1], height=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(CLEAR, color="#1b6b4a", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(-CLEAR, color="#8a2b2b", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("per-unit ROC-AUC: counting field − pLM-NN  (Set A, n=57)")
    ax.set_yticks([])
    ax.set_title("Where the counting field already beat pLM-NN on the spent "
                 "external set\n"
                 f"green: clear margin (≥{CLEAR:.2f} and ours≥{FLOOR}); "
                 f"{d['clear_margin_rule']['n_clear_margin']} vs "
                 f"{d['clear_margin_rule']['n_clear_margin_mirror']} mirror  |  "
                 f"mean still −0.034")
    # Label the named GTPases and the worst loss.
    by = {r["unit"]: i for i, r in enumerate(rows)}
    for r in d["named_oncology_gtpases_in_the_clear_margin"]:
        i = by[r["unit"]]
        ax.annotate(f"{r['unit']} {d['what_those_accessions_are'][r['uniprot']].split(',')[0]}",
                    xy=(r["ours_minus_plmnn"], len(rows) - 1 - i),
                    xytext=(5, 0), textcoords="offset points", fontsize=8,
                    color="#1b6b4a", va="center")
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


def report(d: dict) -> None:
    c = d["clear_margin_rule"]
    print(f"\nSet A units vs pLM-NN (decomposition of the confirmatory read)")
    print(f"  ahead {d['reproduces_the_confirmatory_split']['n_ahead']} / "
          f"behind {d['reproduces_the_confirmatory_split']['n_behind']}  "
          f"(matches EXTERNAL_READ)")
    print(f"  clear margin (≥{CLEAR} and ours≥{FLOOR}): "
          f"{c['n_clear_margin']}  |  mirror {c['n_clear_margin_mirror']}")
    for r in d["clear_margin_units"]:
        name = d["what_those_accessions_are"].get(r["uniprot"], "")
        print(f"    {r['unit']:10s} Δ={r['ours_minus_plmnn']:+.3f}  "
              f"ours={r['table_field']:.3f} plm={r['plmnn']:.3f}  "
              f"{r['uniprot']} {name}")
    print(f"  mean is still the confirmatory −0.0340; this names the 27, "
          f"it does not reverse them")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        report(json.loads(OUT.read_text()))
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    figure(d)
    report(d)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
