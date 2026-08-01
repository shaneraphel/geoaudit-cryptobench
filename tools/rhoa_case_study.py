#!/usr/bin/env python3
"""Residue-level case study: RhoA (P61586, 9n4c_A) on the spent external set.

Why this file exists
--------------------
``WHERE_WE_BEAT_PLMNN.json`` names unit-level wins. A reviewer who asks for
something *they* do not see needs a residue. On RhoA the counting field's
within-unit ROC-AUC is 0.966 against pLM-NN's 0.803. Three labelled cryptic
residues (30, 32, 33) sit in our top fifth of the chain and in pLM-NN's bottom
half — residue 30 is our 84th percentile and pLM-NN's 2nd. That is not a mean; it
is a place.

Nothing here re-scores a model. Scores are the frozen Set A archives. The unit is
part of the spent confirmatory set, so this is a decomposition of a published
comparison, not a second confirmatory read. ``clinical_grade`` is false; nothing
claims affinity or a therapeutic decision.

Usage: PYTHONPATH=src:tools python3.12 tools/rhoa_case_study.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNIT = "9n4c_A"
LABEL = ROOT / "data/external/labels/9n4c_A_labels.json"
OURS = ROOT / "results/external/predictions/table_field.json"
PLM = ROOT / "results/baselines/PLMNN_EXTERNAL_SCORES.json"
P2R = ROOT / "results/external/predictions/p2rank.json"
SET = ROOT / "results/external/EXTERNAL_SET.json"
WHERE = ROOT / "results/external/WHERE_WE_BEAT_PLMNN.json"
OUT = ROOT / "results/external/RHOA_CASE_STUDY.json"
FIG = ROOT / "results/external/fig_rhoa_case_source.png"
SCHEMA = "geoaudit.rhoa_case_study.v1"

OURS_PCT, PLM_PCT = 0.80, 0.50  # fixed before names are emphasised


def _plm_scores() -> dict[int, float]:
    raw = json.loads(PLM.read_text())
    units = raw["units"]
    if isinstance(units, list):
        u = next(x for x in units if x.get("unit_id") == UNIT)
        s = u["scores"]
    else:
        u = units[UNIT]
        s = u.get("scores") or u["residue_scores"]
    return {int(k): float(v) for k, v in s.items()}


def _ranks(scores: dict[int, float], keys: list[int]) -> dict[int, float]:
    vals = np.array([scores[k] for k in keys])
    order = np.argsort(vals)
    out = {}
    for rank, i in enumerate(order, start=1):
        out[keys[i]] = rank / len(keys)
    return out


def build() -> dict:
    pos = {int(r) for r in json.loads(LABEL.read_text())["cryptic_residues"]}
    ours = {int(k): float(v) for k, v in
            json.loads(OURS.read_text())["units"][UNIT]["residue_scores"].items()}
    p2 = {int(k): float(v) for k, v in
          json.loads(P2R.read_text())["units"][UNIT]["residue_scores"].items()}
    plm = _plm_scores()
    keys = sorted(set(ours) & set(plm) & set(p2))
    ro, rp, r2 = _ranks(ours, keys), _ranks(plm, keys), _ranks(p2, keys)

    recovered = sorted(
        k for k in keys
        if k in pos and ro[k] >= OURS_PCT and rp[k] <= PLM_PCT)
    mirror = sorted(
        k for k in keys
        if k in pos and rp[k] >= OURS_PCT and ro[k] <= PLM_PCT)

    meta = next(u for u in json.loads(SET.read_text())["units"]
                if f"{u['apo_pdb']}_{u['apo_chain']}" == UNIT)
    where = next(r for r in json.loads(WHERE.read_text())
                 ["all_units_sorted_by_ours_minus_plmnn"] if r["unit"] == UNIT)

    rows = [{
        "resseq": k,
        "cryptic": k in pos,
        "ours_score": round(ours[k], 6),
        "plmnn_score": round(plm[k], 6),
        "p2rank_score": round(p2[k], 6),
        "ours_percentile": round(ro[k], 6),
        "plmnn_percentile": round(rp[k], 6),
        "p2rank_percentile": round(r2[k], 6),
        "ours_minus_plmnn_percentile": round(ro[k] - rp[k], 6),
    } for k in keys]

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "reads_test_fold": False,
        "why_this_is_not_a_second_confirmatory_read": (
            "scores are the frozen Set A archives; the unit-level comparison is "
            "already in EXTERNAL_READ / WHERE_WE_BEAT_PLMNN. This file names "
            "residues inside that unit"),
        "unit": UNIT,
        "protein": {
            "uniprot": "P61586",
            "name": "RhoA",
            "what_it_is": (
                "Ras-family GTPase; a named oncology signalling protein. Stating "
                "the accession is not a therapeutic claim"),
            "resolution": meta["resolution"],
            "released": meta["released"],
            "n_cryptic_residues": len(pos),
        },
        "unit_level_auc": {
            "table_field": where["table_field"],
            "plmnn": where["plmnn"],
            "p2rank": where["p2rank"],
            "pocketminer": where["pocketminer"],
            "ours_minus_plmnn": where["ours_minus_plmnn"],
        },
        "residue_rule": {
            "ours_percentile_at_least": OURS_PCT,
            "plmnn_percentile_at_most": PLM_PCT,
            "applies_only_to_labelled_cryptic_residues": True,
            "fixed_before_emphasising_names": True,
        },
        "recovered_cryptic_residues": recovered,
        "mirror_cryptic_residues": mirror,
        "why_the_mirror_is_here": (
            "large rank gaps exist in both directions; only listing ours would be "
            "a selection"),
        "headline_residue": {
            "resseq": 30,
            "ours_percentile": round(ro[30], 6),
            "plmnn_percentile": round(rp[30], 6),
            "why_named": (
                "labelled cryptic; counting field ranks it above 84% of the chain; "
                "pLM-NN ranks it above 1.7% of the chain"),
        },
        "n_residues_shared": len(keys),
        "per_residue": rows,
        "source_sha256": {
            "table_field": hashlib.sha256(OURS.read_bytes()).hexdigest(),
            "plmnn": hashlib.sha256(PLM.read_bytes()).hexdigest(),
            "label": hashlib.sha256(LABEL.read_bytes()).hexdigest(),
        },
        "what_this_cannot_show": [
            "that the counting field beats pLM-NN on average",
            "binding affinity, druggability, or any clinical property",
        ],
    }


def figure(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in d["per_residue"] if r["cryptic"]]
    xs = [r["plmnn_percentile"] for r in rows]
    ys = [r["ours_percentile"] for r in rows]
    ids = [r["resseq"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(xs, ys, c="#1b6b4a", s=36, zorder=3)
    ax.axhline(OURS_PCT, color="#1b6b4a", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(PLM_PCT, color="#8a2b2b", ls="--", lw=0.8, alpha=0.7)
    ax.fill_between([0, PLM_PCT], OURS_PCT, 1, color="#1b6b4a", alpha=0.08)
    for r in rows:
        if r["resseq"] in d["recovered_cryptic_residues"]:
            ax.annotate(str(r["resseq"]),
                        (r["plmnn_percentile"], r["ours_percentile"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=8, color="#1b6b4a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("pLM-NN within-chain percentile")
    ax.set_ylabel("counting field within-chain percentile")
    ax.set_title("RhoA (P61586, 9n4c_A): 26 cryptic residues\n"
                 f"shaded: ours≥{OURS_PCT:.0%}, pLM≤{PLM_PCT:.0%}  —  "
                 f"residues {d['recovered_cryptic_residues']}")
    ax.set_aspect("equal")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        d = json.loads(OUT.read_text())
        print(f"RhoA {d['unit']}: recovered {d['recovered_cryptic_residues']}, "
              f"mirror {d['mirror_cryptic_residues']}")
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    figure(d)
    print(f"RhoA {UNIT}: unit Δ={d['unit_level_auc']['ours_minus_plmnn']:+.3f}")
    print(f"  recovered cryptic residues (ours high, pLM low): "
          f"{d['recovered_cryptic_residues']}")
    print(f"  mirror: {d['mirror_cryptic_residues']}")
    print(f"  headline residue 30: ours "
          f"{d['headline_residue']['ours_percentile']:.1%} vs pLM "
          f"{d['headline_residue']['plmnn_percentile']:.1%}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
