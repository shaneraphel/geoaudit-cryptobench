#!/usr/bin/env python3
"""Residue-level recoveries against pLM-NN on every spent Set A unit.

The RhoA case (``RHOA_CASE_STUDY.json``) fixed a rule before naming residues:

    labelled cryptic, ours ≥ 80th percentile of the chain, pLM-NN ≤ 50th.

This file applies that same rule to all 57 units, reports the mirror, and lists
every recovered residue with its UniProt accession. Nothing is re-scored. The
unit-level mean remains the confirmatory −0.0340; this is the residue content
behind the 27/30 split, not a reversal of it.

Usage: PYTHONPATH=src:tools python3.12 tools/external_residue_recoveries.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/external/labels"
OURS = ROOT / "results/external/predictions/table_field.json"
PLM = ROOT / "results/baselines/PLMNN_EXTERNAL_SCORES.json"
P2R = ROOT / "results/external/predictions/p2rank.json"
SET = ROOT / "results/external/EXTERNAL_SET.json"
OUT = ROOT / "results/external/RESIDUE_RECOVERIES_VS_PLMNN.json"
FIG = ROOT / "results/external/fig_residue_recoveries_source.png"
SCHEMA = "geoaudit.residue_recoveries_vs_plmnn.v1"

OURS_PCT, PLM_PCT = 0.80, 0.50


def _plm_all() -> dict[str, dict[int, float]]:
    raw = json.loads(PLM.read_text())
    units = raw["units"]
    out = {}
    if isinstance(units, list):
        for u in units:
            out[u["unit_id"]] = {int(k): float(v) for k, v in u["scores"].items()}
    else:
        for uid, u in units.items():
            s = u.get("scores") or u["residue_scores"]
            out[uid] = {int(k): float(v) for k, v in s.items()}
    return out


def _ranks(scores: dict[int, float], keys: list[int]) -> dict[int, float]:
    vals = np.array([scores[k] for k in keys])
    order = np.argsort(vals)
    return {keys[i]: (rank + 1) / len(keys) for rank, i in enumerate(order)}


def build() -> dict:
    ours_u = json.loads(OURS.read_text())["units"]
    p2_u = json.loads(P2R.read_text())["units"]
    plm_u = _plm_all()
    meta = {f"{u['apo_pdb']}_{u['apo_chain']}": u
            for u in json.loads(SET.read_text())["units"]}

    recovered, mirror, per_unit = [], [], []
    for uid in sorted(ours_u):
        lab = LABELS / f"{uid}_labels.json"
        if not lab.exists():
            continue
        pos = {int(r) for r in json.loads(lab.read_text())["cryptic_residues"]}
        o = {int(k): float(v) for k, v in ours_u[uid]["residue_scores"].items()}
        p = plm_u.get(uid) or {}
        r = {int(k): float(v) for k, v in
             (p2_u.get(uid) or {}).get("residue_scores", {}).items()}
        keys = sorted(set(o) & set(p) & set(r))
        if not keys or not (pos & set(keys)):
            continue
        ro, rp = _ranks(o, keys), _ranks(p, keys)
        rec = sorted(k for k in keys
                     if k in pos and ro[k] >= OURS_PCT and rp[k] <= PLM_PCT)
        mir = sorted(k for k in keys
                     if k in pos and rp[k] >= OURS_PCT and ro[k] <= PLM_PCT)
        m = meta[uid]
        per_unit.append({
            "unit": uid, "uniprot": m["uniprot"], "cluster": m["cluster"],
            "n_cryptic": len(pos), "n_recovered": len(rec), "n_mirror": len(mir),
            "recovered": rec, "mirror": mir,
            "ours_auc_proxy_mean_top": None,
        })
        for k in rec:
            recovered.append({
                "unit": uid, "uniprot": m["uniprot"], "resseq": k,
                "ours_percentile": round(ro[k], 6),
                "plmnn_percentile": round(rp[k], 6),
                "gap": round(ro[k] - rp[k], 6),
            })
        for k in mir:
            mirror.append({
                "unit": uid, "uniprot": m["uniprot"], "resseq": k,
                "ours_percentile": round(ro[k], 6),
                "plmnn_percentile": round(rp[k], 6),
                "gap": round(rp[k] - ro[k], 6),
            })

    recovered.sort(key=lambda x: -x["gap"])
    mirror.sort(key=lambda x: -x["gap"])
    units_with = sum(1 for u in per_unit if u["n_recovered"])
    units_mir = sum(1 for u in per_unit if u["n_mirror"])

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "reads_test_fold": False,
        "why_this_is_not_a_second_confirmatory_read": (
            "frozen Set A scores only; the confirmatory mean −0.0340 is unchanged. "
            "This lists labelled cryptic residues the counting field ranks high and "
            "pLM-NN ranks low, under the rule fixed in RHOA_CASE_STUDY.json"),
        "rule": {
            "ours_percentile_at_least": OURS_PCT,
            "plmnn_percentile_at_most": PLM_PCT,
            "labelled_cryptic_only": True,
            "source_of_the_rule": "results/external/RHOA_CASE_STUDY.json",
        },
        "counts": {
            "n_units": len(per_unit),
            "n_recovered_residues": len(recovered),
            "n_mirror_residues": len(mirror),
            "n_units_with_a_recovery": units_with,
            "n_units_with_a_mirror": units_mir,
            "asymmetry_residues": len(recovered) - len(mirror),
            "asymmetry_units": units_with - units_mir,
        },
        "named_oncology_accessions_with_a_recovery": sorted({
            r["uniprot"] for r in recovered
            if r["uniprot"] in ("P61586", "P63000")
        }),
        "what_those_are": {
            "P61586": "RhoA", "P63000": "RAC1",
        },
        "top_recoveries_by_gap": recovered[:40],
        "top_mirrors_by_gap": mirror[:40],
        "per_unit": per_unit,
        "all_recovered": recovered,
        "all_mirror": mirror,
        "what_this_cannot_show": [
            "that the counting field beats pLM-NN on average",
            "affinity, druggability, or anything clinical",
        ],
        "source_sha256": {
            "table_field": hashlib.sha256(OURS.read_bytes()).hexdigest(),
            "plmnn": hashlib.sha256(PLM.read_bytes()).hexdigest(),
        },
    }


def figure(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = d["counts"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = ["recovered\nresidues", "mirror\nresidues",
              "units with\nrecovery", "units with\nmirror"]
    vals = [c["n_recovered_residues"], c["n_mirror_residues"],
            c["n_units_with_a_recovery"], c["n_units_with_a_mirror"]]
    colors = ["#1b6b4a", "#8a2b2b", "#1b6b4a", "#8a2b2b"]
    ax.bar(labels, vals, color=colors, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title("Set A residue recoveries vs pLM-NN\n"
                 f"rule: cryptic ∩ ours≥{OURS_PCT:.0%} ∩ pLM≤{PLM_PCT:.0%}  |  "
                 f"mean unit deficit still −0.034")
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
        print(d["counts"])
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    figure(d)
    c = d["counts"]
    print(f"\nresidue recoveries on Set A (same rule as RhoA case)")
    print(f"  recovered residues {c['n_recovered_residues']} across "
          f"{c['n_units_with_a_recovery']} units")
    print(f"  mirror residues    {c['n_mirror_residues']} across "
          f"{c['n_units_with_a_mirror']} units")
    print(f"  asymmetry (residues) {c['asymmetry_residues']:+d}")
    for r in d["top_recoveries_by_gap"][:8]:
        print(f"    {r['unit']} res {r['resseq']}  "
              f"ours {r['ours_percentile']:.1%}  plm {r['plmnn_percentile']:.1%}  "
              f"Δpct {r['gap']:+.2f}  {r['uniprot']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
