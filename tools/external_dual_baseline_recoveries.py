#!/usr/bin/env python3
"""Residues where the counting field recovers a cryptic site both baselines miss.

Rule fixed before naming residues (stricter than the RhoA / pLM-only rule):

    labelled cryptic,
    ours ≥ 80th percentile of the chain,
    pLM-NN ≤ 50th percentile of the chain,
    P2Rank ≤ 50th percentile of the chain.

Mirror: labelled cryptic, pLM-NN and P2Rank both ≥ 80th, ours ≤ 50th.

This is a decomposition of the spent Set A archives, not a second confirmatory
read. The confirmatory mean against pLM-NN remains −0.0340. Residues that survive
here are the content behind "we have, they do not" when *both* published
baselines are the "they".

Usage: PYTHONPATH=src:tools python3.12 tools/external_dual_baseline_recoveries.py
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
OUT = ROOT / "results/external/DUAL_BASELINE_RECOVERIES.json"
FIG = ROOT / "results/external/fig_dual_baseline_recoveries_source.png"
FIGDIR = ROOT / "figures"
SCHEMA = "geoaudit.dual_baseline_recoveries.v1"

OURS_HI, BASE_LO = 0.80, 0.50


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
        ro, rp, rr = _ranks(o, keys), _ranks(p, keys), _ranks(r, keys)
        rec = sorted(k for k in keys
                     if k in pos and ro[k] >= OURS_HI
                     and rp[k] <= BASE_LO and rr[k] <= BASE_LO)
        mir = sorted(k for k in keys
                     if k in pos and rp[k] >= OURS_HI and rr[k] >= OURS_HI
                     and ro[k] <= BASE_LO)
        m = meta[uid]
        per_unit.append({
            "unit": uid, "uniprot": m["uniprot"], "cluster": m["cluster"],
            "n_cryptic": len(pos), "n_recovered": len(rec), "n_mirror": len(mir),
            "recovered": rec, "mirror": mir,
        })
        for k in rec:
            recovered.append({
                "unit": uid, "uniprot": m["uniprot"], "resseq": k,
                "ours_percentile": round(ro[k], 6),
                "plmnn_percentile": round(rp[k], 6),
                "p2rank_percentile": round(rr[k], 6),
                "gap_vs_worse_baseline": round(
                    ro[k] - max(rp[k], rr[k]), 6),
            })
        for k in mir:
            mirror.append({
                "unit": uid, "uniprot": m["uniprot"], "resseq": k,
                "ours_percentile": round(ro[k], 6),
                "plmnn_percentile": round(rp[k], 6),
                "p2rank_percentile": round(rr[k], 6),
            })

    recovered.sort(key=lambda x: -x["gap_vs_worse_baseline"])
    units_with = sum(1 for u in per_unit if u["n_recovered"])
    units_mir = sum(1 for u in per_unit if u["n_mirror"])
    named = sorted({r["uniprot"] for r in recovered
                    if r["uniprot"] in ("P61586", "P63000", "P10721", "P14618")})
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "reads_test_fold": False,
        "why_this_is_not_a_second_confirmatory_read": (
            "frozen Set A scores only; confirmatory means unchanged "
            "(−0.0340 vs pLM-NN, +0.0443 vs P2Rank). This lists labelled cryptic "
            "residues the counting field ranks high and *both* published "
            "baselines rank low"),
        "rule": {
            "ours_percentile_at_least": OURS_HI,
            "plmnn_percentile_at_most": BASE_LO,
            "p2rank_percentile_at_most": BASE_LO,
            "labelled_cryptic_only": True,
            "stricter_than": "results/external/RESIDUE_RECOVERIES_VS_PLMNN.json",
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
        "named_accessions_with_a_recovery": named,
        "what_those_are": {
            "P61586": "RhoA", "P63000": "RAC1",
            "P10721": "KIT", "P14618": "PKM",
        },
        "top_recoveries_by_gap": recovered[:40],
        "top_mirrors": mirror[:40],
        "per_unit": per_unit,
        "all_recovered": recovered,
        "all_mirror": mirror,
        "what_this_cannot_show": [
            "that the counting field beats either baseline on average",
            "affinity, druggability, or anything clinical",
        ],
        "source_sha256": {
            "table_field": hashlib.sha256(OURS.read_bytes()).hexdigest(),
            "plmnn": hashlib.sha256(PLM.read_bytes()).hexdigest(),
            "p2rank": hashlib.sha256(P2R.read_bytes()).hexdigest(),
        },
    }


def figure(d: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = d["counts"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = ["recovered\n(both miss)", "mirror\n(both hit)",
              "units with\nrecovery", "units with\nmirror"]
    vals = [c["n_recovered_residues"], c["n_mirror_residues"],
            c["n_units_with_a_recovery"], c["n_units_with_a_mirror"]]
    ax.bar(labels, vals, color=["#1b6b4a", "#8a2b2b", "#1b6b4a", "#8a2b2b"],
           width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title("Set A: counting field recovers cryptic residues\n"
                 "both pLM-NN and P2Rank rank ≤50%  |  mean deficits unchanged")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    out = FIGDIR / "fig_dual_baseline_recoveries.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


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
    fig = figure(d)
    c = d["counts"]
    print("\ndual-baseline recoveries on spent Set A")
    print(f"  recovered (both miss) {c['n_recovered_residues']} across "
          f"{c['n_units_with_a_recovery']} units")
    print(f"  mirror (both hit)     {c['n_mirror_residues']} across "
          f"{c['n_units_with_a_mirror']} units")
    print(f"  asymmetry (residues)  {c['asymmetry_residues']:+d}")
    for r in d["top_recoveries_by_gap"][:12]:
        print(f"    {r['unit']} res {r['resseq']}  ours {r['ours_percentile']:.1%}  "
              f"plm {r['plmnn_percentile']:.1%}  p2 {r['p2rank_percentile']:.1%}  "
              f"{r['uniprot']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {fig.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
