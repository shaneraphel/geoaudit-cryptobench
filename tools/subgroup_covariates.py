#!/usr/bin/env python3
"""Five protein-level covariates for the fold, computed without touching a score.

A paired difference summarised over 192 chains says nothing about which chains.
The obvious question a reader asks next is whether the field wins on the easy
structures or the hard ones, on the shallow pockets or the deep, on the ones that
barely move between apo and holo or the ones that rearrange. That question needs
covariates, and covariates chosen after seeing which ones split the differences
favourably would be worthless.

So this file exists separately from the read, and it is the reason the subgroup
analysis can be preregistered at all. Every quantity here comes from the CryptoBench
deposit, the committed labels or the receptor coordinates. None comes from a
prediction, a score or a metric. Running this on a machine that had never seen a
detector output would give the same numbers.

The five, and why each is the one a reviewer named:

  ``prmsd``          the deposit's own apo/holo pocket RMSD. This is CryptoBench's
                     measure of how cryptic a site is, so it is the covariate the
                     benchmark itself would pick. Where a chain has several holo
                     partners the maximum is taken: a chain is as cryptic as its
                     most-rearranging pocket, and the labels are the union over
                     partners, so the maximum is the one that matches the label.
  ``n_true``         pocket size in labelled residues.
  ``positive_rate``  labelled residues as a fraction of the scored universe. Not a
                     restatement of ``n_true``: a 20-residue pocket on a 90-residue
                     chain and on a 600-residue chain are different problems.
  ``chain_length``   the scored universe. Also the covariate the preregistered
                     statistic already stratified on, where it bought nothing, so
                     it is carried here as a negative control.
  ``mean_bfactor``   mean B-factor over the labelled residues' atoms. The deposit's
                     receptors have their headers stripped, so resolution is not
                     recoverable and this is the structural-quality proxy that is:
                     a poorly ordered pocket is one where the coordinates every
                     geometric descriptor reads are least trustworthy.

Usage: PYTHONPATH=src:tools python3.12 tools/subgroup_covariates.py [--check]
"""
from __future__ import annotations

import argparse
import json
import statistics

from pocket_bench.paths import ROOT

MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OSF_TEST = ROOT / "data/cryptobench_apo/_osf/test.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"
OUT = ROOT / "results/official_fold/SUBGROUP_COVARIATES.json"
SCHEMA = "geoaudit.subgroup_covariates.v1"

# Ordered so the artifact, the plan and the read all iterate the same way; a
# subgroup analysis whose covariate order depended on dict insertion would give
# a different Bonferroni denominator on a different Python.
COVARIATES = ("prmsd", "n_true", "positive_rate", "chain_length",
              "mean_bfactor")


def _bfactors(path, resids: set[int]) -> list[float]:
    """Mean B-factor per labelled residue, from the receptor as it is scored.

    Reads the same file the detectors read, so a residue that is absent from the
    coordinates is absent here too rather than being imputed. The caller decides
    what to do with a chain whose labelled residues are all missing.
    """
    per: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            resi = int(line[22:26])
            b = float(line[60:66])
        except ValueError:
            continue
        if resi in resids:
            per.setdefault(resi, []).append(b)
    return [statistics.fmean(v) for v in per.values() if v]


def build() -> dict:
    man = json.loads(MANIFEST.read_text())
    osf = json.loads(OSF_TEST.read_text())
    # PER_STRUCTURE carries the scored universe and the label count that every
    # published metric was computed against. Taking them from here rather than
    # recounting means a covariate cannot silently disagree with the denominator
    # the AUCs used.
    per = {f"{r['pdb']}_{r['chain']}": r
           for r in json.loads(PER_STRUCTURE.read_text())}

    rows, no_prmsd = [], []
    for e in man["entries"]:
        uid = f"{e['pdb']}_{e['chain']}"
        s = per[uid]

        # The deposit keys apo records by apo PDB id and repeats one per holo
        # partner, so the chain has to be matched as well as the entry.
        pairs = [p for p in osf.get(e["pdb"], [])
                 if p.get("apo_chain") == e["chain"]]
        vals = [p["pRMSD"] for p in pairs if p.get("pRMSD") is not None]
        if not vals:
            no_prmsd.append(uid)

        lab = json.loads((ROOT / e["label_path"]).read_text())
        bf = _bfactors(ROOT / e["receptor_path"], set(lab["cryptic_residues"]))

        rows.append({
            "unit_id": uid,
            "cluster_id": e["cluster_id"],
            "prmsd": round(max(vals), 4) if vals else None,
            "n_holo_partners": len(pairs),
            "n_true": s["n_true"],
            "positive_rate": round(s["n_true"] / s["n_universe"], 6),
            "chain_length": s["n_universe"],
            "mean_bfactor": round(statistics.fmean(bf), 4) if bf else None,
            "n_labelled_residues_with_coordinates": len(bf),
        })
    rows.sort(key=lambda r: r["unit_id"])

    def _dist(name):
        v = sorted(r[name] for r in rows if r[name] is not None)
        n = len(v)
        # Tertile cuts by order statistic rather than by value, so a covariate
        # with heavy ties still splits into three groups of near-equal size and
        # the group sizes are known before the read rather than after it.
        lo, hi = v[n // 3], v[(2 * n) // 3]
        return {"n_defined": n, "min": v[0], "max": v[-1],
                "median": round(statistics.median(v), 6),
                "tertile_cuts": [lo, hi],
                "group_sizes": [sum(1 for x in v if x < lo),
                                sum(1 for x in v if lo <= x < hi),
                                sum(1 for x in v if x >= hi)]}

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "what_this_is": (
            "protein-level covariates for the official test fold, computed from "
            "the CryptoBench deposit, the committed labels and the receptor "
            "coordinates only"),
        "why_it_touches_no_score": (
            "the subgroups have to be definable before the differences inside "
            "them are looked at, or choosing a covariate would be choosing an "
            "answer. Nothing here reads a prediction, a score or a metric, so "
            "this file is not a reading of the fold in the sense the ledger "
            "counts"),
        "covariates": list(COVARIATES),
        "covariate_sources": {
            "prmsd": "max pRMSD over the unit's holo partners in "
                     "data/cryptobench_apo/_osf/test.json, the CryptoBench "
                     "deposit's own apo/holo pocket RMSD",
            "n_true": "labelled cryptic residues, from PER_STRUCTURE, which is "
                      "the count every published metric used",
            "positive_rate": "n_true divided by the scored universe",
            "chain_length": "the scored universe; carried as a negative control "
                            "because the preregistered statistic already "
                            "stratified on it and it bought nothing",
            "mean_bfactor": "mean over labelled residues of that residue's mean "
                            "atomic B-factor in the receptor as scored. The "
                            "deposit's receptors have stripped headers so "
                            "resolution is not recoverable; this is the "
                            "structural-quality proxy that is",
        },
        "n_units": len(rows),
        "units_without_a_deposited_prmsd": no_prmsd,
        "distributions": {c: _dist(c) for c in COVARIATES},
        "rows": rows,
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "no detector output is opened. The labels and coordinates are the "
            "benchmark's inputs, not its answers, and they were already read to "
            "score anything at all"),
    }


def _report(d: dict) -> None:
    print(f"{d['n_units']} units, {len(d['covariates'])} covariates")
    for c in d["covariates"]:
        s = d["distributions"][c]
        print(f"  {c:<14s} n={s['n_defined']:<4d} "
              f"[{s['min']:>8.3f}, {s['max']:>9.3f}] median {s['median']:>9.3f} "
              f"cuts {s['tertile_cuts'][0]:>8.3f}/{s['tertile_cuts'][1]:<8.3f} "
              f"groups {s['group_sizes']}")
    if d["units_without_a_deposited_prmsd"]:
        print(f"  no deposited pRMSD: {d['units_without_a_deposited_prmsd']}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") is not None:
        bad.append("covariates must not claim a read index")
    # Recomputed rather than trusted: the covariates are what the subgroup
    # boundaries are drawn from, so a drift here silently redraws the groups.
    live = build()
    if d.get("rows") != live["rows"]:
        moved = [a["unit_id"] for a, b in zip(d.get("rows", []), live["rows"])
                 if a != b]
        bad.append(f"{len(moved)} covariate rows no longer follow from the "
                   f"deposit and the labels: {moved[:5]}")
    if d.get("distributions") != live["distributions"]:
        bad.append("the tertile cuts have moved, which redraws every subgroup")
    for c in d.get("covariates", []):
        s = d["distributions"][c]
        if sum(s["group_sizes"]) != s["n_defined"]:
            bad.append(f"{c}: the three groups do not account for its units")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
