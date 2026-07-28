#!/usr/bin/env python3
"""Four structures that show what the aggregate numbers are made of, chosen by rule.

A fold mean says one method is a little ahead. It does not say whether that is
the same structures done slightly better or different structures done very
differently, and on this benchmark it is emphatically the second. So four cases
are pulled out: one where both methods find the site, one where each finds it
and the other does not, and one where neither does.

The last of those is not a courtesy. Of the \\N{DIGIT ONE}92 evaluation units,
most are missed by both methods at any threshold worth calling a hit, and a case
study that showed only the three interesting outcomes would misrepresent the
benchmark as solved. The counts are reported beside the cases for that reason.

Chosen by rule, not by eye, because a hand-picked example is an illustration of
the author rather than of the method:

  locate       per-residue F1 >= 0.5 on the labelled cryptic residues. The
               harmonic mean of precision and recall at half is a low bar; it is
               stated rather than tuned, and every count below moves with it.
  both         both locate; the case is the one maximising the smaller of the
               two F1 values, so it is the clearest joint success rather than
               the largest.
  ours only    we locate, P2Rank does not; the case is the largest F1 margin.
  theirs only  the mirror image, same rule.
  neither      neither locates; the case is the one with the largest labelled
               pocket, because a small pocket missed is a less interesting
               failure than a large one missed.

Every metric here is recomputed from the committed labels and the raw
per-residue output, never read from the frozen telemetry, so a case study cannot
quietly disagree with the tables. ``--check`` re-derives and fails on drift.

What each case is then asked, structurally, is not "what was the F1" but "where
did the method put its mass": the distance from the centre of its positive calls
to the centre of the labelled pocket, and how buried those calls are compared
with the pocket. A method that fails by calling the surface and one that fails
by finding a different buried cavity are different failures, and the F1 alone
cannot tell them apart.

Usage:
  PYTHONPATH=src python3.12 tools/select_case_studies.py
  PYTHONPATH=src python3.12 tools/select_case_studies.py --check
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

PREDS = ROOT / "results/cryptobench_official/predictions"
LABELS = ROOT / "data/cryptobench_apo/official_labels"
RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"
OUT = ROOT / "results/official_fold/CASE_STUDIES.json"

OURS, BASELINE = "table_field", "p2rank"
LOCATE_F1 = 0.5
BURIAL_RADIUS = 10.0


def f1_of(positive: set[int], truth: set[int]) -> float:
    tp = len(positive & truth)
    if not positive or not truth or tp == 0:
        return 0.0
    precision, recall = tp / len(positive), tp / len(truth)
    return 2 * precision * recall / (precision + recall)


def receptors_available() -> bool:
    """The bulk PDBs are not committed; everything spatial here needs them.

    Labels and raw per-residue output are in the tree, so the F1 half of this
    file re-derives anywhere, including in CI. The geometry and the burial
    analysis cannot, and a gate that demanded them would be permanently red on
    every machine that has not fetched the receptors. So they are computed
    where the coordinates exist and declared absent where they do not, rather
    than being quietly dropped.
    """
    return RECEPTORS.is_dir() and any(RECEPTORS.glob("*_receptor.pdb"))


def ca_coords(unit: str) -> dict[int, np.ndarray]:
    """One point per residue: the C-alpha, which is what 'where' means here."""
    path = RECEPTORS / f"{unit}_receptor.pdb"
    out: dict[int, np.ndarray] = {}
    if not path.is_file():
        return out
    for atom in parse_pdb_atoms(path.read_text(errors="ignore")):
        if atom["record"] == "ATOM" and atom["name"] == "CA":
            out.setdefault(atom["resseq"],
                           np.array([atom["x"], atom["y"], atom["z"]]))
    return out


def burial(coords: dict[int, np.ndarray]) -> dict[int, int]:
    """C-alpha neighbours within 10 A: a coordination count, not an SASA.

    Deliberately crude and dependency-free. It is used only to say whether a
    method's calls sit where the protein is dense or where it is not, and a
    proper solvent-accessible surface would not change that reading.
    """
    keys = sorted(coords)
    xyz = np.array([coords[k] for k in keys])
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    return dict(zip(keys, (d2 <= BURIAL_RADIUS ** 2).sum(1) - 1))


def geometry(unit: str, truth: set[int], calls: dict[str, set[int]]) -> dict:
    """Where each method put its positive calls, relative to the labelled site."""
    coords = ca_coords(unit)
    known = set(coords)
    pocket = sorted(truth & known)
    if not pocket:
        return {}
    dense = burial(coords)
    centre = np.mean([coords[r] for r in pocket], axis=0)
    out = {
        "n_residues_with_ca": len(known),
        "pocket_burial_mean": float(np.mean([dense[r] for r in pocket])),
        "chain_burial_mean": float(np.mean(list(dense.values()))),
        "methods": {},
    }
    for name, called in calls.items():
        hit = sorted(called & known)
        if not hit:
            out["methods"][name] = {"n_called": 0}
            continue
        centroid = np.mean([coords[r] for r in hit], axis=0)
        out["methods"][name] = {
            "n_called": len(hit),
            "centroid_offset_angstrom": float(
                np.linalg.norm(centroid - centre)),
            "fraction_within_8a_of_pocket": float(np.mean([
                min(float(np.linalg.norm(coords[r] - coords[p]))
                    for p in pocket) <= 8.0 for r in hit])),
            "called_burial_mean": float(np.mean([dense[r] for r in hit])),
        }
    return out


def score_all() -> list[dict]:
    """Every unit both methods scored, with F1 recomputed from raw output."""
    preds = {m: json.loads((PREDS / f"{m}.json").read_text())["units"]
             for m in (OURS, BASELINE)}
    rows = []
    for unit in sorted(preds[OURS]):
        a, b = preds[OURS][unit], preds[BASELINE].get(unit)
        if b is None or a["status"] != "OK" or b["status"] != "OK":
            continue
        label = json.loads((LABELS / f"{unit}_labels.json").read_text())
        universe = {int(k) for k in a["residue_scores"]}
        truth = set(label["cryptic_residues"]) & universe
        calls = {OURS: set(a["residue_positive"]) & universe,
                 BASELINE: set(b["residue_positive"]) & universe}
        rows.append({
            "unit_id": unit,
            "pdb_id": label["pdb_id"],
            "chain": label["chain"],
            "n_universe": len(universe),
            "n_cryptic": len(truth),
            "f1": {OURS: f1_of(calls[OURS], truth),
                   BASELINE: f1_of(calls[BASELINE], truth)},
            "_truth": truth,
            "_calls": calls,
        })
    return rows


def choose(rows: list[dict]) -> list[dict]:
    ours = [r for r in rows if r["f1"][OURS] >= LOCATE_F1]
    theirs = [r for r in rows if r["f1"][BASELINE] >= LOCATE_F1]
    both = [r for r in ours if r in theirs]
    only_ours = [r for r in ours if r not in theirs]
    only_theirs = [r for r in theirs if r not in ours]
    neither = [r for r in rows
               if r not in ours and r not in theirs]

    picks = []
    if both:
        picks.append(("both_locate",
                      "both methods locate the site",
                      max(both, key=lambda r: min(r["f1"].values()))))
    if only_ours:
        picks.append(("table_field_only",
                      "the table field locates the site and P2Rank does not",
                      max(only_ours,
                          key=lambda r: r["f1"][OURS] - r["f1"][BASELINE])))
    if only_theirs:
        picks.append(("p2rank_only",
                      "P2Rank locates the site and the table field does not",
                      max(only_theirs,
                          key=lambda r: r["f1"][BASELINE] - r["f1"][OURS])))
    if neither:
        picks.append(("neither_locates",
                      "neither method locates the site, which is the "
                      "commonest outcome on this fold",
                      max(neither, key=lambda r: r["n_cryptic"])))

    out = []
    for key, why, row in picks:
        case = {k: v for k, v in row.items() if not k.startswith("_")}
        case["case"] = key
        case["why_this_one"] = why
        case["geometry"] = geometry(row["unit_id"], row["_truth"], row["_calls"])
        case["called_residues"] = {m: sorted(s) for m, s in row["_calls"].items()}
        case["cryptic_residues"] = sorted(row["_truth"])
        out.append(case)
    return out


def outcome_of(row: dict) -> str:
    ours = row["f1"][OURS] >= LOCATE_F1
    theirs = row["f1"][BASELINE] >= LOCATE_F1
    return ("both_locate" if ours and theirs else "table_field_only" if ours
            else "p2rank_only" if theirs else "neither_locates")


def burial_analysis(rows: list[dict]) -> dict:
    """Why the failures are the failures, measured over all of them.

    The single failing case turned out not to be a one-off. Grouping the fold by
    outcome shows the sites both methods find sitting well above their chain's
    mean coordination, the sites neither finds sitting barely above it, and both
    methods calling buried residues either way. That is one shared prior, not
    two methods disagreeing: it pays where the cryptic site happens to be
    buried, and both fail together where it is not. It is the most useful thing
    in this file, and it is the opposite of a result about which method wins.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(outcome_of(row), []).append(row)

    by_outcome, by_method = [], []
    for name in ("both_locate", "table_field_only", "p2rank_only",
                 "neither_locates"):
        group = groups.get(name) or []
        pockets, chains = [], []
        calls: dict[str, list[float]] = {OURS: [], BASELINE: []}
        for row in group:
            coords = ca_coords(row["unit_id"])
            dense = burial(coords)
            pocket = [dense[r] for r in row["_truth"] if r in dense]
            if not pocket:
                continue
            chain_mean = float(np.mean(list(dense.values())))
            pockets.append(float(np.mean(pocket)))
            chains.append(chain_mean)
            for method, called in row["_calls"].items():
                hit = [dense[r] for r in called if r in dense]
                if hit:
                    calls[method].append(float(np.mean(hit)) - chain_mean)
        if not pockets:
            continue
        by_outcome.append({
            "outcome": name,
            "n_units": len(pockets),
            "pocket_burial_mean": float(np.mean(pockets)),
            "chain_burial_mean": float(np.mean(chains)),
            "pocket_excess_over_chain": float(
                np.mean(np.array(pockets) - np.array(chains))),
        })
        for method, values in calls.items():
            if values:
                by_method.append({
                    "outcome": name, "method": method, "n_units": len(values),
                    "called_excess_over_chain": float(np.mean(values)),
                })

    return {
        "definition": "C-alpha neighbours within "
                      f"{BURIAL_RADIUS:g} A, excluding self",
        "by_outcome": by_outcome,
        "calls_by_method": by_method,
        "reading": "the labelled sites both methods find are well above their "
                   "chain's mean coordination and the sites neither finds are "
                   "barely above it, while both methods call buried residues "
                   "whatever the outcome; the shared failure mode is a burial "
                   "prior that the benchmark's less buried cryptic sites do "
                   "not follow",
    }


def build() -> dict:
    rows = score_all()
    ours = sum(1 for r in rows if r["f1"][OURS] >= LOCATE_F1)
    theirs = sum(1 for r in rows if r["f1"][BASELINE] >= LOCATE_F1)
    both = sum(1 for r in rows
               if r["f1"][OURS] >= LOCATE_F1 and r["f1"][BASELINE] >= LOCATE_F1)
    return {
        "schema": "geoaudit.case_studies.v1",
        "clinical_grade": False,
        "why": "a fold mean cannot say whether one method is slightly better "
               "everywhere or very different somewhere; these cases are chosen "
               "by a stated rule so that they illustrate the method rather "
               "than the author",
        "locate_threshold_f1": LOCATE_F1,
        "metric_source": "recomputed from data/cryptobench_apo/official_labels "
                         "and results/cryptobench_official/predictions, not "
                         "read from the frozen telemetry",
        "population": {
            "n_units": len(rows),
            "n_located_by_table_field": ours,
            "n_located_by_p2rank": theirs,
            "n_located_by_both": both,
            "n_located_by_neither": len(rows) - ours - theirs + both,
        },
        "receptors_available": receptors_available(),
        "burial": burial_analysis(rows) if receptors_available() else None,
        "cases": choose(rows),
    }


def _report(rec: dict) -> None:
    p = rec["population"]
    print(f"{p['n_units']} units at F1 >= {rec['locate_threshold_f1']}: "
          f"both {p['n_located_by_both']}, table field only "
          f"{p['n_located_by_table_field'] - p['n_located_by_both']}, "
          f"P2Rank only {p['n_located_by_p2rank'] - p['n_located_by_both']}, "
          f"neither {p['n_located_by_neither']}")
    if not rec.get("receptors_available"):
        print("receptors absent: the cases and their F1 are complete, the "
              "geometry and burial blocks are not computed")
        for c in rec["cases"]:
            print(f"  {c['case']:17s} {c['unit_id']}  F1 "
                  f"{c['f1'][OURS]:.3f} / {c['f1'][BASELINE]:.3f}")
        return

    print("\nburial, C-alpha neighbours within 10 A above the chain mean:")
    calls = {(m["outcome"], m["method"]): m["called_excess_over_chain"]
             for m in rec["burial"]["calls_by_method"]}
    for o in rec["burial"]["by_outcome"]:
        name = o["outcome"]
        print(f"  {name:17s} n={o['n_units']:3d}  pocket "
              f"{o['pocket_excess_over_chain']:+.2f}   calls "
              f"{OURS} {calls.get((name, OURS), float('nan')):+.2f}, "
              f"{BASELINE} {calls.get((name, BASELINE), float('nan')):+.2f}")

    for c in rec["cases"]:
        g = c["geometry"]["methods"]
        print(f"\n  {c['case']:17s} {c['unit_id']}  "
              f"{c['n_cryptic']} cryptic of {c['n_universe']} residues")
        print(f"    F1 table field {c['f1'][OURS]:.3f}, "
              f"P2Rank {c['f1'][BASELINE]:.3f}")
        for m in (OURS, BASELINE):
            d = g.get(m) or {}
            if not d.get("n_called"):
                print(f"    {m:12s} called nothing")
                continue
            print(f"    {m:12s} {d['n_called']:3d} calls, centroid "
                  f"{d['centroid_offset_angstrom']:5.1f} A from the pocket, "
                  f"{d['fraction_within_8a_of_pocket'] * 100:3.0f}% within 8 A, "
                  f"burial {d['called_burial_mean']:.1f} "
                  f"(pocket {c['geometry']['pocket_burial_mean']:.1f}, "
                  f"chain {c['geometry']['chain_burial_mean']:.1f})")


def _same(a, b, tol=1e-9) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-derive from labels and raw output; fail on drift")
    args = ap.parse_args(argv)

    rec = build()
    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        old = json.loads(OUT.read_text())
        spatial = receptors_available()
        for field in ("population", "locate_threshold_f1"):
            if not _same(old.get(field), rec[field]):
                print(f"STALE {OUT.relative_to(ROOT)}: {field} no longer "
                      f"follows from the labels and the raw predictions")
                return 1
        # Which cases were chosen, and their F1, need no coordinates. Where the
        # receptors are absent the spatial blocks are not compared, because
        # this build could not have produced them.
        keys = ("case", "unit_id", "f1", "n_cryptic", "n_universe",
                "cryptic_residues", "called_residues")
        if spatial:
            keys += ("geometry",)
        for a, b in zip(old.get("cases") or [], rec["cases"]):
            for key in keys:
                if not _same(a.get(key), b.get(key)):
                    print(f"STALE {OUT.relative_to(ROOT)}: case "
                          f"{b['unit_id']} field {key} no longer follows from "
                          f"the labels and the raw predictions")
                    return 1
        if len(old.get("cases") or []) != len(rec["cases"]):
            print(f"STALE {OUT.relative_to(ROOT)}: a different number of "
                  f"cases is selected")
            return 1
        if spatial and not _same(old.get("burial"), rec["burial"]):
            print(f"STALE {OUT.relative_to(ROOT)}: the burial analysis no "
                  f"longer follows from the receptors")
            return 1
        note = ("" if spatial else "; receptors absent, so the geometry and "
                "burial blocks were not rechecked")
        print(f"{OUT.relative_to(ROOT)} re-derives from the raw "
              f"predictions{note}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, allow_nan=False) + "\n")
    _report(rec)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
