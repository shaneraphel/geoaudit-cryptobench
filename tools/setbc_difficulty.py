#!/usr/bin/env python3
"""What the frozen sets look like, and one confound named before the read.

Why this has to be written now
------------------------------
``EXTERNAL_SET_DIFFICULTY.json`` found something about Set A that nobody had
looked for: three of four methods score higher on it than on the official fold,
and the size of the gain tracks how much context each method reads --- P2Rank,
which reads local surface geometry, moves +0.0035, while pLM-NN, which reads a
whole sequence, moves +0.0517. The explanation on offer was chain length: Set A's
units are 336 residues at the median against the official fold's 290, and a
longer chain is a resource a context-reading method can spend and a local one
cannot.

Sets B and C are cryo-EM, and cryo-EM structures are large assemblies. Their
median chain is **469 residues, 1.6 times the official fold's**, and their median
positive rate is **3.3 % against 6.2 %**. So the property that made Set A's
comparison awkward is *stronger* here, and it bears directly on the read this
repository is about to take.

**The confound, stated in full before any score is compared to any label.**
``geometry_field`` reads more context than ``table_field``: void topology is
computed from a chain's whole Delaunay tetrahedralisation and its convex hull, and
the three aggregations reach two steps out on the contact graph. If a longer chain
inflates the advantage of whichever method reads further, then
``geometry_field`` − ``table_field`` on these sets could come out *larger* than
the training fold's +0.0121 for a reason that has nothing to do with the four
families generalising.

That is not a reason to skip the read. It is a reason to write down, first, what
would distinguish the two readings:

* **P2Rank is the control.** It reads local surface geometry and moved +0.0035
  between the official fold and Set A. If it jumps on Sets B and C, they are
  simply easier and every method's gain is uninformative. If it stays flat while
  the context-readers rise again, the confound is live and the
  ``geometry_field`` − ``table_field`` difference has to be read with it.
* **The stratification the plan already pins.** Chain length and positive rate
  move together here, so the per-unit comparison split by pocket size --- already
  a secondary analysis in ``PREREGISTERED_SETBC.json`` --- is where a
  context-driven inflation would show up as an interaction rather than a level
  shift.
* **What no analysis here can fix.** 45 units cannot separate a context effect
  from a generalisation effect. If the read comes out positive, the honest claim
  is that the stack transfers *on sets of this shape*, and the shape is not the
  benchmark's. Saying so is cheaper now than conceding it later.

This tool reads label metadata and receptor coordinates. It opens no prediction,
compares nothing against a label, and must be committed before the read runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.setbc_difficulty.v1"
MANIFEST = ROOT / "data/external/setbc_manifest.json"
PER = ROOT / "results/external/SETBC_PER_STRUCTURE.json"
SETA = ROOT / "results/external/EXTERNAL_SET_DIFFICULTY.json"
OUT = ROOT / "results/external/SETBC_DIFFICULTY.json"


def _summary(rows: list[tuple[int, int]]) -> dict:
    nr = np.array([r[0] for r in rows], float)
    nc = np.array([r[1] for r in rows], float)
    pr = nc / np.maximum(nr, 1.0)
    return {
        "n_units": len(rows),
        "chain_length": {"median": float(np.median(nr)),
                         "mean": float(nr.mean()), "max": float(nr.max())},
        "cryptic_residues": {"median": float(np.median(nc)),
                             "mean": float(nc.mean())},
        "positive_rate": {"median": float(np.median(pr)),
                          "mean": float(pr.mean())},
        "share_under_ten_cryptic": float((nc < 10).mean()),
    }


def build(write: bool) -> int:
    per = {f"{r['pdb']}_{r['chain']}": r["n_universe"]
           for r in json.loads(PER.read_text())}
    entries = json.loads(MANIFEST.read_text())["entries"]
    by_set: dict[str, list[tuple[int, int]]] = {}
    for e in entries:
        uid = f"{e['pdb']}_{e['chain']}"
        by_set.setdefault(e["set"], []).append(
            (per[uid], e["n_cryptic_residues"]))
    pooled = [x for v in by_set.values() for x in v]

    a = json.loads(SETA.read_text())
    ref = a["label_geometry"]
    moved = a["achieved_accuracy"]["external_minus_official"]

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "why_this_declares_no_read": (
            "label metadata and receptor coordinates only. No prediction is "
            "opened and nothing is compared against a label. It exists to be "
            "committed before the read"),
        "question": (
            "what shape are the frozen cryo-EM sets, and does the confound "
            "found on Set A apply to them more strongly"),
        "label_geometry": {
            "official_test_fold": ref["official_test_fold"],
            "external_set_a": ref["external_set_a"],
            "set_b": _summary(by_set.get("set_b", [])),
            "set_c": _summary(by_set.get("set_c", [])),
            "set_b_and_c_pooled": _summary(pooled),
        },
        "the_confound_named_before_the_read": {
            "what": (
                "geometry_field reads more context than table_field: void "
                "topology is computed from a chain's whole Delaunay "
                "tetrahedralisation and convex hull, and the three aggregations "
                "reach two steps out on the contact graph. If a longer chain "
                "inflates whichever method reads further, then geometry_field "
                "minus table_field on these sets could exceed the training "
                "fold's +0.0121 for a reason unrelated to the four families "
                "generalising"),
            "why_it_applies_here_more_than_to_set_a": (
                "the median chain is "
                f"{_summary(pooled)['chain_length']['median']:.0f} residues "
                f"against {ref['official_test_fold']['chain_length']['median']:.0f}"
                f" for the official fold and "
                f"{ref['external_set_a']['chain_length']['median']:.0f} for "
                f"Set A, and the median positive rate is "
                f"{_summary(pooled)['positive_rate']['median']:.2%} against "
                f"{ref['official_test_fold']['positive_rate']['median']:.2%}"),
            "the_control": {
                "method": "p2rank",
                "why": (
                    "it reads local surface geometry and moved "
                    f"{moved['p2rank']:+.4f} between the official fold and "
                    f"Set A while pLM-NN moved {moved['plmnn']:+.4f}. If P2Rank "
                    f"jumps on these sets they are simply easier and no method's "
                    f"gain is informative; if it stays flat while the "
                    f"context-readers rise, the confound is live"),
            },
            "where_it_would_show_up": (
                "the per-unit comparison split by pocket size, already a "
                "secondary analysis in PREREGISTERED_SETBC.json. A "
                "context-driven inflation appears as an interaction rather than "
                "a level shift"),
            "what_no_analysis_here_can_fix": (
                "45 units cannot separate a context effect from a "
                "generalisation effect. If the read is positive, the honest "
                "claim is that the stack transfers on sets of this shape, and "
                "this shape is not the benchmark's"),
        },
        "per_unit": [
            {"unit": f"{e['pdb']}_{e['chain']}", "set": e["set"],
             "n_residues": per[f"{e['pdb']}_{e['chain']}"],
             "n_cryptic": e["n_cryptic_residues"],
             "resolution": e.get("resolution")}
            for e in entries],
    }

    print("label geometry, medians\n")
    print(f"  {'set':<24} {'chain':>7} {'cryptic':>8} {'rate':>7}")
    for name, s in (("official test fold", ref["official_test_fold"]),
                    ("external set A", ref["external_set_a"]),
                    ("set B", doc["label_geometry"]["set_b"]),
                    ("set C", doc["label_geometry"]["set_c"]),
                    ("set B + C pooled",
                     doc["label_geometry"]["set_b_and_c_pooled"])):
        print(f"  {name:<24} {s['chain_length']['median']:>7.0f} "
              f"{s['cryptic_residues']['median']:>8.0f} "
              f"{s['positive_rate']['median']:>6.2%}")
    print(f"\nconfound named before the read: these chains are "
          f"{doc['label_geometry']['set_b_and_c_pooled']['chain_length']['median'] / ref['official_test_fold']['chain_length']['median']:.1f}x "
          f"the official fold's at the median.")
    print(f"the control is P2Rank, which moved {moved['p2rank']:+.4f} on Set A.")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
        print("commit this before the read")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
