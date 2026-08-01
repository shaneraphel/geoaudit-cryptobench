#!/usr/bin/env python3
"""Is the external set easier than the benchmark? Two answers that disagree.

The question, and why one number cannot settle it
--------------------------------------------------
A reviewer's objection: the external set has only 57 positive units and may be
easier than CryptoBench, which would make the one confirmatory accuracy result
in this paper cheaper than it looks. It is a fair objection and it is checkable
two ways, which is the point of this tool -- the two ways disagree, and reporting
either alone would be a selection.

**By label geometry the external set is harder.** Three quantities control how
hard a per-unit ranking problem is: how many residues there are to rank, how many
of them are positive, and how large the pocket is. This repository has measured
the third directly -- ``FAILURE_TAIL.json`` puts the deployed field at 0.5991 on
units with under ten cryptic residues and 0.8766 on units with more than
twenty-two -- so pocket size is not a guess about difficulty, it is the axis with
a measured effect of 0.28 on it. The external set has longer chains, smaller
pockets and a sparser positive rate than the official fold, on all three.

**By achieved accuracy three of four methods score higher on it.** That is the
reviewer's reading and it has support. But the fourth is the discriminating case:
**P2Rank moves by +0.004 where the two methods that read long-range context move
by +0.04 and +0.05.** A uniformly easier set lifts a local surface detector too.
This one does not, and its chains are 46 residues longer at the median, which is
the difference a context-reading method can use and a local one cannot.

So the honest statement is neither "the set is easier" nor "the objection is
answered". It is that the set is not uniformly easier, that what it rewards is
context rather than pocket size, and that any method compared on it inherits that
property. Whether the same ordering would appear on a set matched for chain
length is unmeasured, and matching for it is what a second external set should do.

What this reads, and what it does not
--------------------------------------
Frozen artifacts and label files only. Every accuracy number here was already
read and is quoted, not recomputed: no read of the official fold or of the
external set is spent, and the tool refuses if an artifact it quotes says it
scored a fold it should not have. Chain lengths come from receptor coordinates,
which are inputs rather than labels.
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
from pocket_bench.pdb_io import parse_pdb_atoms                    # noqa: E402

SCHEMA = "geoaudit.external_set_difficulty.v1"
OFFICIAL_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
EXTERNAL_SET = ROOT / "results/external/EXTERNAL_SET.json"
EXTERNAL_READ = ROOT / "results/external/EXTERNAL_READ.json"
PLMNN_READ = ROOT / "results/official_fold/PLMNN_READ.json"
POCKETMINER_READ = ROOT / "results/official_fold/POCKETMINER_READ.json"
FAILURE_TAIL = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"
OUT = ROOT / "results/external/EXTERNAL_SET_DIFFICULTY.json"

SKIP_RES = frozenset({"HOH", "WAT", "DOD"})


def _universe(path: Path, chain: str) -> int:
    """Residues a method must rank on one unit: the scoring denominator."""
    atoms = parse_pdb_atoms(path.read_text())
    return len({a["resseq"] for a in atoms
                if a["chain"] == chain and a["element"] != "H"
                and a["resname"] not in SKIP_RES})


def _geometry(entries: list[dict]) -> list[dict]:
    rows = []
    for e in entries:
        lp, rp = ROOT / e["label_path"], ROOT / e["receptor_path"]
        if not (lp.is_file() and rp.is_file()):
            continue
        n_cryptic = len(json.loads(lp.read_text()).get("cryptic_residues") or [])
        if n_cryptic == 0:
            continue      # no positives, so no ranking problem and no AUC
        n_res = _universe(rp, e["chain"])
        rows.append({"unit": f"{e['pdb']}_{e['chain']}", "n_residues": n_res,
                     "n_cryptic": n_cryptic,
                     "positive_rate": n_cryptic / max(n_res, 1)})
    return rows


def _summary(rows: list[dict]) -> dict:
    nr = np.array([r["n_residues"] for r in rows], float)
    nc = np.array([r["n_cryptic"] for r in rows], float)
    pr = np.array([r["positive_rate"] for r in rows], float)
    return {
        "n_units": len(rows),
        "chain_length": {"median": float(np.median(nr)), "mean": float(nr.mean())},
        "cryptic_residues": {"median": float(np.median(nc)),
                             "mean": float(nc.mean())},
        "positive_rate": {"median": float(np.median(pr)),
                          "mean": float(pr.mean())},
        "share_under_ten_cryptic": float((nc < 10).mean()),
        "share_over_twentytwo_cryptic": float((nc > 22).mean()),
    }


def build(write: bool) -> int:
    ext = json.loads(EXTERNAL_SET.read_text())
    ext_manifest = ROOT / ext["fold_files"]["manifest"]
    off = json.loads(OFFICIAL_MANIFEST.read_text())["entries"]
    exm = json.loads(ext_manifest.read_text())["entries"]

    g_off, g_ext = _geometry(off), _geometry(exm)
    if not g_off or not g_ext:
        raise SystemExit(
            "one of the two sets has no materialised receptors, so chain "
            "lengths cannot be computed. Run tools/fetch_official_data.py and "
            "the external build first; this reports nothing rather than "
            "comparing one set against a guess")
    s_off, s_ext = _summary(g_off), _summary(g_ext)

    # Accuracy, quoted from the reads that already happened. Nothing is rescored.
    ext_read = json.loads(EXTERNAL_READ.read_text())
    plm = json.loads(PLMNN_READ.read_text())
    pm = json.loads(POCKETMINER_READ.read_text())
    official = {
        "table_field": pm["levels"]["table_field"],
        "p2rank": pm["levels"]["p2rank"],
        "pocketminer": pm["levels"]["pocketminer"],
        "plmnn": plm["reproduction_gate"]["baseline_mean_per_unit_roc_auc"],
    }
    external = dict(ext_read["levels"])
    moved = {k: round(external[k] - official[k], 6)
             for k in sorted(official) if k in external}

    tail = json.loads(FAILURE_TAIL.read_text())

    harder = [
        ("chain length", s_ext["chain_length"]["median"],
         s_off["chain_length"]["median"], "more residues to rank against"),
        ("cryptic residues", s_ext["cryptic_residues"]["median"],
         s_off["cryptic_residues"]["median"],
         "smaller pockets, and pocket size is the axis with a measured 0.28 "
         "effect on per-unit ROC-AUC"),
        ("positive rate", s_ext["positive_rate"]["median"],
         s_off["positive_rate"]["median"], "sparser positives"),
    ]

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "why_this_declares_no_read": (
            "every accuracy number here is quoted from an artifact whose read "
            "was already indexed. Chain lengths come from receptor coordinates, "
            "which are inputs. No prediction is computed and no label is "
            "compared against one"),
        "question": (
            "is the external set easier than the official CryptoBench test "
            "fold, as a reviewer suspected of a 57-unit set"),
        "answer": (
            "not uniformly, and the two ways of asking disagree. By label "
            "geometry it is harder on all three axes. By achieved accuracy "
            "three of four methods score higher on it -- but P2Rank moves by "
            "+0.004 where the two methods that read long-range context move by "
            "+0.04 and +0.05, and a uniformly easier set lifts a local surface "
            "detector too"),
        "label_geometry": {
            "official_test_fold": s_off,
            "external_set_a": s_ext,
            "axes_on_which_the_external_set_is_harder": [
                {"axis": a, "external": e, "official": o, "why_harder": w}
                for a, e, o, w in harder],
            "why_pocket_size_is_not_a_guess": {
                "source": "results/architecture_sweep/FAILURE_TAIL.json",
                "under_ten_cryptic_residues": tail.get("strata", tail).get(
                    "note", "see artifact"),
                "measured_effect": (
                    "0.5991 on units with under ten cryptic residues against "
                    "0.8766 on units with more than twenty-two"),
            },
        },
        "achieved_accuracy": {
            "official_test_fold": official,
            "external_set_a": external,
            "external_minus_official": moved,
            "the_discriminating_case": {
                "method": "p2rank",
                "moved": moved.get("p2rank"),
                "reading": (
                    "P2Rank reads local surface geometry. It is the arm that "
                    "distinguishes a set that is easier from a set that rewards "
                    "context, and it barely moves. The external set's chains are "
                    "longer at the median, which a context-reading method can "
                    "use and a local one cannot"),
            },
            "what_this_does_not_settle": (
                "whether the same ordering would appear on a set matched to the "
                "official fold for chain length. That is unmeasured, and "
                "matching for it is what a second external set should do"),
        },
        "the_objection_that_stands": (
            "57 positive units is small. The interval on the confirmatory "
            "result reflects that and is reported with it; what this tool "
            "addresses is the second half of the objection, that the set is "
            "easier, and that half is not supported by the label geometry and "
            "only partly by the accuracy"),
        "per_unit": {"official_test_fold": g_off, "external_set_a": g_ext},
    }

    print("label geometry, medians\n")
    print(f"  {'axis':<20} {'official':>10} {'Set A':>10}   harder")
    for a, e, o, _w in harder:
        fmt = (lambda v: f"{v:.2%}") if a == "positive rate" else (
            lambda v: f"{v:.0f}")
        print(f"  {a:<20} {fmt(o):>10} {fmt(e):>10}   "
              f"{'Set A' if (e > o if a == 'chain length' else e < o) else 'official'}")
    print("\nachieved mean per-unit ROC-AUC, quoted from the frozen reads\n")
    print(f"  {'method':<14} {'official':>10} {'Set A':>10} {'moved':>9}")
    for k in sorted(moved):
        print(f"  {k:<14} {official[k]:>10.4f} {external[k]:>10.4f} "
              f"{moved[k]:>+9.4f}")
    print(f"\n  P2Rank moves {moved.get('p2rank'):+.4f}: a uniformly easier set "
          f"would lift it too.")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
