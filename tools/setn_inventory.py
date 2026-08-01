#!/usr/bin/env python3
"""The half of the external set that was built, frozen, and never scored.

What is here that nobody has looked at
--------------------------------------
``EXTERNAL_SET.json`` holds 57 units with a cryptic pocket and **503 units without
one**. Both halves came out of the same funnel --- X-ray, released after
CryptoBench's newest structure, no UniRef50 cluster shared with CryptoBench, one
unit per cluster --- and both were frozen and hashed in the same commit. The 57
have been read once, under a preregistration, and are spent. The 503 have never had
a single residue scored by any method, ours or published.

They answer a question the 57 cannot. A per-unit ROC-AUC over a chain that has a
cryptic pocket asks whether the pocket's residues rank above the rest of that chain.
It cannot ask whether a method knows there is nothing to find, because every unit it
is computed on has something to find. Every published CryptoBench comparison,
including this repository's, is silent on that. On 503 chains with no cryptic pocket
the question is the only one available, and it is worth more than its novelty
suggests: a sequence model that has learned "this family has pockets" has no
coordinate evidence with which to withhold a call, and a method reading a specific
apo conformation does.

Why the frozen artifact is not enough on its own
------------------------------------------------
A unit is negative when no apo-holo pair of that chain was labelled cryptic. But the
labelling rule has three outcomes, not two: pocket RMSD above the floor plus the
guard band is cryptic, below the floor minus the band is not cryptic, and between
them is **not labelled either way**. A unit whose only movement fell inside the band
is recorded as negative and is in truth unknown.

The frozen file cannot tell you which those are: it strips the ``pairs`` key from
the negatives, keeping it only for the 57 positives. Subtracting the positives'
verdicts from the set-wide totals says how many there are --- 186 guard-band pairs
among the negatives, so at most 186 of the 503 units are affected and at least 317
are clean --- but not which.

So this tool re-derives them. Every structure the original build downloaded is still
cached under ``data/external/_structures`` (2,442 files, 3.8 GB), the selection is
reproducible from the cached inventory, and ``build_external_set.py`` is imported
unedited: ``AGENTS.md`` forbids modifying a tool that feeds a frozen artifact, and
this needs nothing modified.

The reproduction check that makes the rest believable
-----------------------------------------------------
Re-deriving 560 units gives back the 57 positives with their residue sets. If those
do not match the frozen file exactly --- same units, same residues --- then the
re-derivation is not the same computation the frozen set came from and its verdicts
for the negatives cannot be trusted either. The check runs first and the tool
refuses to write anything if it fails.

What this tool does not do
--------------------------
It scores nothing, reads no prediction, and freezes nothing. Selecting the clean
subset is a rule over labels alone; the set it defines is frozen by
``freeze_setn.py`` and read only after a preregistration, in that order.

Usage: PYTHONPATH=src:tools python3.12 tools/setn_inventory.py [--jobs 9] [--check]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external/SETN_INVENTORY.json"
FROZEN = ROOT / "results/external/EXTERNAL_SET.json"
SCHEMA = "geoaudit.setn_inventory.v1"

_CTX: dict = {}


def _init() -> None:
    """One import of the builder per worker, and no network in any of them."""
    import build_external_set as bes
    _CTX["bes"] = bes


def _label(job: tuple) -> dict:
    uniprot, cluster, apo, partners, n_avail, accepted = job
    bes = _CTX["bes"]
    u = bes.label_unit(apo["pdb"], apo["chain"], partners, accepted)
    return {"uniprot": uniprot, "cluster": cluster,
            "apo_pdb": apo["pdb"], "apo_chain": apo["chain"],
            "resolution": apo.get("resolution"), "released": apo.get("released"),
            "n_partners_available": n_avail,
            "residues": [list(r) for r in u["residues"]],
            "pairs": u["pairs"], "dropped": u["dropped"]}


def rederive(jobs: int) -> list[dict]:
    import build_external_set as bes
    units, _ = bes.candidates()
    cb = json.loads(bes.CB_DATASET.read_text())
    accepted = {r["ligand"] for v in cb.values() for r in v}
    work = [(u["uniprot"], u["cluster"], u["apo"], u["partners"],
             u["n_partners_available"], accepted) for u in units]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, initializer=_init) as pool:
        out = []
        for i, rec in enumerate(pool.imap_unordered(_label, work, chunksize=4), 1):
            out.append(rec)
            if i % 50 == 0 or i == len(work):
                print(f"  {i}/{len(work)} units, {time.time() - t0:.0f}s",
                      flush=True)
    return sorted(out, key=lambda r: (r["apo_pdb"], r["apo_chain"]))


def _dropped_after_labelling(frozen: dict) -> dict[str, str]:
    """Units the frozen build labelled and then lost while writing a receptor.

    Labelling and receptor writing are separate stages, and one unit did not
    survive the second: ``9lym_CA`` has chain identifiers two characters wide, the
    legacy PDB format gives a chain one column, and the receptor came out with zero
    heavy atoms. It is the same class of loss as Set B's ``9t97_A3a``.

    That unit is a **positive** --- the labelling stage found a cryptic pocket on
    it --- so the external positive set is 57 rather than 58 for a reason that is
    about a file format and not about the protein. This function names the losses so
    the reproduction check can subtract them, rather than skipping any unit that
    happens to disagree.
    """
    out = {}
    for rec in (frozen.get("receptors") or {}).get("dropped") or []:
        unit = str(rec.get("unit") or "")
        if unit.endswith("_receptor"):
            unit = unit[: -len("_receptor")]
        why = str(rec.get("why") or "")
        # The stored reason carries a checkout root, which AGENTS.md exempts by
        # name because rewriting the frozen artifact would break its hash. Only
        # the part before the path is repeated here.
        out[unit] = why.split(" for /")[0]
    return out


def reproduction(rows: list[dict]) -> dict:
    """Do the re-derived positives equal the frozen ones, unit for unit?"""
    frozen = json.loads(FROZEN.read_text())
    lost = _dropped_after_labelling(frozen)
    want = {f"{u['apo_pdb']}_{u['apo_chain']}":
            {tuple(r) for r in u["residues"]} for u in frozen["units"]}
    got = {f"{r['apo_pdb']}_{r['apo_chain']}": {tuple(x) for x in r["residues"]}
           for r in rows if r["residues"]}
    accounted = {u: lost[u] for u in sorted(set(got) - set(want)) if u in lost}
    missing = sorted(set(want) - set(got))
    extra = sorted(u for u in set(got) - set(want) if u not in lost)
    differ = sorted(u for u in set(want) & set(got) if want[u] != got[u])
    return {
        "n_frozen_positive_units": len(want),
        "n_rederived_positive_units": len(got),
        "positives_the_frozen_build_lost_writing_receptors": accounted,
        "why_those_are_subtracted": (
            "labelling and receptor writing are separate stages and these units "
            "passed the first and failed the second, so a re-derivation that stops "
            "at labelling is right to find them and the frozen set is right not to "
            "carry them. The external positive set is 57 rather than 58 because of "
            "a PDB chain-identifier column, which is worth saying out loud"),
        "units_frozen_but_not_rederived": missing,
        "units_rederived_but_not_frozen": extra,
        "units_whose_residue_set_differs": differ,
        "identical": not (missing or extra or differ),
        "why_this_gates_everything_below": (
            "the negatives' verdicts come from the same call that produced these "
            "residue sets. If the positives do not come back byte-identical, the "
            "negatives' verdicts are from a different computation than the one the "
            "frozen set records, and the clean subset below would be a guess"),
    }


def partition(rows: list[dict]) -> dict:
    """Split the negatives into decided and undecided, by the rule stated here.

    A negative unit is **clean** when every pair it has was decided --- verdict
    ``not cryptic`` --- and it has at least one. It is **undecided** when any pair
    fell in the guard band, and **unpaired** when no pair survived to a verdict at
    all, because a chain whose partners were all dropped for a technical reason has
    not been shown to lack a pocket; it has not been examined.

    The rule reads verdicts and nothing else. No prediction exists yet.
    """
    clean, undecided, unpaired, positive = [], [], [], []
    for r in rows:
        verdicts = [p["verdict"] for p in r["pairs"]]
        if any(v == "cryptic" for v in verdicts) or r["residues"]:
            positive.append(r)
        elif not verdicts:
            unpaired.append(r)
        elif any(v.startswith("inside the guard band") for v in verdicts):
            undecided.append(r)
        else:
            clean.append(r)
    return {"clean": clean, "undecided": undecided, "unpaired": unpaired,
            "positive": positive}


def build(jobs: int) -> dict:
    rows = rederive(jobs)
    rep = reproduction(rows)
    part = partition(rows)
    frozen = json.loads(FROZEN.read_text())
    n_guard = sum(1 for r in part["undecided"] for p in r["pairs"]
                  if p["verdict"].startswith("inside the guard band"))
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_prediction": False,
        "is_a_frozen_set": False,
        "what_this_is": (
            "the pair verdicts of every unit the external funnel produced, "
            "re-derived from the cached structures so that the 503 units without a "
            "cryptic pocket can be split into those shown to lack one and those "
            "merely not shown to have one"),
        "source_of_the_selection": "tools/build_external_set.py, imported unedited",
        "no_network_was_used": True,
        "reproduction_of_the_frozen_positives": rep,
        "counts": {
            "n_units_examined": len(rows),
            "n_positive": len(part["positive"]),
            "n_negative_clean": len(part["clean"]),
            "n_negative_undecided": len(part["undecided"]),
            "n_negative_unpaired": len(part["unpaired"]),
            "guard_band_pairs_in_undecided_units": n_guard,
        },
        "consistency_with_the_frozen_file": {
            "frozen_n_with_a_cryptic_pocket": frozen["n_units_with_a_cryptic_pocket"],
            "frozen_n_without_one": frozen["n_units_without_one"],
            "rederived_n_negative_total": (len(part["clean"])
                                           + len(part["undecided"])
                                           + len(part["unpaired"])),
        },
        "rule_for_clean": (
            "every pair decided 'not cryptic', at least one such pair, no pair in "
            "the guard band. Fixed here before any method is run on any of them"),
        "what_a_clean_negative_is_not": [
            "a chain with no pocket. It is a chain no deposited holo partner shows "
            "a cryptic pocket on, which is a statement about the PDB as of the "
            "cutoff and not about the protein",
            "a claim about druggability, affinity, or anything clinical",
        ],
        "clean": part["clean"],
        "undecided": [{k: v for k, v in r.items() if k != "pairs"}
                      for r in part["undecided"]],
        "unpaired": [{k: v for k, v in r.items() if k != "pairs"}
                     for r in part["unpaired"]],
    }


def report(d: dict) -> None:
    c, rep = d["counts"], d["reproduction_of_the_frozen_positives"]
    print(f"\nexternal funnel re-derived: {c['n_units_examined']} units")
    print(f"  reproduction of the frozen 57 positives: "
          f"{'identical' if rep['identical'] else 'MISMATCH'}")
    if not rep["identical"]:
        print(f"    missing {len(rep['units_frozen_but_not_rederived'])}, "
              f"extra {len(rep['units_rederived_but_not_frozen'])}, "
              f"differing {len(rep['units_whose_residue_set_differs'])}")
    print(f"  positive                {c['n_positive']:4d}")
    print(f"  negative, decided       {c['n_negative_clean']:4d}   <- scorable")
    print(f"  negative, guard band    {c['n_negative_undecided']:4d}   "
          f"({c['guard_band_pairs_in_undecided_units']} undecided pairs)")
    print(f"  negative, no pair       {c['n_negative_unpaired']:4d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(9, os.cpu_count() or 4))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        report(d)
        ok = d["reproduction_of_the_frozen_positives"]["identical"]
        print(f"{'OK' if ok else 'FAILED'} {OUT.relative_to(ROOT)}")
        return 0 if ok else 1
    d = build(a.jobs)
    report(d)
    if not d["reproduction_of_the_frozen_positives"]["identical"]:
        print("\nrefusing to write: the frozen positives did not come back "
              "identical, so these verdicts are not the frozen set's verdicts")
        return 1
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
