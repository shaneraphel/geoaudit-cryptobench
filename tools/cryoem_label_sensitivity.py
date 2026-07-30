#!/usr/bin/env python3.12
"""How the cryptic-pocket labelling behaves as cryo-EM resolution loosens.

Why this is a separate artifact
-------------------------------
It is a property of the two cryo-EM sets and it is computed with no method run, so
it could have gone inside them. It does not, for one reason: Set B is already frozen
at a digest a future preregistration will pin, and editing it to add a diagnostic
would change that digest. A finding about a frozen set belongs beside it, not in it.

What prompted it
----------------
The pair verdicts move in a way worth understanding before anyone reads either set:

    Set A, X-ray            9.2 per cent of pairs inside the guard band
    Set B, cryo-EM 2.5 A   17.3 per cent
    Set C, cryo-EM 3.0 A   31.9 per cent

The guard band is the interval around the pRMSD floor where the recovered rule
declines to label either way, so a third of Set C's pairs cannot be decided. And the
share called cryptic rises with it, 6.0 to 7.1 to 12.2 per cent, which has two
readings that mean opposite things for a benchmark. Either these proteins really do
move more, or coordinate error inflates the pRMSD and pushes pairs over the floor, in
which case the extra labels are measurement noise read as conformational change.

What is measured here
---------------------
The question is answerable from the sets themselves, without any prediction: if noise
inflation is what is happening, units at coarser resolution should be called cryptic
more often. Two forms of the same test are reported, one binned and one rank-based,
because a bin boundary is a choice and a rank statistic is not.

The rank form counts, over pairs of units with different labels, how often the one
called cryptic is the coarser of the two. Fifty per cent is no relation. It is an
integer count of comparisons and needs no model.

What the answer is allowed to support
-------------------------------------
A weak, non-monotone dependence does not establish noise inflation, and this file
says so where the number is. What it does establish is that the labelling of a
cryo-EM external set is not resolution-neutral, which is a limitation of any such set
and belongs in front of a reader of a future result on one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.cryoem_label_sensitivity.v1"
SETS = {
    "Set A, X-ray": ROOT / "results/external/EXTERNAL_SET.json",
    "Set B, cryo-EM 2.5 A": ROOT / "results/external/SETB_SET.json",
    "Set C, cryo-EM 3.0 A": ROOT / "results/external/SETC_SET.json",
}
CRYO = ("Set B, cryo-EM 2.5 A", "Set C, cryo-EM 3.0 A")
BANDS = ((1.5, 2.0), (2.0, 2.3), (2.3, 2.6), (2.6, 2.8), (2.8, 3.01))


def verdict_mix(doc: dict) -> dict:
    """The verdict counts the builder computed, over every pair it examined.

    Read from ``pair_verdicts`` and not by walking the units. The builder strips
    ``pairs`` from ``units_without_a_cryptic_pocket``, so summing over units counts
    only the pairs of units that ended up with a pocket -- which put Set B at 84
    per cent cryptic on a first pass, a number absurd enough to catch itself.
    """
    if not doc.get("pair_verdicts"):
        raise SystemExit("this set carries no pair_verdicts; the counts cannot be "
                         "recovered by walking units, whose pairs are stripped")
    v = dict(doc["pair_verdicts"])
    n = sum(v.values()) or 1
    return {"n_pairs": n,
            "counts": v,
            "percent": {k: round(100 * c / n, 1) for k, c in sorted(v.items())}}


def units_with_resolution(doc: dict) -> list[tuple[float, int]]:
    out = []
    for u in doc["units"]:
        if u.get("resolution") is not None:
            out.append((float(u["resolution"]), 1))
    for u in doc.get("units_without_a_cryptic_pocket") or []:
        if u.get("resolution") is not None:
            out.append((float(u["resolution"]), 0))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/external/CRYOEM_LABEL_SENSITIVITY.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    mixes, present = {}, {}
    for name, p in SETS.items():
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        present[name] = doc
        mixes[name] = {"artifact": str(p.relative_to(ROOT)), **verdict_mix(doc)}

    rows: list[tuple[float, int]] = []
    for name in CRYO:
        if name in present:
            rows += units_with_resolution(present[name])
    if not rows:
        raise SystemExit("neither cryo-EM set is present")
    res = np.array([r for r, _ in rows])
    lab = np.array([k for _, k in rows])

    binned = []
    for lo, hi in BANDS:
        m = (res >= lo) & (res < hi)
        if not m.any():
            continue
        binned.append({
            "band_angstrom": [lo, hi],
            "n_units": int(m.sum()),
            "n_called_cryptic": int(lab[m].sum()),
            "percent_called_cryptic": round(100 * float(lab[m].mean()), 1),
        })
    pos, neg = res[lab == 1], res[lab == 0]
    conc = int((pos[:, None] > neg[None, :]).sum())
    disc = int((pos[:, None] < neg[None, :]).sum())
    tot = conc + disc
    pct = round(100 * conc / max(tot, 1), 1)
    monotone = all(b["percent_called_cryptic"] <= c["percent_called_cryptic"]
                   for b, c in zip(binned, binned[1:]))

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "no_method_has_been_run": True,
        "is_a_frozen_set": False,
        "what_this_is": (
            "how the recovered cryptic-pocket labelling behaves as cryo-EM "
            "resolution loosens, measured from the two cryo-EM sets themselves"),
        "why_it_is_not_inside_either_set": (
            "Set B is frozen at a digest a preregistration will pin. Editing it to "
            "add a diagnostic would change that digest, so a finding about a frozen "
            "set is filed beside it rather than in it"),
        "pair_verdicts_by_set": mixes,
        "the_two_readings": (
            "the share of pairs inside the guard band roughly doubles from X-ray to "
            "2.5 A and again to 3.0 A, and the share called cryptic rises with it. "
            "Either these proteins move more, or coordinate error inflates the "
            "pRMSD and pushes pairs over the floor, in which case the extra labels "
            "are measurement noise read as conformational change. The two mean "
            "opposite things for a benchmark"),
        "n_cryo_em_units": int(len(res)),
        "resolution_range_angstrom": [round(float(res.min()), 2),
                                      round(float(res.max()), 2)],
        "called_cryptic_by_resolution_band": binned,
        "rank_test": {
            "what_it_counts": "over pairs of units with different labels, how often "
                              "the one called cryptic is the coarser of the two",
            "n_discordant_pairs": tot,
            "n_coarser_called_cryptic": conc,
            "percent": pct,
            "no_relation_would_be": 50.0,
            "why_a_rank_test_as_well": "a bin boundary is a choice and this is not; "
                                       "it is an integer count of comparisons and "
                                       "needs no model",
        },
        "monotone_in_resolution": bool(monotone),
        "what_this_supports": (
            f"the labelling of a cryo-EM external set is not resolution-neutral. "
            f"The dependence is {pct:.1f} per cent against 50 for no relation and it "
            f"is {'monotone' if monotone else 'not monotone'} across bands, which is "
            f"too weak to establish that the higher yield at 3.0 A is noise. What it "
            f"does establish is that resolution is a covariate of the label, which a "
            f"reader of any future result on either set needs in front of them"),
        "what_this_does_not_support": (
            "a correction. Nothing here adjusts a label or a set; the sets are "
            "frozen as built and this is a statement about how to read them"),
    }
    if a.write:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1) + "\n")

    print("  guard band and cryptic share, by set:")
    for name, m in mixes.items():
        gb = next((v for k, v in m["percent"].items() if "guard" in k), 0.0)
        cr = m["percent"].get("cryptic", 0.0)
        print(f"    {name:22s} {m['n_pairs']:5d} pairs   guard band {gb:5.1f}%   "
              f"cryptic {cr:5.1f}%")
    print(f"\n  {len(res)} cryo-EM units, apo resolution "
          f"{res.min():.2f}..{res.max():.2f} A")
    for b in binned:
        lo, hi = b["band_angstrom"]
        print(f"    {lo:.2f}-{hi:.2f} A   {b['n_units']:4d} units   "
              f"{b['n_called_cryptic']:3d} cryptic   "
              f"{b['percent_called_cryptic']:5.1f}%")
    print(f"\n  rank test: {pct}% of {tot} discordant pairs have the coarser unit "
          f"called cryptic (50% is no relation)")
    print(f"  monotone across bands: {monotone}")
    if a.write:
        print(f"\nwrote {a.out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
