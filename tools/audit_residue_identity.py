#!/usr/bin/env python3.12
"""What counts as one residue, measured on the official fold rather than assumed.

Every number in this paper is an average over a residue universe, so the rule
that decides when two atoms belong to the same residue is upstream of all of
them. The rule here is (chain, resseq): the insertion code is not part of the
identity and the alternate-location indicator is not either.

That is a choice, and it is forced. CryptoBench labels a cryptic site with bare
integers, so a label cannot distinguish residue 132 from residue 132A; keying
the universe any more finely would put residues into the evaluation that no
label can ever mark positive, and they would all count as true negatives. The
cost is that where insertion codes exist, several residues share one slot.

The reason this file exists rather than a sentence in the README is that the
cost was never counted. This measures it: how many residues merge, where, and
how many labelled residues are missing from the universe that scores them.

That last quantity is not a defect. CryptoBench transfers a label from a holo
structure onto an apo one, and an apo crystal can leave a residue unresolved, so
five labelled residues in four structures have no coordinates for any method to
see. Dropping them is the only defensible handling -- a detector cannot be
charged for a residue it was never shown -- but it does mean the reported recall
is over labelled residues *present in the apo structure*, and that phrasing has
to survive into the paper. The count is pinned below so that a data refresh
which starts deleting positives in bulk fails instead of quietly shrinking the
denominator.

  PYTHONDONTWRITEBYTECODE=1 python3.12 tools/audit_residue_identity.py [--write]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"
LABELS = ROOT / "data/cryptobench_apo/official_labels"
OUT = ROOT / "results/official_fold/RESIDUE_IDENTITY_AUDIT.json"

# Measured on the official fold at commit 68194da. Raising this is a decision
# about what the paper's recall denominator means, so it should require an edit
# here and an explanation, not happen as a side effect of refetching data.
EXPECTED_UNRESOLVED_POSITIVES = 5
EXPECTED_UNITS_WITH_UNRESOLVED_POSITIVES = 4


def survey_one(path: Path, chain: str) -> dict:
    """Residue identity facts for one receptor, read straight from the columns."""
    by_slot: dict[int, set[str]] = collections.defaultdict(set)
    altlocs: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        if line[21] != chain:
            continue
        resseq = int(line[22:26])
        icode = line[26]
        by_slot[resseq].add(icode)
        if line[16] != " ":
            altlocs.add(line[16])
    merged = {str(k): sorted(v) for k, v in by_slot.items() if len(v) > 1}
    return {
        "n_slots": len(by_slot),
        "n_residues_if_insertion_codes_were_distinct":
            sum(len(v) for v in by_slot.values()),
        "slots_holding_more_than_one_residue": merged,
        "negative_or_zero_numbered_residues": sorted(k for k in by_slot if k <= 0),
        "altloc_indicators_present": sorted(altlocs),
        "universe": sorted(by_slot),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="record the audit")
    args = ap.parse_args(argv)

    if not RECEPTORS.is_dir():
        print(f"receptors absent ({RECEPTORS.relative_to(ROOT)}); "
              "fetch them with tools/fetch_official_data.py")
        return 0

    per_unit: dict[str, dict] = {}
    tag_units: list[str] = []
    icode_units: list[str] = []
    altloc_units: list[str] = []
    orphan_labels: dict[str, list[int]] = {}

    for lab_path in sorted(LABELS.glob("*_labels.json")):
        lab = json.loads(lab_path.read_text())
        pdb, chain = lab["pdb_id"], lab["chain"]
        unit = f"{pdb}_{chain}"
        rec = RECEPTORS / f"{unit}_receptor.pdb"
        if not rec.exists():
            continue
        facts = survey_one(rec, chain)
        universe = set(facts.pop("universe"))
        truth = {int(r) for r in (lab.get("cryptic_residues") or [])}
        missing = sorted(truth - universe)
        if missing:
            orphan_labels[unit] = missing
        facts["n_labelled"] = len(truth)
        facts["n_labelled_absent_from_universe"] = len(missing)
        per_unit[unit] = facts
        if facts["negative_or_zero_numbered_residues"]:
            tag_units.append(unit)
        if facts["slots_holding_more_than_one_residue"]:
            icode_units.append(unit)
        if facts["altloc_indicators_present"]:
            altloc_units.append(unit)

    n_merged = sum(
        f["n_residues_if_insertion_codes_were_distinct"] - f["n_slots"]
        for f in per_unit.values())
    n_tag = sum(len(f["negative_or_zero_numbered_residues"]) for f in per_unit.values())
    n_labelled = sum(f["n_labelled"] for f in per_unit.values())
    n_unresolved = sum(len(v) for v in orphan_labels.values())

    print(f"{len(per_unit)} official units audited")
    print(f"  tag residues numbered zero or below: {n_tag} across "
          f"{len(tag_units)} units")
    print(f"  residues sharing a slot with an insertion-coded twin: {n_merged} "
          f"across {len(icode_units)} units {icode_units}")
    print(f"  units carrying alternate locations: {len(altloc_units)}")
    print(f"  labelled residues unresolved in the apo structure: {n_unresolved} "
          f"of {n_labelled} ({100 * n_unresolved / n_labelled:.2f}%) across "
          f"{len(orphan_labels)} units")
    for unit, miss in sorted(orphan_labels.items()):
        print(f"    {unit}: {miss}")

    if args.write:
        OUT.write_text(json.dumps({
            "schema": "geoaudit.residue_identity_audit.v1",
            "clinical_grade": False,
            "purpose": "the residue-identity rule and what it costs, measured",
            "rule": "a residue is (chain, resseq); the insertion code and the "
                    "alternate-location indicator are not part of its identity",
            "why": "CryptoBench labels are bare integers, so a finer key would "
                   "create residues no label can mark positive",
            "n_units": len(per_unit),
            "n_tag_residues_zero_or_below": n_tag,
            "units_with_tag_residues": sorted(tag_units),
            "n_residues_merged_by_insertion_code": n_merged,
            "units_with_insertion_codes": sorted(icode_units),
            "units_with_altlocs": sorted(altloc_units),
            "n_labelled_residues": n_labelled,
            "n_labelled_unresolved_in_apo": n_unresolved,
            "recall_denominator": "labelled residues present in the apo "
                                  "structure; a residue with no coordinates is "
                                  "outside every method's universe and is "
                                  "neither a hit nor a miss",
            "labelled_residues_unresolved_in_apo": orphan_labels,
            "per_unit": per_unit,
        }, indent=2, allow_nan=False) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")

    if (n_unresolved != EXPECTED_UNRESOLVED_POSITIVES
            or len(orphan_labels) != EXPECTED_UNITS_WITH_UNRESOLVED_POSITIVES):
        print(f"\nthe recall denominator moved: expected "
              f"{EXPECTED_UNRESOLVED_POSITIVES} unresolved labelled residues "
              f"across {EXPECTED_UNITS_WITH_UNRESOLVED_POSITIVES} units, "
              f"measured {n_unresolved} across {len(orphan_labels)}. "
              f"Update the pins in this file and say why in the README.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
