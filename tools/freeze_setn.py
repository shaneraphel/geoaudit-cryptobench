#!/usr/bin/env python3
"""Freeze Set N: the external chains shown to have no cryptic pocket.

Set N is the 333 units of ``SETN_INVENTORY.json`` whose every apo-holo pair was
decided ``not cryptic`` --- no pair in the guard band, at least one pair decided.
They come from the same funnel as Set A's 57 positives, were selected by the same
rule at the same time, and have never been scored.

What freezing means here, and why it is not ceremonial
------------------------------------------------------
The unit list, the receptor bytes and the label files are hashed into one document,
and the preregistration pins that hash. The read then refuses to run unless the hash
still matches, so a set cannot be quietly widened after a disappointing number.
``AGENTS.md`` puts the order plainly: build, freeze, hash, preregister, read once.

Receptors go to a new directory. ``data/external/receptors`` holds Set A's 57 and is
pinned by a frozen artifact; writing into it would change a hash that a confirmatory
read depends on. The writer itself is the repository's own
``write_receptor_only_pdb``, the same one that prepared the CryptoBench inputs and
Set A, because a difference in score between two methods must not be able to come
from a difference in how their input was prepared.

Labels are written even though every one of them is empty
----------------------------------------------------------
A label file with ``cryptic_residues: []`` looks like a file that failed to be
written, and the distinction matters enough to be explicit: these chains were
examined against every deposited holo partner and no pair moved a pocket far enough
to be called cryptic. ``examined_pairs`` and ``n_partners_available`` are carried in
each label so that an empty list can be told apart from an absent measurement --- the
failure ``AGENTS.md`` records as printing ``0 clusters remain`` when the truth was
that the cluster count was unavailable.

Usage: PYTHONPATH=src:tools python3.12 tools/freeze_setn.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results/external/SETN_INVENTORY.json"
OUT = ROOT / "results/external/SETN_SET.json"
RECEPTORS = ROOT / "data/external/setn_receptors"
LABELS = ROOT / "data/external/setn_labels"
MANIFEST = ROOT / "data/external/setn_manifest.json"
SCHEMA = "geoaudit.setn_set.v1"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build() -> dict:
    import build_external_set as bes
    import mmcif_atoms
    from pocket_bench.pdb_io import parse_pdb_atoms, write_receptor_only_pdb

    inv = json.loads(INVENTORY.read_text())
    if not inv["reproduction_of_the_frozen_positives"]["identical"]:
        raise SystemExit("the inventory did not reproduce the frozen positives")
    RECEPTORS.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    units, dropped, refusals, entries = [], [], {}, []
    for u in inv["clean"]:
        uid = f"{u['apo_pdb']}_{u['apo_chain']}"
        src = bes.path_of(u["apo_pdb"])
        if src is None:
            dropped.append({"unit": uid, "why": "apo structure not cached"})
            continue
        dest = RECEPTORS / f"{uid}_receptor.pdb"
        rendered, refused = mmcif_atoms.as_pdb_text(src)
        if refused:
            refusals[uid] = refused
        try:
            write_receptor_only_pdb(parse_pdb_atoms(rendered), dest,
                                    chain=u["apo_chain"])
        except ValueError as exc:
            # Same refusal Set A's builder handles, and the same reason for
            # stripping the checkout root out of the message before shipping it.
            dropped.append({"unit": uid, "why": str(exc).replace(f"{ROOT}/", "")})
            continue
        body = dest.read_bytes()
        if body.count(b"\nATOM") < 30:
            dropped.append({"unit": uid, "why": "fewer than 30 atoms"})
            dest.unlink()
            continue

        lp = LABELS / f"{uid}_labels.json"
        lp.write_text(json.dumps({
            "schema": "cryptobench.external_label.v1",
            "clinical_grade": False,
            "pdb_id": u["apo_pdb"], "chain": u["apo_chain"],
            "cryptic_residues": [],
            "binding_residues": [],
            "this_empty_list_is_a_measurement": (
                "every deposited holo partner of this chain was examined and every "
                "pair was decided 'not cryptic'. It is not an absent label"),
            "n_pairs_decided_not_cryptic": len(u["pairs"]),
            "n_partners_available": u["n_partners_available"],
            "uniprot": u["uniprot"], "uniref50": u["cluster"],
            "released": u["released"], "resolution": u["resolution"],
            "rule": "tools/recover_cryptobench_rule.py, recovered from "
                    "CryptoBench's own training records",
        }, indent=1) + "\n")

        units.append({k: u[k] for k in ("apo_pdb", "apo_chain", "uniprot",
                                        "cluster", "resolution", "released",
                                        "n_partners_available")}
                     | {"n_pairs_decided_not_cryptic": len(u["pairs"])})
        entries.append({
            "pdb": u["apo_pdb"], "chain": u["apo_chain"],
            "cluster_id": u["cluster"],
            "receptor_path": str(dest.relative_to(ROOT)),
            "receptor_sha256": _sha(body),
            "label_path": str(lp.relative_to(ROOT)),
            "label_sha256": _sha(lp.read_bytes()),
            "split": "external_negative",
        })

    MANIFEST.write_text(json.dumps({
        "schema": "cryptobench.external_negative_set.v1",
        "clinical_grade": False, "fold": "external_negative",
        "n": len(entries), "entries": entries}, indent=1) + "\n")

    clusters = {u["cluster"] for u in units}
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "no_method_has_been_run": True,
        "question": ("on chains that were examined against every deposited holo "
                     "partner and shown to have no cryptic pocket, how often does "
                     "each method call one anyway"),
        "why_this_is_not_a_second_read_of_set_a": (
            "these units are disjoint from Set A's 57 by construction --- one unit "
            "per UniRef50 cluster, and a cluster contributes either a positive or a "
            "negative and never both. Set A's preregistered read compared per-unit "
            "ROC-AUC over units that have positives; no residue of any unit here "
            "was part of it"),
        "provenance": {
            "selected_by": "tools/build_external_set.py, imported unedited",
            "verdicts_rederived_by": "tools/setn_inventory.py",
            "inventory_sha256": _sha(INVENTORY.read_bytes()),
            "frozen_positives_reproduced_identically": True,
        },
        "externality": {
            "cutoff": "2024-05-08",
            "no_uniref50_cluster_shared_with_cryptobench": True,
            "clusters_disjoint_from_set_a": True,
        },
        "n_units": len(units),
        "n_clusters": len(clusters),
        "n_pairs_decided_not_cryptic": sum(u["n_pairs_decided_not_cryptic"]
                                           for u in units),
        "receptors": {"directory": str(RECEPTORS.relative_to(ROOT)),
                      "n": len(entries),
                      "writer": "pocket_bench.pdb_io.write_receptor_only_pdb",
                      "dropped": dropped,
                      "atoms_the_legacy_format_could_not_hold": refusals},
        "labels": {"directory": str(LABELS.relative_to(ROOT)),
                   "every_cryptic_residue_list_is_empty_by_construction": True},
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "units": units,
        "what_a_unit_here_is_not": [
            "a chain with no pocket. It is a chain that no holo structure deposited "
            "before the cutoff shows a cryptic pocket on",
            "a negative control for druggability, affinity, or anything clinical",
        ],
    }


def report(d: dict) -> None:
    print(f"\nSet N: {d['n_units']} units, {d['n_clusters']} UniRef50 clusters, "
          f"{d['n_pairs_decided_not_cryptic']} pairs decided not cryptic")
    print(f"  receptors written {d['receptors']['n']}, "
          f"dropped {len(d['receptors']['dropped'])}")
    for x in d["receptors"]["dropped"][:5]:
        print(f"    {x['unit']}: {x['why'][:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        report(d)
        miss = [e for e in json.loads(MANIFEST.read_text())["entries"]
                if not (ROOT / e["receptor_path"]).exists()
                or _sha((ROOT / e["receptor_path"]).read_bytes())
                != e["receptor_sha256"]]
        print(f"  receptor digests: {len(miss)} mismatched or absent")
        print(f"{'OK' if not miss else 'FAILED'} {OUT.relative_to(ROOT)}")
        return 0 if not miss else 1
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    report(d)
    print(f"wrote {OUT.relative_to(ROOT)}  sha256 {_sha(OUT.read_bytes())[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
