#!/usr/bin/env python3
"""Write receptors, labels and a manifest for the frozen cryo-EM sets.

What this does, and what it must not
------------------------------------
``SETB_SET.json`` and ``SETC_SET.json`` were built, frozen and hashed as *label*
sets: they name apo chains, their holo partners, the pocket-RMSD verdict for each
pair and the cryptic residues that follow. What they do not carry is anything a
method can be run on. There is no receptor file and no manifest, so neither set
is scorable as it stands.

This materialises them and changes nothing about them. No pair is re-labelled, no
unit is re-selected, no threshold is re-applied. The units are read in the order
the frozen file lists them, each apo chain is written by the same receptor writer
that produced the CryptoBench inputs and Set A's, and the digests are recorded.

The reason to insist on that writer rather than a fresh one is the reason Set A's
builder gives: if the external inputs were prepared differently --- hydrogens
kept, alternates resolved another way, HETATM left in --- a difference in score
could be the preparation rather than the protein.

Why this is not a read
----------------------
Nothing here opens a prediction. Labels are copied from the frozen set into the
shape the scorer's manifest expects; coordinates are inputs. The sets remain
unread, and the order ``AGENTS.md`` fixes is untouched: they were built, frozen
and hashed, this materialises them, and a preregistration must land before
anything is scored.

The one thing that is checked rather than trusted
-------------------------------------------------
Both frozen files carry a ``frozen`` block with the digest they were frozen at.
This recomputes it and refuses if it has moved. A materialiser that quietly
worked against an edited set would produce receptors for units nobody froze, and
the whole value of an unread set is that its contents predate the method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402
from pocket_bench.pdb_io import (                                  # noqa: E402
    parse_pdb_atoms, write_receptor_only_pdb,
)
import mmcif_atoms                                                 # noqa: E402

SCHEMA = "geoaudit.frozen_set_materialisation.v1"
STRUCTURES = ROOT / "data/external/_structures"
# The digest each set had when it was frozen, as published in
# docs/AGENT_MEMORY.md section 5 at freeze time. Hard-coded here rather than read
# out of the set, because a digest a file records about itself moves with the
# file: the only version of this check worth running compares the bytes on disk
# against a number written down somewhere the bytes cannot reach.
SETS = {
    "set_b": (ROOT / "results/external/SETB_SET.json", "09381b40"),
    "set_c": (ROOT / "results/external/SETC_SET.json", "ff112a60"),
}
RECEPTORS = ROOT / "data/external/setbc_receptors"
LABELS = ROOT / "data/external/setbc_labels"
MANIFEST = ROOT / "data/external/setbc_manifest.json"
OUT = ROOT / "results/external/SETBC_MATERIALISATION.json"


def _path_of(pdb: str) -> Path:
    for ext in (".cif", ".pdb"):
        p = STRUCTURES / f"{pdb}{ext}"
        if p.is_file():
            return p
    raise SystemExit(
        f"{pdb} is named by a frozen set and is not in "
        f"{STRUCTURES.relative_to(ROOT)}. Refusing to materialise a set whose "
        f"structures are not all present: a partial set is a different set")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(write: bool) -> int:
    RECEPTORS.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    entries, per_set, dropped = [], {}, []
    for name, (path, want_prefix) in SETS.items():
        raw = path.read_bytes()
        digest = _sha(raw)
        if not digest.startswith(want_prefix):
            raise SystemExit(
                f"{path.name} now hashes to {digest[:16]} and was frozen at "
                f"{want_prefix}... A set whose bytes have moved since the freeze "
                f"is not the set a preregistration would pin, and materialising "
                f"it would produce receptors for units nobody froze")
        doc = json.loads(raw.decode())
        if doc.get("no_method_has_been_run") is not True:
            raise SystemExit(
                f"{path.name} no longer declares no_method_has_been_run; it has "
                f"been read and materialising it now would be pointless")
        recorded = digest

        n_written = 0
        for u in doc["units"]:
            unit = f"{u['apo_pdb']}_{u['apo_chain']}"
            src = _path_of(u["apo_pdb"])
            dest = RECEPTORS / f"{unit}_receptor.pdb"
            rendered, refused = mmcif_atoms.as_pdb_text(src)
            atoms = parse_pdb_atoms(rendered)
            try:
                write_receptor_only_pdb(atoms, dest, chain=u["apo_chain"])
            except ValueError as exc:
                # Two different failures reach this branch and they must not be
                # recorded the same way. The writer refuses a receptor under 50
                # heavy atoms, which is a statement about the chain. It also sees
                # zero atoms when the deposition uses a chain identifier the
                # legacy PDB format cannot hold in its single column -- 9t97
                # labels its chains A3a, A3b, A3c -- which is a statement about
                # the format and about nothing biological. Reporting the second
                # as "too small" describes the symptom and hides the cause.
                why = str(exc).replace(f"{ROOT}/", "")
                cause = ("the deposition uses a chain identifier the legacy PDB "
                         "format cannot hold in one column, so no atom of this "
                         "chain survives rendering. Nothing about the protein"
                         if refused and not atoms else
                         "the chain itself is below the writer's floor")
                dropped.append({"set": name, "unit": unit, "why": why,
                                "cause": cause,
                                "format_refusals": list(refused)[:4] or None,
                                "n_atoms_after_rendering": len(atoms)})
                continue
            body = dest.read_bytes()
            if body.count(b"\nATOM") < 30:
                dropped.append({"set": name, "unit": unit,
                                "why": "fewer than 30 atoms"})
                dest.unlink()
                continue

            lab = LABELS / f"{unit}_labels.json"
            lab.write_text(json.dumps({
                "schema": "geoaudit.external_label.v1",
                "clinical_grade": False,
                "pdb_id": u["apo_pdb"],
                "chain": u["apo_chain"],
                # Copied, not recomputed. The residues are the frozen set's own.
                "cryptic_residues": [r[0] for r in u["residues"]],
                "residue_keys": u["residues"],
                "source_set": path.relative_to(ROOT).as_posix(),
                "source_set_sha256": recorded,
            }, indent=2) + "\n")

            entries.append({
                "pdb": u["apo_pdb"], "chain": u["apo_chain"],
                "set": name,
                "cluster_id": u.get("cluster") or u.get("uniprot"),
                "resolution": u.get("resolution"),
                "released": u.get("released"),
                "receptor_path": dest.relative_to(ROOT).as_posix(),
                "receptor_sha256": _sha(body),
                "label_path": lab.relative_to(ROOT).as_posix(),
                "label_sha256": _sha(lab.read_bytes()),
                "n_cryptic_residues": len(u["residues"]),
            })
            n_written += 1

        per_set[name] = {
            "source": path.relative_to(ROOT).as_posix(),
            "digest_now": recorded,
            "frozen_at_prefix": want_prefix,
            "n_units_in_the_frozen_set": len(doc["units"]),
            "n_units_materialised": n_written,
            "resolution_ceiling": doc.get("resolution_ceiling"),
            "modality": doc.get("modality"),
        }

    manifest = {
        "schema": "geoaudit.setbc_manifest.v1",
        "clinical_grade": False,
        "what_this_is": (
            "the two frozen cryo-EM sets made scorable: one receptor and one "
            "label file per unit, by the same writer that produced the "
            "CryptoBench and Set A inputs"),
        "no_method_has_been_run": True,
        "sets": per_set,
        "n_entries": len(entries),
        "entries": entries,
        "dropped": dropped,
        "writer": "pocket_bench.pdb_io.write_receptor_only_pdb",
        "labels_are_copied_not_recomputed": (
            "the cryptic residues in every label file are the frozen set's own, "
            "copied across. No pair is re-labelled, no threshold re-applied and "
            "no unit re-selected, because the value of an unread set is that its "
            "contents predate the method that will be read on it"),
    }
    if write:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "why_this_is_not_a_read": (
            "no prediction is opened. Labels are copied into the shape the "
            "scorer's manifest expects and coordinates are inputs. Both sets "
            "remain unread and a preregistration must land before anything is "
            "scored"),
        "sets": per_set,
        "n_units_materialised": len(entries),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "why_a_drop_must_be_declared_before_the_read": (
            "a unit removed after scores were seen is a selection. These drops "
            "are format limitations established before any method ran, and the "
            "preregistration has to pin this count so that coverage cannot move "
            "afterwards"),
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": (_sha(MANIFEST.read_bytes()) if MANIFEST.exists()
                            else None),
        "receptors_directory": RECEPTORS.relative_to(ROOT).as_posix(),
        "resolution_is_a_covariate_of_the_label": (
            "CRYOEM_LABEL_SENSITIVITY.json records the recovered rule declining "
            "9.2 % of pairs for X-ray Set A, 17.3 % for Set B and 31.9 % for "
            "Set C, with the cryptic share rising alongside. Any read of these "
            "sets carries that covariate and it belongs in front of the reader, "
            "not in a footnote"),
        "what_has_to_happen_next_and_in_this_order": [
            "finalise the detector; a field compiled after the plan is a "
            "different experiment",
            "preregister the read: pin both set digests and this manifest's, "
            "name the comparisons, fix the statistic, write the losing sentences",
            "read once",
        ],
    }

    for name, s in per_set.items():
        print(f"{name}: {s['n_units_materialised']} of "
              f"{s['n_units_in_the_frozen_set']} units materialised "
              f"({s['modality']}, <= {s['resolution_ceiling']} A)")
    if dropped:
        print("dropped:")
        for d in dropped:
            print(f"  {d['unit']}: {d['cause']}")
    print(f"total scorable units: {len(entries)}")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        print("(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
