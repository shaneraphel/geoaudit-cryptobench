#!/usr/bin/env python3
"""Chain sequences for the official pLM-NN baseline, in the order it indexes them.

CryptoBench's own baseline takes one ESM2-3B embedding vector per *observed*
residue of the apo chain, zero-based, in the order the coordinates give them.
Their example note warns that this ordering is not the same as the label indexing
in the deposit -- unobserved residues are skipped and the labels are one-based --
and getting that wrong would produce a baseline that is wrong in a way no metric
would reveal.

So the alignment is built explicitly, from the same receptor files our own
detectors read, and every residue row records the ``resseq`` it belongs to rather
than relying on position. Three hazards are handled rather than assumed away:

  Insertion codes. One chain of the fold carries them. Two residues sharing a
  ``resseq`` are two embedding rows but one entry in the evaluation universe,
  which keys on the integer alone, so the mapping is many-to-one and is recorded
  as such.

  Non-monotone numbering. Two chains number their residues out of order in the
  file. The universe is sorted, the embedding is in file order, and conflating
  the two would silently permute the scores of exactly those chains.

  Non-standard residues. Anything outside the twenty is written as ``X``, which
  is what ESM's alphabet expects, and counted so the number is reportable rather
  than discovered later.

This file writes no embedding and loads no model. It is the alignment, and it is
separate so that it can be tested against the deposit's own example.

Usage: PYTHONPATH=src:tools python3.12 tools/plmnn_sequences.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json

from pocket_bench.paths import ROOT

MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"
OUT = ROOT / "results/baselines/PLMNN_SEQUENCES.json"
SCHEMA = "geoaudit.plmnn_sequences.v1"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # The deposit's receptors are ligand-stripped but keep modified residues,
    # and ESM's alphabet has no code for them. Mapping to the parent residue
    # would be a guess about chemistry; X is what the alphabet provides.
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _chain_residues(text: str, chain: str) -> list[tuple[int, str, str]]:
    """(resseq, insertion code, three-letter name) per residue, in file order.

    Grouping is on the full residue key including the insertion code, so a
    residue numbered 52A is its own row rather than being merged into 52.
    """
    out: list[tuple[int, str, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        key = (line[21], int(line[22:26]), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        out.append((key[1], key[2], line[17:20].strip()))
    return out


def build() -> dict:
    man = json.loads(MANIFEST.read_text())
    per = {f"{r['pdb']}_{r['chain']}": r
           for r in json.loads(PER_STRUCTURE.read_text())}

    rows, n_x, with_icode, non_monotone = [], 0, [], []
    for e in man["entries"]:
        uid = f"{e['pdb']}_{e['chain']}"
        res = _chain_residues((ROOT / e["receptor_path"]).read_text(),
                              e["chain"])
        seq = "".join(THREE_TO_ONE.get(name, "X") for _, _, name in res)
        n_x += seq.count("X")
        resseq = [r for r, _, _ in res]
        if any(ic for _, ic, _ in res):
            with_icode.append(uid)
        if resseq != sorted(resseq):
            non_monotone.append(uid)

        universe = sorted(set(resseq))
        if len(universe) != per[uid]["n_universe"]:
            raise SystemExit(
                f"{uid}: the receptor gives {len(universe)} distinct resseq "
                f"but the frozen metrics were computed over "
                f"{per[uid]['n_universe']}; the baseline would be scored on a "
                f"different universe from every other detector")

        rows.append({
            "unit_id": uid,
            "chain": e["chain"],
            "receptor_sha256": e["receptor_sha256"],
            "sequence": seq,
            "n_residues": len(res),
            "resseq_per_row": resseq,
            "n_universe": len(universe),
            "rows_share_a_resseq": len(res) != len(universe),
        })
    rows.sort(key=lambda r: r["unit_id"])

    lens = [r["n_residues"] for r in rows]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "what_this_is": (
            "the apo chain sequences of the official test fold in the order "
            "CryptoBench's pLM-NN baseline indexes them, with the resseq each "
            "embedding row belongs to"),
        "why_the_mapping_is_explicit": (
            "the baseline's embedding is one row per observed residue in file "
            "order while our evaluation universe is the sorted set of integer "
            "resseq. Those two orders differ on the chains that number "
            "residues out of order, and their lengths differ on the chain that "
            "carries insertion codes. Recording the resseq per row makes the "
            "join exact instead of positional"),
        "source": "data/cryptobench_apo/official_receptors, the same files "
                  "every other detector in this repository reads",
        "n_units": len(rows),
        "n_residues_total": sum(lens),
        "shortest_chain": min(lens),
        "longest_chain": max(lens),
        "n_non_standard_residues_written_as_X": n_x,
        "units_with_insertion_codes": with_icode,
        "units_numbered_out_of_order": non_monotone,
        "units_where_rows_outnumber_the_universe": [
            r["unit_id"] for r in rows if r["rows_share_a_resseq"]],
        "sequence_sha256": hashlib.sha256(
            "".join(r["sequence"] for r in rows).encode()).hexdigest(),
        "rows": rows,
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "a sequence is an input to the benchmark, not an answer from it. "
            "No label, prediction or metric is opened here"),
    }


def _report(d: dict) -> None:
    print(f"{d['n_units']} chains, {d['n_residues_total']} residues "
          f"({d['shortest_chain']}-{d['longest_chain']} per chain)")
    print(f"  non-standard residues written as X: "
          f"{d['n_non_standard_residues_written_as_X']}")
    print(f"  insertion codes: {d['units_with_insertion_codes'] or 'none'}")
    print(f"  numbered out of order: "
          f"{d['units_numbered_out_of_order'] or 'none'}")
    print(f"  rows outnumber the universe: "
          f"{d['units_where_rows_outnumber_the_universe'] or 'none'}")
    print(f"  sequence digest {d['sequence_sha256'][:16]}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") is not None:
        bad.append("sequences must not claim a read index")
    live = build()
    if d.get("sequence_sha256") != live["sequence_sha256"]:
        bad.append(f"the sequences no longer follow from the receptors: digest "
                   f"{d.get('sequence_sha256', '')[:12]} against "
                   f"{live['sequence_sha256'][:12]}")
    if d.get("rows") != live["rows"]:
        moved = [a["unit_id"] for a, b in zip(d.get("rows", []), live["rows"])
                 if a != b]
        bad.append(f"{len(moved)} sequence rows changed: {moved[:5]}")
    for r in d.get("rows", []):
        if len(r["sequence"]) != r["n_residues"]:
            bad.append(f"{r['unit_id']}: sequence length disagrees with its "
                       f"residue count")
        if len(r["resseq_per_row"]) != r["n_residues"]:
            bad.append(f"{r['unit_id']}: the resseq map does not cover every "
                       f"row, so the embedding could not be joined")
        if len(set(r["resseq_per_row"])) != r["n_universe"]:
            bad.append(f"{r['unit_id']}: the resseq map does not reproduce the "
                       f"evaluation universe")
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
