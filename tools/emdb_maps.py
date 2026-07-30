#!/usr/bin/env python3.12
"""Fetch EMDB density maps with the experimental method asserted, not discovered.

Why this tool exists rather than a ``curl`` line
------------------------------------------------
The two maps in this repository were first fetched by hand, and one of them was
the wrong kind of experiment. ``SETB_POOL.json`` already recorded that RCSB's
``experimental_method == "EM"`` returns electron crystallography alongside
single-particle cryo-EM, and it stated the practical consequence: MicroED entries
sit at the top of any selection ordered by resolution. EMD-46871 was then picked
as "the high-resolution cryo-EM example" by taking the best resolution on offer.
It is ``electronCrystallography`` at 1.09 A, titled *Structure of proteinase K
from energy-filtered MicroED data*.

So the selection is declared here as a table, each row carrying the method it is
expected to be, and the tool refuses to record a map whose EMDB metadata
disagrees with its row. A contamination noted in prose is not a filter; an
assertion is.

Two independent witnesses, both cheap
-------------------------------------
``structure_determination.method`` from the EMDB entry API is authoritative and
is what the assertion reads. The map header is checked as well, because it would
have caught this one without any network call: single-particle reconstructions
arrive on a cubic box with axis order 1/2/3 and a voxel at or above about 0.4 A,
while a map derived from crystallography carries the unit cell of the crystal --
EMD-46871 has a 0.2677 A voxel over a 55.15 x 58.89 x 60.77 A cell with axis
order 3/2/1. The two witnesses are recorded side by side so a reader can see that
they agree, and ``header_consistent_with_method`` says whether they did.

What this file is not
---------------------
It is not part of the benchmark. No number in the primary claim reads a density
map; the detector consumes apo coordinates from CryptoBench and the external
sets. These maps exist to illustrate what the word "resolution" spans across
methods that share the label "EM", and nothing here supports a claim about a
ligand pose, a binding affinity or a therapeutic decision.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pocket_bench.paths import ROOT  # noqa: E402

SCHEMA = "geoaudit.emdb_maps.v1"
DEST = ROOT / "data/external/_emdb"
OUT = ROOT / "results/external/EMDB_MAPS.json"
ENTRY_API = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{emd}"
MAP_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd}/map/emd_{emd}.map.gz"

# A single-particle reconstruction is delivered on a box the reconstruction chose;
# a map derived from a crystal carries the crystal's unit cell. This is the voxel
# below which the second explanation becomes the likely one -- it is a screen for
# the recording, not a criterion for the assertion, which reads EMDB metadata.
CRYSTALLOGRAPHIC_VOXEL_CEILING = 0.40

# Declared before anything is fetched. ``method`` is what EMDB must say; a
# mismatch is an error, not a note in the output.
MAPS = (
    {
        "emd": "55233",
        "method": "singleParticle",
        "why": (
            "a VHL-recruiting PROTAC ternary complex containing ERa "
            "(EloB/EloC/VHL/CV2a/14-3-3zeta/ERa, local refinement), which is the "
            "experimental counterpart of the degrader modality in the ESR1 "
            "appendix. Its 4.3 A resolution is the reason it is here: at 4.3 A "
            "individual ligand atoms are not placeable, so it is evidence for "
            "the appendix's refusal to assert a binding pose rather than "
            "evidence of one"
        ),
    },
    {
        "emd": "46871",
        "method": "electronCrystallography",
        "why": (
            "not cryo-EM, and kept for that reason. Proteinase K from "
            "energy-filtered MicroED at 1.09 A is the entry that was mistaken "
            "for a high-resolution cryo-EM example by ordering candidates on "
            "resolution. It stays as the recorded instance of the contamination "
            "SETB_POOL.json describes, and as the contrast that shows how far "
            "the label EM stretches: 1.09 A here against 4.3 A for a "
            "single-particle reconstruction, a factor of four in the same word"
        ),
    },
)


class MethodMismatch(RuntimeError):
    """EMDB reports a different experiment than the declaration expected."""


def _get_json(url: str, timeout: float = 60.0, tries: int = 4) -> dict:
    """Fetch JSON, retrying on a dropped connection.

    EBI closed the connection on the third call inside two minutes while the
    first two succeeded, so this is throttling rather than a broken route: the
    same request over the same link had just worked. Backoff, and let a failure
    that survives four attempts raise rather than degrade into a default.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    last: Exception | None = None
    for i in range(tries):
        if i:
            time.sleep(2.0 * i)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            last = e
    raise RuntimeError(f"{url} failed after {tries} attempts: {last!r}")


def entry_metadata(emd: str) -> dict:
    """Method, resolution and title as EMDB states them."""
    d = _get_json(ENTRY_API.format(emd=emd))
    sd = d.get("structure_determination_list", {}).get(
        "structure_determination", [{}])[0]
    res, res_type = None, None
    proc = sd.get("image_processing")
    for block in (proc if isinstance(proc, list) else [proc]):
        if not isinstance(block, dict):
            continue
        for key in ("final_reconstruction", "final_three_d_reconstruction"):
            got = block.get(key)
            if isinstance(got, dict) and isinstance(got.get("resolution"), dict):
                res = got["resolution"].get("valueOf_")
                res_type = got["resolution"].get("res_type")
    return {
        "method": sd.get("method"),
        "aggregation_state": sd.get("aggregation_state"),
        "resolution_angstrom": float(res) if res is not None else None,
        "resolution_type": res_type,
        "title": d.get("admin", {}).get("title"),
    }


def map_header(path: Path) -> dict:
    """The MRC/CCP4 header fields that distinguish a box from a unit cell.

    Layout, in 4-byte words: 1-3 NX/NY/NZ, 4 MODE, 8-10 MX/MY/MZ intervals along
    the cell edges, 11-13 CELLA in Angstrom, 14-16 CELLB angles, 17-19 the
    column/row/section to axis mapping.
    """
    with gzip.open(path, "rb") as fh:
        h = fh.read(1024)
    nx, ny, nz, mode = struct.unpack_from("<4i", h, 0)
    mx, my, mz = struct.unpack_from("<3i", h, 28)
    ax, ay, az = struct.unpack_from("<3f", h, 40)
    al, be, ga = struct.unpack_from("<3f", h, 52)
    mapc, mapr, maps = struct.unpack_from("<3i", h, 64)
    vox = [ax / mx if mx else None, ay / my if my else None,
           az / mz if mz else None]
    return {
        "grid": [nx, ny, nz],
        "mode": mode,
        "intervals": [mx, my, mz],
        "cell_angstrom": [round(ax, 4), round(ay, 4), round(az, 4)],
        "cell_angles_degrees": [round(al, 2), round(be, 2), round(ga, 2)],
        "axis_order_col_row_sec": [mapc, mapr, maps],
        "voxel_angstrom": [None if v is None else round(v, 4) for v in vox],
        "voxel_is_cubic": len({round(v, 3) for v in vox if v is not None}) == 1,
        "axis_order_is_canonical": [mapc, mapr, maps] == [1, 2, 3],
    }


def looks_crystallographic(hdr: dict) -> bool:
    """Whether the header alone would have flagged this as crystal-derived."""
    vox = [v for v in hdr["voxel_angstrom"] if v is not None]
    fine = bool(vox) and min(vox) < CRYSTALLOGRAPHIC_VOXEL_CEILING
    return fine or not hdr["axis_order_is_canonical"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def download(emd: str, dest: Path, timeout: float = 600.0) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"emd_{emd}.map.gz"
    if out.exists():
        return out
    url = MAP_URL.format(emd=emd)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=timeout) as fh, tmp.open("wb") as w:
        while blk := fh.read(1 << 20):
            w.write(blk)
    tmp.rename(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download any declared map that is not present")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    rows, problems = [], []
    for spec in MAPS:
        emd, declared = spec["emd"], spec["method"]
        path = DEST / f"emd_{emd}.map.gz"
        if not path.exists():
            if not a.fetch:
                problems.append(f"EMD-{emd}: absent; rerun with --fetch")
                continue
            print(f"  fetching EMD-{emd} ...", flush=True)
            path = download(emd, DEST)

        meta = entry_metadata(emd)
        hdr = map_header(path)
        if meta["method"] != declared:
            raise MethodMismatch(
                f"EMD-{emd} is declared {declared!r} but EMDB reports "
                f"{meta['method']!r} ({meta['title']!r}). Fix the declaration or "
                f"drop the entry; do not record it as the declared method.")

        crys = looks_crystallographic(hdr)
        expected_crys = declared != "singleParticle"
        rows.append({
            "emd_id": f"EMD-{emd}",
            "declared_method": declared,
            "why_this_entry": spec["why"],
            "emdb_metadata": meta,
            "map_header": hdr,
            "header_says_crystallographic": crys,
            "header_consistent_with_method": crys == expected_crys,
            "is_single_particle_cryo_em": declared == "singleParticle",
            "source_url": MAP_URL.format(emd=emd),
            "metadata_url": ENTRY_API.format(emd=emd),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "path": str(path.relative_to(ROOT)),
            "tracked_in_git": False,
        })
        print(f"  EMD-{emd}  {meta['method']}  "
              f"{meta['resolution_angstrom']} A  voxel "
              f"{hdr['voxel_angstrom'][0]} A  axes "
              f"{hdr['axis_order_col_row_sec']}  "
              f"header/method agree: {crys == expected_crys}", flush=True)

    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    sp = [r for r in rows if r["is_single_particle_cryo_em"]]
    art = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        # The manifest classifies by the artifact's own declaration. This is not a
        # frozen set: nothing has been selected, labelled, hashed or preregistered
        # from these two maps, and without the field the path fall-through would
        # register it as "an earlier frozen state", which is the field defect the
        # classifier's own comment records from the Set B inventory.
        "is_a_frozen_set": False,
        "what_this_is": (
            "provenance for the EMDB density maps held under data/external/_emdb, "
            "with the experimental method asserted against EMDB metadata rather "
            "than inferred from the entry title or the resolution"),
        "what_this_is_not": (
            "not an input to the benchmark. No number in the primary claim reads "
            "a density map; the detector consumes apo coordinates. Nothing here "
            "supports a claim about a ligand pose or a binding affinity"),
        "why_the_method_is_asserted": (
            "SETB_POOL.json records that RCSB's experimental_method == 'EM' also "
            "returns electron crystallography, and that MicroED entries occupy "
            "the top of any resolution-ordered selection. EMD-46871 was then "
            "selected as a high-resolution cryo-EM example by exactly that "
            "ordering. The declaration in MAPS now carries the expected method "
            "and a mismatch raises"),
        "crystallographic_voxel_ceiling_angstrom": CRYSTALLOGRAPHIC_VOXEL_CEILING,
        "n_maps": len(rows),
        "n_single_particle_cryo_em": len(sp),
        "maps_are_gitignored": True,
        "why_gitignored": (
            "139 MB of density volumes; .gitignore excludes data/external/_emdb/ "
            "and this file carries the URL and sha256 so a reader can refetch and "
            "check byte identity"),
        "maps": rows,
    }
    if a.write:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(art, indent=2) + "\n")
        print(f"\nwrote {a.out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
