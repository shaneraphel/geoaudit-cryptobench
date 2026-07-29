"""Stage two: apo-holo units and cryptic labels for structures CryptoBench never saw.

This takes the inventory of post-cutoff entries and the UniRef50 clusters, picks
one apo chain per cluster, finds its holo partners, and labels its cryptic
residues with the rule recovered from CryptoBench's own training records. The
output is a fold file in the same shape the harness already reads, plus receptor
PDBs written by the same writer that produced the CryptoBench receptors, so every
method sees the new set exactly as it sees the old one.

Three things make this a validation set rather than another test fold:

  the date     both structures of every pair were released after 2024-05-08, the
               last day CryptoBench could have seen anything
  the sequence no accession, and no UniRef50 cluster at 50% identity, is shared
               with anything CryptoBench names, in either fold
  the order    this file computes labels and never touches a prediction. Nothing
               here can be tuned against a result, because no result exists yet

No method is run here, no score is read, and no threshold is chosen. The
preregistration that locks the methods comes after this artefact is frozen, and
refers to it by hash.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import mmcif_atoms
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms, write_receptor_only_pdb
from recover_cryptobench_rule import (CONTACT_ANGSTROM, COUNT_HYDROGENS,
                                      MIN_POCKET_ATOMS_FOR_PRMSD,
                                      PRMSD_FLOOR, PRMSD_GUARD_BAND, _atoms,
                                      _heavy_by_residue, _is_hydrogen,
                                      contact_residues, pocket_rmsd,
                                      seqres_names)

INVENTORY = ROOT / "data/external/INVENTORY.json.gz"
UNIREF = ROOT / "data/external/UNIREF50.json"
CB_DATASET = ROOT / "data/cryptobench_apo/_osf/dataset.json"
STRUCTURES = ROOT / "data/external/_structures"
RECEPTORS = ROOT / "data/external/receptors"
LABELS = ROOT / "data/external/labels"
MANIFEST = ROOT / "data/external/external_manifest.json"
OUT = ROOT / "results/external/EXTERNAL_SET.json"

SCHEMA = "geoaudit.external_set.v1"

# A fragment-screening campaign can deposit hundreds of holo chains for one
# protein. Taking all of them would let a single target dominate the label set,
# so partners are capped, ordered by resolution and then by identifier so the
# choice is a rule rather than a preference.
MAX_HOLO_PARTNERS = 20
RCSB = "https://files.rcsb.org/download/{}.pdb.gz"
RCSB_CIF = "https://files.rcsb.org/download/{}.cif.gz"
# How many entries the both-formats equivalence is measured on. Cheap enough to
# do properly and pointless to do on three.
N_FORMAT_CHECK = 30
FETCH_RETRIES = 4
WORKERS = 8
# IncompleteRead is an HTTPException, not an OSError, so it slips past a retry
# guard built from OSError alone and kills the run instead of being retried.
# That is the same gap the rule recovery had to close.
TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
             http.client.HTTPException)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Modified residues the benchmark counts as protein, mapped to their parent
    # so that an alignment is not broken by a selenium atom.
    "MSE": "M", "SEC": "C", "PYL": "K", "KCX": "K", "CME": "C", "CSD": "C",
    "CSO": "C", "OCS": "C", "CAS": "C", "TPO": "T", "SEP": "S", "PTR": "Y",
    "MLY": "K", "M3L": "K", "HYP": "P", "PCA": "Q", "FME": "M", "CGU": "E",
    "SLL": "K", "CMH": "C", "ALY": "K", "DAL": "A", "DLE": "L", "DVA": "V",
}


# --------------------------------------------------------------------------
# fetching


def _download(url: str, dest: Path) -> None:
    raw = urllib.request.urlopen(url, timeout=180).read()
    dest.write_bytes(gzip.decompress(raw))


def _fetch(pid: str) -> dict | None:
    """The entry in whichever format the PDB has, preferring the legacy one.

    A quarter of these entries are too large for the .pdb format and the PDB
    offers only mmCIF. Reading mmCIF is justified by the equivalence measured in
    format_equivalence rather than assumed, and .pdb is still preferred where it
    exists so that the common case goes through the reader the rule was recovered
    with.
    """
    if path_of(pid) is not None:
        return None
    last = None
    for attempt in range(FETCH_RETRIES):
        try:
            _download(RCSB.format(pid), STRUCTURES / f"{pid}.pdb")
            return None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break
            last = exc
        except TRANSIENT as exc:
            last = exc
        time.sleep(0.8 * (attempt + 1))
    for attempt in range(FETCH_RETRIES):
        try:
            _download(RCSB_CIF.format(pid), STRUCTURES / f"{pid}.cif")
            return None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"pdb": pid, "error": "in neither .pdb nor mmCIF"}
            last = exc
        except TRANSIENT as exc:
            last = exc
        time.sleep(0.8 * (attempt + 1))
    return {"pdb": pid, "error": type(last).__name__}


def fetch(pids: list[str]) -> list[dict]:
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    need = [p for p in pids if path_of(p) is None]
    if not need:
        return []
    print(f"  fetching {len(need)} structures", flush=True)
    bad: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(_fetch, need), start=1):
            if r:
                bad.append(r)
            if i % 25 == 0:
                print(f"    {i}/{len(need)}", flush=True)
    return bad


def path_of(pid: str) -> Path | None:
    for ext in (".pdb", ".cif"):
        p = STRUCTURES / f"{pid}{ext}"
        if p.is_file():
            return p
    return None


def read_atoms(path: Path) -> list[tuple]:
    return mmcif_atoms.atoms(path) if path.suffix == ".cif" else _atoms(path)


def read_seqres(path: Path) -> dict[str, set[str]]:
    return (mmcif_atoms.seqres_names(path) if path.suffix == ".cif"
            else seqres_names(path))


def format_equivalence() -> dict:
    """Evidence that reading mmCIF gives what reading .pdb gives.

    Measured on entries the PDB publishes in both formats, so it is a comparison
    and not a claim. Without this the external set would rest on a second parser
    nobody checked, and a quarter of its structures go through that parser.
    """
    # The fetch prefers .pdb and only falls back, so no entry has both formats
    # unless one is asked for. A deterministic sample is fetched here purely so
    # the comparison has something to compare, seeded so it is the same sample on
    # every rebuild.
    pdbs = sorted(p.stem for p in STRUCTURES.glob("*.pdb"))
    rng = np.random.default_rng(20260730)
    want = ([pdbs[i] for i in rng.permutation(len(pdbs))[:N_FORMAT_CHECK]]
            if pdbs else [])
    for pid in want:
        if not (STRUCTURES / f"{pid}.cif").is_file():
            for _ in range(FETCH_RETRIES):
                try:
                    _download(RCSB_CIF.format(pid), STRUCTURES / f"{pid}.cif")
                    break
                except Exception:                     # noqa: BLE001 - transient
                    time.sleep(1.0)
    both = sorted(pid for pid in want
                  if (STRUCTURES / f"{pid}.cif").is_file())
    rows = [mmcif_atoms.agreement(STRUCTURES / f"{pid}.cif",
                                  STRUCTURES / f"{pid}.pdb")
            for pid in both[:N_FORMAT_CHECK]]
    return {
        "n_compared": len(rows),
        "n_identical": sum(r["identical"] for r in rows),
        "worst_coordinate_difference": max(
            (r["worst_coordinate_difference"] for r in rows), default=None),
        "tolerance": mmcif_atoms.COORDINATE_TOLERANCE,
        "what_is_compared": ("record type, residue name, chain, residue number, "
                             "insertion code, atom name and element exactly; "
                             "coordinates to the tolerance, because .pdb stores "
                             "three decimals and mmCIF stores more"),
        "disagreeing": [r for r in rows if not r["identical"]][:10],
    }


# --------------------------------------------------------------------------
# residue correspondence


def chain_sequence(atoms: list[tuple], chain: str,
                   polymer: set[str]) -> tuple[str, list[tuple[int, str]]]:
    """The chain as observed, in file order: one letter and one key per residue.

    Built from the coordinates rather than from SEQRES, because a residue with no
    coordinates cannot be labelled, superposed, or predicted, so it has no place
    in the correspondence.
    """
    seen: dict[tuple[int, str], str] = {}
    order: list[tuple[int, str]] = []
    for rec, _alt, resname, ch, seq, ic, _name, _el, _xyz in atoms:
        if ch != chain:
            continue
        if rec != "ATOM" and resname not in polymer:
            continue
        key = (seq, ic)
        if key in seen:
            continue
        letter = THREE_TO_ONE.get(resname)
        if letter is None:
            continue
        seen[key] = letter
        order.append(key)
    return "".join(seen[k] for k in order), order


def align(a: str, b: str, match: int = 2, mismatch: int = -1,
          gap: int = -3) -> list[tuple[int, int]]:
    """Needleman-Wunsch over two observed chain sequences, returning paired indices.

    An aligner is needed because apo and holo numbering agree for most pairs and
    not all of them; CryptoBench's own records disagree on 530 of 5493. The two
    sequences here come from the same UniProt accession, so the alignment problem
    is easy and a global one with a fixed scoring scheme is enough. Writing it out
    keeps the correspondence auditable instead of delegating it to a binary whose
    version would have to be pinned and recorded.
    """
    n, m = len(a), len(b)
    if not n or not m:
        return []
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    # 0 diagonal, 1 up (gap in b), 2 left (gap in a)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    back[1:, 0] = 1
    back[0, 1:] = 2
    bv = np.frombuffer(b.encode(), dtype=np.uint8)
    for i in range(1, n + 1):
        sub = np.where(bv == ord(a[i - 1]), match, mismatch).astype(np.int32)
        diag = score[i - 1, :-1] + sub
        up = score[i - 1, 1:] + gap
        row = score[i]
        for j in range(1, m + 1):
            best, arg = diag[j - 1], 0
            if up[j - 1] > best:
                best, arg = up[j - 1], 1
            left = row[j - 1] + gap
            if left > best:
                best, arg = left, 2
            row[j] = best
            back[i, j] = arg
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        d = back[i, j]
        if d == 0:
            i, j = i - 1, j - 1
            if a[i] == b[j]:
                pairs.append((i, j))
        elif d == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def correspondence(apo_atoms: list[tuple], apo_chain: str, apo_poly: set[str],
                   holo_atoms: list[tuple], holo_chain: str,
                   holo_poly: set[str]
                   ) -> tuple[dict[tuple[int, str], tuple[int, str]], float]:
    """holo residue key -> apo residue key, with the identity fraction achieved."""
    sa, ka = chain_sequence(apo_atoms, apo_chain, apo_poly)
    sb, kb = chain_sequence(holo_atoms, holo_chain, holo_poly)
    pairs = align(sa, sb)
    if not pairs or not min(len(sa), len(sb)):
        return {}, 0.0
    return ({kb[j]: ka[i] for i, j in pairs},
            len(pairs) / min(len(sa), len(sb)))


# --------------------------------------------------------------------------
# labelling


def _ligand_copies(atoms: list[tuple], chain: str, accepted: set[str],
                   polymer: set[str]) -> list[tuple[str, str, int]]:
    """The relevant ligand copies to test against this chain: (code, chain, seq).

    A copy is relevant when the benchmark accepted that chemical component as a
    ligand somewhere in its own dataset. Restricting the chemistry to what
    CryptoBench itself admitted avoids inventing a relevance filter here, at the
    cost of excluding components new to the PDB since the cutoff -- which is a
    real cost, and is reported.
    """
    out = set()
    for rec, _alt, resname, ch, seq, _ic, _name, _el, _xyz in atoms:
        if rec != "HETATM" or resname not in accepted or resname in polymer:
            continue
        out.add((resname, ch, seq))
    return sorted(out)


def _occupied(apo_atoms: list[tuple], apo_chain: str, keys: set[tuple[int, str]],
              accepted: set[str], apo_poly: set[str]) -> str | None:
    """Is something already sitting in the apo pocket? Then it is not apo.

    Without this test the set would contain pairs where both structures are bound
    and the pocket never had to open, which is the opposite of a cryptic site.
    """
    pocket = _heavy_by_residue(apo_atoms, apo_chain)
    pts = [xyz for k in keys for xyz in pocket.get(k, {}).values()]
    if not pts:
        return None
    P = np.asarray(pts, dtype=np.float64)
    for code, ch, seq in _ligand_copies(apo_atoms, apo_chain, accepted, apo_poly):
        L = np.asarray([xyz for rec, _a, rn, c, s, _ic, _n, el, xyz in apo_atoms
                        if rec == "HETATM" and rn == code and c == ch and s == seq
                        and (COUNT_HYDROGENS or not _is_hydrogen(el))],
                       dtype=np.float64)
        if len(L) and float(np.sqrt(((P[:, None, :] - L[None]) ** 2).sum(-1)
                                    ).min()) <= CONTACT_ANGSTROM:
            return f"{code}_{ch}{seq}"
    return None


def label_unit(apo_pdb: str, apo_chain: str, partners: list[dict],
               accepted: set[str]) -> dict:
    """Every cryptic pocket of one apo chain, unioned, exactly as the deposit does."""
    ap = path_of(apo_pdb)
    unit = {"apo_pdb": apo_pdb, "apo_chain": apo_chain, "residues": [],
            "pairs": [], "dropped": []}
    if ap is None:
        unit["dropped"].append({"why": "apo structure absent"})
        return unit
    apo_atoms = read_atoms(ap)
    apo_poly = read_seqres(ap).get(apo_chain, set())
    apo_res = _heavy_by_residue(apo_atoms, apo_chain)
    if not apo_res:
        unit["dropped"].append({"why": "apo chain has no heavy atoms"})
        return unit

    positives: set[tuple[int, str]] = set()
    for p in partners:
        hp = path_of(p["pdb"])
        if hp is None:
            unit["dropped"].append({"holo": p["pdb"], "why": "structure absent"})
            continue
        holo_atoms = read_atoms(hp)
        holo_poly = read_seqres(hp).get(p["chain"], set())
        holo_res = _heavy_by_residue(holo_atoms, p["chain"])
        corr, identity = correspondence(apo_atoms, apo_chain, apo_poly,
                                        holo_atoms, p["chain"], holo_poly)
        if identity < 0.90:
            unit["dropped"].append({"holo": p["pdb"], "why": "alignment below 0.90",
                                    "identity": round(identity, 4)})
            continue
        for code, lch, lseq in _ligand_copies(holo_atoms, p["chain"], accepted,
                                              holo_poly):
            got = contact_residues(holo_atoms, p["chain"], code, lch, str(lseq),
                                   holo_poly)
            if not got:
                continue
            holo_sel = sorted(k for k in got if k in corr)
            if len(holo_sel) < 3:
                unit["dropped"].append(
                    {"holo": p["pdb"], "ligand": f"{code}{lseq}",
                     "why": "fewer than three contact residues map to the apo chain"})
                continue
            apo_sel = [corr[k] for k in holo_sel]
            seated = _occupied(apo_atoms, apo_chain, set(apo_sel), accepted,
                               apo_poly)
            if seated:
                unit["dropped"].append(
                    {"holo": p["pdb"], "ligand": f"{code}{lseq}",
                     "why": f"the apo pocket is already occupied by {seated}"})
                continue
            prmsd = pocket_rmsd(apo_res, holo_res, apo_sel, holo_sel)
            if prmsd is None:
                unit["dropped"].append(
                    {"holo": p["pdb"], "ligand": f"{code}{lseq}",
                     "why": f"fewer than {MIN_POCKET_ATOMS_FOR_PRMSD} shared "
                            f"pocket atoms for a pRMSD"})
                continue
            rec = {"holo": p["pdb"], "holo_chain": p["chain"],
                   "ligand": code, "ligand_chain": lch, "ligand_index": str(lseq),
                   "prmsd": round(prmsd, 4), "identity": round(identity, 4),
                   "n_pocket_residues": len(apo_sel)}
            if prmsd >= PRMSD_FLOOR + PRMSD_GUARD_BAND:
                rec["verdict"] = "cryptic"
                positives.update(apo_sel)
            elif prmsd <= PRMSD_FLOOR - PRMSD_GUARD_BAND:
                rec["verdict"] = "not cryptic"
            else:
                rec["verdict"] = "inside the guard band, not labelled either way"
            unit["pairs"].append(rec)
    unit["residues"] = sorted(positives)
    return unit


def mmcif_round_trip() -> dict:
    """Rendering mmCIF as PDB text and reparsing it must not move or lose an atom.

    The receptor for an mmCIF-only entry is written from that rendering, so this
    is checked rather than trusted. Refusals are counted separately: a five-letter
    ligand code has no legacy representation, and that is a limit of the format
    being reported, not an atom going missing.
    """
    cifs = sorted(STRUCTURES.glob("*.cif"))[:N_FORMAT_CHECK]
    rows = [mmcif_atoms.round_trip(c) for c in cifs]
    return {"n_compared": len(rows),
            "n_identical": sum(r["identical"] for r in rows),
            "worst_coordinate_difference": max(
                (r["worst_coordinate_difference"] for r in rows), default=None),
            "failing": [r for r in rows if not r["identical"]][:5]}


# --------------------------------------------------------------------------
# selection


def _pick_apo(chains: list[dict]) -> dict:
    """Best resolution, then most residues, then identifier. A rule, not a taste."""
    return sorted(chains, key=lambda c: (c["resolution"] if c["resolution"]
                                         is not None else 99.0,
                                         -(c["length"] or 0),
                                         c["pdb"], c["chain"]))[0]


def candidates() -> tuple[list[dict], dict]:
    inv = json.loads(gzip.decompress(INVENTORY.read_bytes()))
    uni = json.loads(UNIREF.read_text())
    cb = json.loads(CB_DATASET.read_text())
    accepted = {r["ligand"] for v in cb.values() for r in v}
    cb_acc, cb_clu = set(uni["cryptobench"]), set(uni["cryptobench"].values())

    by: dict[str, dict[str, list]] = {}
    for c in inv["chains"]:
        rel = sorted({l["code"] for l in c["ligands"]
                      if l["code"] in accepted and c["chain"] in l["chains"]})
        side = "holo" if rel else "apo"
        by.setdefault(c["uniprot"], {"holo": [], "apo": []})[side].append(
            dict(c, relevant=rel))

    reasons: dict[str, int] = {}

    def drop(why: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1

    per_cluster: dict[str, list[tuple[str, dict]]] = {}
    for acc, v in sorted(by.items()):
        if not (v["holo"] and v["apo"]):
            drop("the accession has no apo chain and holo chain together")
            continue
        if acc in cb_acc:
            drop("the accession itself appears in CryptoBench")
            continue
        clu = uni["candidate"].get(acc)
        if clu is None:
            drop("UniProt cannot place the accession in a UniRef50 cluster")
            continue
        if clu in cb_clu:
            drop("the accession shares a UniRef50 cluster with CryptoBench")
            continue
        per_cluster.setdefault(clu, []).append((acc, v))

    units: list[dict] = []
    for clu, group in sorted(per_cluster.items()):
        # One unit per cluster, so that neither a popular target nor a screening
        # campaign can weight the set. Within a cluster the accession with the
        # best apo structure wins, by the same rule as the chain choice.
        acc, v = sorted(group, key=lambda g: (
            _pick_apo(g[1]["apo"])["resolution"] or 99.0, g[0]))[0]
        if len(group) > 1:
            drop_n = len(group) - 1
            reasons["a second accession in an already-represented cluster"] = (
                reasons.get("a second accession in an already-represented cluster",
                            0) + drop_n)
        apo = _pick_apo(v["apo"])
        partners = sorted(v["holo"], key=lambda c: (
            c["resolution"] if c["resolution"] is not None else 99.0,
            c["pdb"], c["chain"]))[:MAX_HOLO_PARTNERS]
        units.append({"uniprot": acc, "cluster": clu, "apo": apo,
                      "partners": partners,
                      "n_partners_available": len(v["holo"])})
    return units, {"reasons": reasons, "accepted_ligand_codes": len(accepted),
                   "n_clusters": len(per_cluster), "cutoff": inv["cutoff"],
                   "max_resolution": inv["max_resolution"]}


# --------------------------------------------------------------------------


def build(limit: int | None = None) -> dict:
    units, meta = candidates()
    if limit:
        units = units[:limit]
    print(f"{len(units)} candidate units, one per UniRef50 cluster", flush=True)
    want = sorted({u["apo"]["pdb"] for u in units}
                  | {p["pdb"] for u in units for p in u["partners"]})
    missing = fetch(want)
    cb = json.loads(CB_DATASET.read_text())
    accepted = {r["ligand"] for v in cb.values() for r in v}

    labelled, empty = [], []
    for i, u in enumerate(units, start=1):
        got = label_unit(u["apo"]["pdb"], u["apo"]["chain"], u["partners"],
                         accepted)
        got.update({"uniprot": u["uniprot"], "cluster": u["cluster"],
                    "resolution": u["apo"]["resolution"],
                    "released": u["apo"]["released"],
                    "n_partners_available": u["n_partners_available"]})
        (labelled if got["residues"] else empty).append(got)
        if i % 50 == 0 or i == len(units):
            print(f"  labelled {i}/{len(units)}, "
                  f"{len(labelled)} with a cryptic pocket", flush=True)

    n_pairs = sum(len(u["pairs"]) for u in labelled + empty)
    verdicts: dict[str, int] = {}
    for u in labelled + empty:
        for p in u["pairs"]:
            verdicts[p["verdict"]] = verdicts.get(p["verdict"], 0) + 1
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "question": ("does the counting field hold up on apo-holo pairs that no "
                     "part of its development could have seen"),
        "reads_test_fold": False,
        "why_no_read_is_spent": (
            "this file builds labels from structures released after CryptoBench's "
            "last one and never reads a prediction. CryptoBench's accessions are "
            "read as names to exclude, which leaks no performance"),
        "no_method_has_been_run": True,
        "externality": {
            "temporal": (f"both structures of every pair released after "
                         f"{meta['cutoff']}, the newest release date in "
                         f"CryptoBench"),
            "sequence": ("no accession and no UniRef50 cluster at 50% identity is "
                         "shared with either CryptoBench fold"),
            "order": ("labels exist before any method is run; the preregistration "
                      "that fixes the methods refers to this file by hash"),
        },
        "rule": {
            "contact_angstrom": CONTACT_ANGSTROM,
            "hydrogens_count": COUNT_HYDROGENS,
            "prmsd_floor": PRMSD_FLOOR,
            "guard_band": PRMSD_GUARD_BAND,
            "recovered_by": "tools/recover_cryptobench_rule.py",
        },
        "choices_that_shrink_the_set": {
            "ligand_chemistry": (
                "a ligand counts only if CryptoBench accepted that component "
                "somewhere in its own dataset, which excludes components new to "
                "the PDB since the cutoff and biases the set toward established "
                "cofactors and drug-like series"),
            "one_unit_per_cluster": (
                "the largest campaign here offers 534 apo chains for one protein; "
                "taking one unit per UniRef50 cluster stops any target weighting "
                "the set, and discards a great deal of real data"),
            "holo_partners_capped": MAX_HOLO_PARTNERS,
            "guard_band": (
                "a pair whose pocket moves between "
                f"{PRMSD_FLOOR - PRMSD_GUARD_BAND} and "
                f"{PRMSD_FLOOR + PRMSD_GUARD_BAND} A is not labelled either way, "
                "which removes the borderline cases the recovered pRMSD cannot "
                "reproduce and makes this set easier than CryptoBench in that "
                "one respect"),
            "x_ray_only": f"resolution <= {meta['max_resolution']} A",
        },
        "selection": meta,
        "n_units_with_a_cryptic_pocket": len(labelled),
        "n_units_without_one": len(empty),
        "n_pairs_examined": n_pairs,
        "pair_verdicts": verdicts,
        "reading_mmcif_gives_what_reading_pdb_gives": format_equivalence(),
        "structures_unavailable": missing[:50],
        "n_structures_unavailable": len(missing),
        "units": labelled,
        "units_without_a_cryptic_pocket": [
            {k: v for k, v in u.items() if k != "pairs"} for u in empty],
    }


def write_receptors(payload: dict) -> dict:
    """One receptor PDB per unit, by the same writer the CryptoBench inputs used.

    Using the repository's own writer rather than a fresh one is the point: if the
    external inputs were prepared differently -- hydrogens kept, alternates
    resolved another way, HETATM left in -- a difference in score could be the
    preparation rather than the protein.
    """
    RECEPTORS.mkdir(parents=True, exist_ok=True)
    written, dropped, refusals = {}, [], {}
    for u in payload["units"]:
        src = path_of(u["apo_pdb"])
        dest = RECEPTORS / f"{u['apo_pdb']}_{u['apo_chain']}_receptor.pdb"
        rendered, refused = mmcif_atoms.as_pdb_text(src)
        if refused:
            refusals[f"{u['apo_pdb']}_{u['apo_chain']}"] = refused
        atoms = parse_pdb_atoms(rendered)
        try:
            write_receptor_only_pdb(atoms, dest, chain=u["apo_chain"])
        except ValueError as exc:
            # The writer refuses a receptor under 50 heavy atoms. A chain that
            # small cannot carry a pocket, so it is dropped with the reason
            # rather than written and then quietly scored.
            #
            # The writer names the destination in its message, which is an
            # absolute path on whoever ran it. Reasons are shipped, so the
            # checkout root is stripped: an artifact should not carry somebody's
            # home directory, and a reader cannot use it anyway.
            dropped.append({"unit": dest.stem,
                            "why": str(exc).replace(f"{ROOT}/", "")})
            continue
        body = dest.read_bytes()
        if body.count(b"\nATOM") < 30:
            dropped.append({"unit": dest.stem, "why": "fewer than 30 atoms"})
            dest.unlink()
            continue
        written[dest.name] = hashlib.sha256(body).hexdigest()
    payload["receptors"] = {
        "directory": str(RECEPTORS.relative_to(ROOT)),
        "n": len(written), "sha256": written, "dropped": dropped,
        "writer": "pocket_bench.pdb_io.write_receptor_only_pdb",
        "mmcif_entries_are_rendered_as_pdb_text_first": (
            "so that both formats reach one parser and one receptor writer, "
            "rather than the writer learning a second input shape it could "
            "disagree with itself about"),
        "atoms_the_legacy_format_could_not_hold": refusals,
        "round_trip": mmcif_round_trip(),
    }
    if dropped:
        keep = {d["unit"].rsplit("_receptor", 1)[0] for d in dropped}
        payload["units"] = [u for u in payload["units"]
                            if f"{u['apo_pdb']}_{u['apo_chain']}" not in keep]
        payload["n_units_with_a_cryptic_pocket"] = len(payload["units"])
    return payload


def write_fold_files(payload: dict) -> dict:
    """Per-unit label files and a manifest, in the shape the harness already loads.

    The external set is deliberately given the same on-disk form as the official
    fold: one receptor, one label file, one manifest entry with both hashes. Every
    method and every metric in this repository then reaches it through the code
    path they already use, so a difference in result cannot come from a second
    loader that treats a residue slightly differently.

    Residue numbers are collapsed to integers here, dropping insertion codes,
    because that is the harness convention every method and baseline is indexed
    by. Collapsing can merge two residues, so the collisions are counted rather
    than assumed absent.
    """
    LABELS.mkdir(parents=True, exist_ok=True)
    entries, collisions = [], []
    for u in payload["units"]:
        keys = u["residues"]
        collapsed = sorted({int(k[0]) for k in keys})
        if len(collapsed) != len(keys):
            collisions.append({"unit": f"{u['apo_pdb']}_{u['apo_chain']}",
                               "n_keys": len(keys), "n_after": len(collapsed)})
        lp = LABELS / f"{u['apo_pdb']}_{u['apo_chain']}_labels.json"
        lp.write_text(json.dumps({
            "schema": "cryptobench.external_label.v1",
            "clinical_grade": False,
            "pdb_id": u["apo_pdb"], "chain": u["apo_chain"],
            "cryptic_residues": collapsed,
            "binding_residues": collapsed,
            "uniprot": u["uniprot"], "uniref50": u["cluster"],
            "released": u["released"],
            "n_cryptic_pairs": sum(1 for p in u["pairs"]
                                   if p["verdict"] == "cryptic"),
            "rule": "tools/recover_cryptobench_rule.py, recovered from "
                    "CryptoBench's own training records",
        }, indent=1) + "\n")
        rp = RECEPTORS / f"{u['apo_pdb']}_{u['apo_chain']}_receptor.pdb"
        entries.append({
            "pdb": u["apo_pdb"], "chain": u["apo_chain"],
            "cluster_id": u["cluster"],
            "receptor_path": str(rp.relative_to(ROOT)),
            "receptor_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
            "label_path": str(lp.relative_to(ROOT)),
            "label_sha256": hashlib.sha256(lp.read_bytes()).hexdigest(),
            "split": "external",
        })
    MANIFEST.write_text(json.dumps({
        "schema": "cryptobench.external_validation_set.v1",
        "clinical_grade": False,
        "fold": "external",
        "clustering": {"method": "uniref50", "sequence_identity_threshold": 0.50,
                       "source": "UniProt UniRef50, by accession"},
        "disjoint_from": ("every CryptoBench fold, by accession and by UniRef50 "
                          "cluster"),
        "released_after": payload["selection"]["cutoff"],
        "n_entries": len(entries),
        "residue_numbers_collapsed_to_integers": True,
        "n_units_with_a_collapse_collision": len(collisions),
        "collisions": collisions,
        "entries": entries,
    }, indent=1) + "\n")
    payload["fold_files"] = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "labels_directory": str(LABELS.relative_to(ROOT)),
        "n_entries": len(entries),
        "n_units_with_a_collapse_collision": len(collisions),
    }
    return payload


def _report(d: dict) -> None:
    s = d["selection"]
    print(f"external set: {d['n_units_with_a_cryptic_pocket']} units with a "
          f"cryptic pocket, {d['n_units_without_one']} without, from "
          f"{s['n_clusters']} UniRef50 clusters")
    print(f"  pairs examined {d['n_pairs_examined']}: " + ", ".join(
        f"{n} {k}" for k, n in sorted(d["pair_verdicts"].items(),
                                      key=lambda kv: -kv[1])))
    res = sum(len(u["residues"]) for u in d["units"])
    print(f"  positive residues {res}")
    print("  candidates dropped before labelling:")
    for k, n in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {k}")
    if d["n_structures_unavailable"]:
        print(f"  structures the PDB has no .pdb file for: "
              f"{d['n_structures_unavailable']}")


def check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    _report(d)
    if d["reads_test_fold"] or not d["no_method_has_been_run"]:
        print("FAILED: the set claims to have seen a method or a test fold")
        return 1
    cb_acc = set(json.loads(UNIREF.read_text())["cryptobench"])
    cb_clu = set(json.loads(UNIREF.read_text())["cryptobench"].values())
    for u in d["units"]:
        if u["uniprot"] in cb_acc or u["cluster"] in cb_clu:
            print(f"FAILED: {u['apo_pdb']}_{u['apo_chain']} shares an accession "
                  f"or a UniRef50 cluster with CryptoBench, so it is not external")
            return 1
        if u["released"] <= d["selection"]["cutoff"]:
            print(f"FAILED: {u['apo_pdb']} was released {u['released']}, on or "
                  f"before the {d['selection']['cutoff']} cutoff")
            return 1
    clusters = [u["cluster"] for u in d["units"]]
    if len(set(clusters)) != len(clusters):
        print("FAILED: a UniRef50 cluster contributes more than one unit, so the "
              "units are not independent in the way the analysis will assume")
        return 1
    if d["n_units_with_a_cryptic_pocket"] < 30:
        print(f"FAILED: {d['n_units_with_a_cryptic_pocket']} units is too few to "
              f"say anything with, whatever the result")
        return 1
    r = d.get("receptors") or {}
    if r.get("n") != len(d["units"]):
        print(f"FAILED: {r.get('n')} receptor files for {len(d['units'])} units")
        return 1
    e = d.get("reading_mmcif_gives_what_reading_pdb_gives") or {}
    if e.get("n_compared", 0) < 10:
        print(f"FAILED: the mmCIF reader was compared against the .pdb reader on "
              f"{e.get('n_compared')} entries, too few to rest a quarter of the "
              f"set's structures on")
        return 1
    if e["n_identical"] != e["n_compared"]:
        print(f"FAILED: {e['n_compared'] - e['n_identical']} of "
              f"{e['n_compared']} entries read differently in the two formats, so "
              f"the mmCIF structures are not interchangeable with the .pdb ones: "
              f"{e['disagreeing'][:2]}")
        return 1
    rt = (d.get("receptors") or {}).get("round_trip") or {}
    if rt.get("n_compared", 0) and rt["n_identical"] != rt["n_compared"]:
        print(f"FAILED: {rt['n_compared'] - rt['n_identical']} of "
              f"{rt['n_compared']} mmCIF entries do not survive being rendered as "
              f"PDB text, and that rendering is what the receptors are written "
              f"from: {rt['failing'][:2]}")
        return 1
    f = d.get("fold_files") or {}
    if f.get("n_entries") != len(d["units"]):
        print(f"FAILED: {f.get('n_entries')} manifest entries for "
              f"{len(d['units'])} units")
        return 1
    man = ROOT / f.get("manifest", "")
    if not man.is_file():
        print(f"FAILED: the fold manifest {f.get('manifest')} is absent, so no "
              f"method can be pointed at this set")
        return 1
    if hashlib.sha256(man.read_bytes()).hexdigest() != f["manifest_sha256"]:
        print("FAILED: the fold manifest has changed since the set was frozen")
        return 1
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    payload = write_fold_files(write_receptors(build(limit)))
    payload["code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
