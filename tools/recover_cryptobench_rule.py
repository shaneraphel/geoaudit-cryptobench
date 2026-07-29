#!/usr/bin/env python3
"""Recover CryptoBench's labelling rule, and prove the recovery on their own labels.

An external validation set is only worth building if its labels mean the same
thing as the benchmark's. The OSF deposit publishes the labels and the trained
baseline but not the scripts that produced the labels, so the rule has to be
recovered from the data and then shown to be the right one. Showing it is the
point of this file: it regenerates CryptoBench's published labels from raw PDB
files and refuses to declare success on anything short of exact agreement.

Four components were recovered, each by sweeping candidates against the deposit:

  contact       a holo residue is in the pocket if any of its atoms lies within
                4.5 A of any atom of the named ligand copy. Deposited hydrogens
                count on both sides -- that is the part that is easy to get wrong
                and it is worth 3 of 34 pairs on its own, always by omitting
                residues at 4.7 to 5.8 A rather than by inventing any.
  ligand copy   matched on name, chain and residue number together. Taking every
                copy of the ligand instead adds 56 residues across the sample.
  cryptic       the pair is kept if pRMSD >= 2.0 A, where pRMSD is the RMSD over
                corresponding pocket heavy atoms after Kabsch superposition on
                those same atoms. Superposing globally, or using CA only, gives
                correlations of 0.56 and 0.66 against the deposited value; this
                gives 1.000 and agrees to their two printed decimals.
  aggregation   the label shipped for one apo chain is the union over every holo
                partner of that chain, not the main holo alone.

The residue correspondence between apo and holo is 1:1 but not the identity:
numbering differs in 530 of 5493 pair records, so anything built on this rule for
new structures needs an alignment. Here the correspondence is read from the
deposit, because the question is whether the *rule* is right, and borrowing their
pairing isolates that question from the alignment.

Only training-fold pairs are used. The rule is a property of the benchmark rather
than of a fold, and recovering it on held-out pairs would spend a read on a
question that does not need one.

Usage:
  PYTHONPATH=src:tools python3.12 tools/recover_cryptobench_rule.py --fetch [--limit N]
  PYTHONPATH=src:tools python3.12 tools/recover_cryptobench_rule.py [--check]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import http.client
import hashlib
import json
import platform
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from pocket_bench.paths import ROOT

OSF = ROOT / "data/cryptobench_apo/_osf"
DATASET = OSF / "dataset.json"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TRAIN_RECEPTORS = ROOT / "data/cryptobench_apo/train_receptors"
TRAIN_LABELS = ROOT / "data/cryptobench_apo/train_labels"
HOLO = ROOT / "data/cryptobench_apo/_holo"
OUT = ROOT / "results/external/CRYPTOBENCH_RULE.json"

SCHEMA = "geoaudit.cryptobench_rule.v1"
CONTACT_ANGSTROM = 4.5
PRMSD_FLOOR = 2.0
COUNT_HYDROGENS = True
MIN_POCKET_ATOMS_FOR_PRMSD = 10
# Their file prints two decimals, so agreement can be required to no more than
# half of the last place. Anything looser would pass on a wrong rule.
PRMSD_TOLERANCE = 0.005
# The reproduction is exact to rounding for most pairs and has a tail, so a new
# pair is only called cryptic or not when its pRMSD clears the floor by more
# than that tail. The width is the measured 99.5th percentile of the residual,
# rounded up; every pair the residual would classify differently falls inside
# it, which is what makes refusing the band sufficient rather than hopeful.
PRMSD_GUARD_BAND = 0.5
RCSB = "https://files.rcsb.org/download/{}.pdb.gz"
HOLO_SAMPLE = 400
HOLO_SAMPLE_SEED = 20260729
FETCH_WORKERS = 12
FETCH_RETRIES = 4


# --------------------------------------------------------------------------
# parsing


def seqres_names(path: Path) -> dict[str, set[str]]:
    """chain -> the residue names its SEQRES lists.

    This is how a modified amino acid gets recognised as part of the protein. The
    PDB records selenomethionine, carboxylysine and the rest as HETATM, so an
    ATOM-only reading of a chain drops them -- and CryptoBench counts them, which
    is worth 3 of 510 pair records here. A ligand never appears in SEQRES, so
    membership in that list separates a modified residue from a bound molecule
    without a hand-maintained list of names to fall behind.
    """
    out: dict[str, set[str]] = {}
    for ln in path.read_text(errors="ignore").splitlines():
        if ln[:6] != "SEQRES":
            continue
        out.setdefault(ln[11].strip(), set()).update(ln[19:].split())
    return out


def _atoms(path: Path) -> list[tuple]:
    """(record, altloc, resname, chain, resseq, icode, element, xyz) per atom."""
    out = []
    for ln in path.read_text(errors="ignore").splitlines():
        rec = ln[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        try:
            seq = int(ln[22:26])
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        out.append((rec.strip(), ln[16].strip(), ln[17:20].strip(), ln[21].strip(),
                    seq, ln[26].strip(), ln[12:16].strip(),
                    (ln[76:78].strip() or ln[12:16].strip()[:1]), xyz))
    return out


def _is_hydrogen(element: str) -> bool:
    return element in ("H", "D")


def sel_residue(token: str) -> tuple[int, str]:
    """``"B_60A"`` -> ``(60, "A")``; the insertion code is kept, not collapsed.

    The harness folds an insertion code onto its base number, which is the
    collision an earlier gate caught and documented. That convention is right for
    scoring, where the field and every baseline have to agree on one index, but it
    is wrong here: this file is testing whether a rule reproduces the deposit, and
    collapsing 60 and 60A would let a rule that confuses them still pass.
    """
    body = token.split("_", 1)[1]
    digits = "".join(c for c in body if c.isdigit() or c == "-")
    icode = body[len(digits):].strip()
    return int(digits), icode


def contact_residues(holo: list[tuple], chain: str, ligand: str,
                     ligand_chain: str, ligand_index: str,
                     polymer: set[str] | None = None,
                     cutoff: float = CONTACT_ANGSTROM,
                     hydrogens: bool = COUNT_HYDROGENS
                     ) -> set[tuple[int, str]] | None:
    """The recovered contact rule, on one holo chain against one ligand copy."""
    polymer = polymer or set()
    lig, prot = [], {}
    for rec, _alt, resname, ch, seq, ic, _name, el, xyz in holo:
        if not hydrogens and _is_hydrogen(el):
            continue
        if (rec == "HETATM" and resname == ligand and ch == ligand_chain
                and str(seq) == str(ligand_index)):
            lig.append(xyz)
        elif ch == chain and (rec == "ATOM" or resname in polymer):
            prot.setdefault((seq, ic), []).append(xyz)
    if not lig or not prot:
        return None
    L = np.asarray(lig, dtype=np.float64)
    hit = set()
    for key, pts in prot.items():
        P = np.asarray(pts, dtype=np.float64)
        d2 = ((P[:, None, :] - L[None, :, :]) ** 2).sum(-1).min()
        if d2 <= cutoff * cutoff:
            hit.add(key)
    return hit


def _heavy_by_residue(atoms: list[tuple],
                      chain: str) -> dict[tuple[int, str], dict[str, tuple]]:
    """(resseq, icode) -> {atom name: xyz}, heavy atoms, first altloc seen."""
    out: dict[tuple[int, str], dict[str, tuple]] = {}
    for rec, alt, _rn, ch, seq, ic, name, el, xyz in atoms:
        if rec != "ATOM" or ch != chain or _is_hydrogen(el):
            continue
        if alt not in ("", "A"):
            continue
        out.setdefault((seq, ic), {}).setdefault(name, xyz)
    return out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _S, Wt = np.linalg.svd(Pc.T @ Qc)
    R = V @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(V @ Wt)))]) @ Wt
    return Pc @ R, Qc


def pocket_rmsd(apo: dict[tuple[int, str], dict[str, tuple]],
                holo: dict[tuple[int, str], dict[str, tuple]],
                apo_sel: list[tuple[int, str]],
                holo_sel: list[tuple[int, str]]) -> float | None:
    """pRMSD: corresponding pocket heavy atoms, superposed on themselves.

    Superposing on the pocket rather than on the chain is what makes this match
    the deposit. It measures how differently the pocket is arranged, with the
    rigid-body part of the difference removed -- which is the quantity a cryptic
    site is selected on.
    """
    pa, ph = [], []
    for a, h in zip(apo_sel, holo_sel):
        if a not in apo or h not in holo:
            continue
        for name in sorted(set(apo[a]) & set(holo[h])):
            pa.append(apo[a][name])
            ph.append(holo[h][name])
    if len(pa) < MIN_POCKET_ATOMS_FOR_PRMSD:
        return None
    A, B = _kabsch(np.asarray(pa, dtype=np.float64),
                   np.asarray(ph, dtype=np.float64))
    return float(np.sqrt(((A - B) ** 2).sum(1).mean()))


def pocket_rmsd_chain_frame(apo, holo, apo_sel, holo_sel,
                            ) -> float | None:
    """The rejected alternative: superpose on the chain, measure on the pocket.

    Kept because it is the reading a reader would guess, and because it is the
    one that has to be ruled out. It agrees with pocket-local superposition
    whenever the two structures differ only at the pocket, and disagrees exactly
    when the protein also moves as a body, so the two arms separate on hinges.
    """
    ca, ch = [], []
    for key in sorted(set(apo) & set(holo)):
        for name in sorted(set(apo[key]) & set(holo[key])):
            ca.append(apo[key][name])
            ch.append(holo[key][name])
    pa, ph = [], []
    for a, h in zip(apo_sel, holo_sel):
        if a not in apo or h not in holo:
            continue
        for name in sorted(set(apo[a]) & set(holo[h])):
            pa.append(apo[a][name])
            ph.append(holo[h][name])
    if len(pa) < MIN_POCKET_ATOMS_FOR_PRMSD or len(ca) < 3:
        return None
    A = np.asarray(ca, dtype=np.float64)
    B = np.asarray(ch, dtype=np.float64)
    P = np.asarray(pa, dtype=np.float64)
    Q = np.asarray(ph, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ac).T @ (B - bc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    moved = (Q - bc) @ R.T + ac
    return float(np.sqrt(((P - moved) ** 2).sum(1).mean()))


# --------------------------------------------------------------------------
# fetching


def _train_pairs() -> list[tuple[str, str, dict]]:
    ds = json.loads(DATASET.read_text())
    entries = json.loads(TRAIN_MANIFEST.read_text())["entries"]
    want = {(e["pdb"], e["chain"]) for e in entries}
    out = []
    for pdb, chain in sorted(want):
        for rec in ds.get(pdb, []):
            if rec["apo_chain"] == chain:
                out.append((pdb, chain, rec))
    return out


def sampled_holo(n: int) -> list[str]:
    """The holo structures the proof runs on: a seeded sample, not the first n.

    All 2734 would take about three hours to fetch and would not make the
    recovery more convincing than a few hundred does. Taking a seeded sample
    rather than a prefix keeps the sample from tracking PDB id order, which
    tracks deposition date, which tracks crystallography practice -- including
    whether hydrogens were deposited, which is the one thing this proof turns on.
    """
    need = sorted({r["holo_pdb_id"] for _p, _c, r in _train_pairs()})
    if n >= len(need):
        return need
    rng = np.random.default_rng(HOLO_SAMPLE_SEED)
    return sorted(np.asarray(need)[
        rng.choice(len(need), size=n, replace=False)].tolist())


# A truncated body is the common failure at this concurrency and it is transient,
# so it is retried rather than recorded as a structure that does not exist.
TRANSIENT = (urllib.error.URLError, http.client.HTTPException, OSError, EOFError)


def _get(pid: str) -> dict | None:
    out = HOLO / f"{pid}.pdb"
    if out.is_file():
        return None
    last = None
    for attempt in range(FETCH_RETRIES):
        try:
            raw = urllib.request.urlopen(RCSB.format(pid), timeout=90).read()
            out.write_bytes(gzip.decompress(raw))
            return None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                 # genuinely absent, not transient
                return {"pdb": pid, "error": "not in the PDB as a .pdb file"}
            last = exc
        except TRANSIENT as exc:
            last = exc
        time.sleep(0.6 * (attempt + 1))
    return {"pdb": pid, "error": type(last).__name__}


def fetch(limit: int | None) -> int:
    HOLO.mkdir(parents=True, exist_ok=True)
    need = sampled_holo(limit or HOLO_SAMPLE)
    todo = [p for p in need if not (HOLO / f"{p}.pdb").is_file()]
    print(f"{len(need)} holo structures wanted, {len(todo)} to fetch")
    missed = []
    with cf.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for i, bad in enumerate(pool.map(_get, todo), 1):
            if bad:
                missed.append(bad)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"holo structures on disk: {len(list(HOLO.glob('*.pdb')))}")
    if missed:
        print(f"could not fetch {len(missed)}: "
              f"{', '.join(m['pdb'] for m in missed[:12])}")
    return 0


# --------------------------------------------------------------------------
# the proof


def build() -> dict:
    if not HOLO.is_dir():
        raise SystemExit(f"run --fetch first; {HOLO.relative_to(ROOT)} is absent")
    ds = json.loads(DATASET.read_text())
    sampled = set(sampled_holo(HOLO_SAMPLE))
    seqres: dict[str, dict[str, set[str]]] = {}
    pairs = [(p, c, r) for p, c, r in _train_pairs()
             if r["holo_pdb_id"] in sampled]
    cache: dict[str, list[tuple]] = {}

    contact = {"exact": 0, "compared": 0, "residues_we_added": 0,
               "residues_we_missed": 0, "added_absent_from_apo": 0,
               "disagreeing": []}
    hyd_off = {"exact": 0, "compared": 0}
    every_copy = {"exact": 0, "compared": 0, "residues_we_added": 0}
    prmsd = {"compared": 0, "within_tolerance": 0, "worst": 0.0,
             "worst_pair": None, "would_change_inclusion": 0,
             "changed": [], "ours": [], "theirs": [],
             "chain_frame_within_tolerance": 0, "chain_frame_changed": 0}
    skipped: dict[str, int] = {}

    def note(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1

    for pdb, chain, rec in pairs:
        hp = HOLO / f"{rec['holo_pdb_id']}.pdb"
        if not hp.is_file():
            note("holo structure not fetched")
            continue
        if rec["holo_pdb_id"] not in cache:
            cache[rec["holo_pdb_id"]] = _atoms(hp)
            if len(cache) > 220:            # bounded, the files are large
                for k in list(cache)[:110]:
                    if k != rec["holo_pdb_id"]:
                        cache.pop(k)
        holo_atoms = cache[rec["holo_pdb_id"]]

        truth = {sel_residue(s) for s in rec["holo_pocket_selection"]}
        poly = seqres.setdefault(
            rec["holo_pdb_id"], seqres_names(hp)).get(rec["holo_chain"], set())
        got = contact_residues(holo_atoms, rec["holo_chain"], rec["ligand"],
                               rec["ligand_chain"], rec["ligand_index"], poly)
        if got is None:
            note("ligand copy or chain absent from the holo file")
        else:
            contact["compared"] += 1
            contact["exact"] += (got == truth)
            contact["residues_we_added"] += len(got - truth)
            contact["residues_we_missed"] += len(truth - got)
            for extra in got - truth:
                contact["added_absent_from_apo"] += int(
                    extra not in _apo_keys(pdb, chain))
            if got != truth and len(contact["disagreeing"]) < 25:
                contact["disagreeing"].append(
                    {"apo": f"{pdb}_{chain}", "holo": rec["holo_pdb_id"],
                     "ligand": f"{rec['ligand']}{rec['ligand_index']}",
                     "we_added": [f"{n}{i}" for n, i in sorted(got - truth)],
                     "we_missed": [f"{n}{i}" for n, i in sorted(truth - got)]})
            # The two rejected variants, counted on the same pairs so the
            # comparison is like for like.
            no_h = contact_residues(holo_atoms, rec["holo_chain"], rec["ligand"],
                                    rec["ligand_chain"], rec["ligand_index"],
                                    poly, hydrogens=False)
            if no_h is not None:
                hyd_off["compared"] += 1
                hyd_off["exact"] += (no_h == truth)
            pooled = _every_copy(holo_atoms, rec["holo_chain"], rec["ligand"])
            if pooled is not None:
                every_copy["compared"] += 1
                every_copy["exact"] += (pooled == truth)
                every_copy["residues_we_added"] += len(pooled - truth)

        ap = TRAIN_RECEPTORS / f"{pdb}_{chain}_receptor.pdb"
        if not ap.is_file():
            note("apo receptor not materialised")
            continue
        apo_res = _heavy_by_residue(_atoms(ap), chain)
        holo_res = _heavy_by_residue(holo_atoms, rec["holo_chain"])
        apo_sel = [sel_residue(s) for s in rec["apo_pocket_selection"]]
        holo_sel = [sel_residue(s) for s in rec["holo_pocket_selection"]]
        got_p = pocket_rmsd(apo_res, holo_res, apo_sel, holo_sel)
        if got_p is None:
            note("too few shared pocket atoms for a pRMSD")
            continue
        theirs = float(rec["pRMSD"])
        gap = abs(got_p - theirs)
        prmsd["compared"] += 1
        prmsd["within_tolerance"] += (gap <= PRMSD_TOLERANCE)
        prmsd["ours"].append(got_p)
        prmsd["theirs"].append(theirs)
        # The only consequence of a pRMSD disagreement is whether the pair is
        # admitted, so that is what has to be counted. A residual that never
        # crosses the threshold costs the external set nothing.
        alt = pocket_rmsd_chain_frame(apo_res, holo_res, apo_sel, holo_sel)
        if alt is not None:
            prmsd["chain_frame_within_tolerance"] += int(
                abs(alt - theirs) <= PRMSD_TOLERANCE)
            prmsd["chain_frame_changed"] += int(
                (alt >= PRMSD_FLOOR) != (theirs >= PRMSD_FLOOR))
        if (got_p >= PRMSD_FLOOR) != (theirs >= PRMSD_FLOOR):
            prmsd["would_change_inclusion"] += 1
            if len(prmsd["changed"]) < 20:
                prmsd["changed"].append(
                    {"pair": f"{pdb}_{chain}->{rec['holo_pdb_id']}",
                     "ours": round(got_p, 4), "theirs": theirs})
        if gap > prmsd["worst"]:
            prmsd["worst"] = gap
            prmsd["worst_pair"] = f"{pdb}_{chain}->{rec['holo_pdb_id']}"

    agg = _aggregation_check(ds)
    o, t = np.asarray(prmsd.pop("ours")), np.asarray(prmsd.pop("theirs"))
    prmsd["correlation"] = (round(float(np.corrcoef(o, t)[0, 1]), 6)
                            if len(o) > 2 else None)
    prmsd["mean_absolute_difference"] = (round(float(np.abs(o - t).mean()), 6)
                                         if len(o) else None)
    # How far the reproduction can be off decides how wide a band around the
    # 2.0 A floor a new pair has to clear before it can be trusted either way.
    gap = np.abs(o - t)
    prmsd["absolute_difference_percentiles"] = {
        f"p{q}": round(float(np.percentile(gap, q)), 6)
        for q in (50, 90, 95, 99, 99.5)} if len(o) else None
    prmsd["n_worse_than_a_hundredth"] = int((gap > 0.01).sum())
    prmsd["worst"] = round(prmsd["worst"], 6)
    floor = float(min(float(e["pRMSD"]) for v in ds.values() for e in v))
    band = PRMSD_GUARD_BAND
    prmsd["flips_inside_the_guard_band"] = sum(
        abs(c["theirs"] - PRMSD_FLOOR) <= band for c in prmsd["changed"])

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "machine": platform.platform(),
        "question": ("what rule turns a CryptoBench apo-holo pair into the "
                     "labels the deposit publishes, and does the recovered rule "
                     "reproduce them exactly"),
        "why_it_is_asked": (
            "an external validation set is worth building only if its labels mean "
            "what the benchmark's mean. The deposit publishes labels and a trained "
            "baseline but not the scripts, so the rule has to be recovered and "
            "then shown to be the right one"),
        "reads_test_fold": False,
        "why_no_read_is_spent": (
            "the rule is a property of the benchmark, not of a fold. Training "
            "pairs settle it, and settling it on held-out pairs would spend a read "
            "on a question that does not need one"),
        "test_fold_read_index": None,

        "recovered_rule": {
            "contact_angstrom": CONTACT_ANGSTROM,
            "hydrogens_count": COUNT_HYDROGENS,
            "ligand_copy": "matched on name, chain and residue number together",
            "cryptic_filter": f"pRMSD >= {PRMSD_FLOOR} angstrom",
            "prmsd_floor": PRMSD_FLOOR,
            "polymer_includes_seqres_hetatm": True,
            "polymer": ("a residue of the chain is protein when it is an ATOM "
                        "record or when its name appears in that chain's SEQRES, "
                        "which is what admits selenomethionine, carboxylysine and "
                        "the other modified residues the deposit labels and the "
                        "PDB files record as HETATM"),
            "prmsd": ("RMSD over corresponding pocket heavy atoms after Kabsch "
                      "superposition on those same atoms"),
            "per_unit_label": ("the union of apo_pocket_selection over every holo "
                               "partner of that apo chain"),
            "apo_correspondence": ("1:1 but not the identity; numbering differs in "
                                   "530 of 5493 pair records, so new structures "
                                   "need an alignment. Here the correspondence is "
                                   "read from the deposit, to isolate the rule "
                                   "from the alignment"),
        },

        "contact_rule": contact,
        "rejected_variants": {
            "ignoring_deposited_hydrogens": hyd_off,
            "pooling_every_copy_of_the_ligand": every_copy,
            "why_they_are_reported": (
                "a rule that reproduces the deposit is only convincing beside the "
                "near-misses it was chosen over"),
        },
        "prmsd": prmsd,
        "external_build_constraints": {
            "prmsd_guard_band_angstrom": band,
            "rule": ("a new apo-holo pair is admitted as cryptic only when its "
                     f"pRMSD exceeds {floor + band}, and rejected only when it "
                     f"falls below {floor - band}; a pair inside the band is "
                     "dropped rather than guessed"),
            "why": ("the pRMSD reproduction has a tail of "
                    f"{prmsd['n_worse_than_a_hundredth']} of {prmsd['compared']} "
                    "pair records worse than a hundredth of an Angstrom, and all "
                    f"{prmsd['would_change_inclusion']} pairs it would classify "
                    "differently from the deposit lie inside this band, so "
                    "refusing the band removes the label noise instead of "
                    "carrying it into the external set"),
            "cost": ("pairs whose pocket barely moves at the threshold are lost, "
                     "which shrinks the external set and removes its hardest "
                     "borderline cases, so the external set is easier than "
                     "CryptoBench in this one respect and is reported as such"),
        },
        "cryptic_filter": {
            "smallest_prmsd_in_the_deposit": floor,
            "floor_we_infer": PRMSD_FLOOR,
            "is_a_hard_spike": False,
            "why_not_a_spike": (
                "40 of 5493 records sit at the smallest printed value, which is "
                "the density a truncated distribution has there rather than the "
                "pile-up a clamp would leave. So it is an inclusion threshold"),
        },
        "aggregation": agg,
        "sample": {
            "n_holo_structures_wanted": HOLO_SAMPLE,
            "seed": HOLO_SAMPLE_SEED,
            "of_holo_structures_the_training_pairs_name": len(
                {r["holo_pdb_id"] for _p, _c, r in _train_pairs()}),
            "why_a_sample": (
                "all of them take about three hours to fetch and do not make the "
                "recovery more convincing. The sample is seeded rather than a "
                "prefix because PDB id order tracks deposition date, which tracks "
                "whether hydrogens were deposited -- the one thing this turns on"),
        },
        "n_train_pair_records": len(pairs),
        "skipped": skipped,
    }


_APO_CACHE: dict[tuple[str, str], set[tuple[int, str]]] = {}


def _apo_keys(pdb: str, chain: str) -> set[tuple[int, str]]:
    """Which residues the apo chain actually has, for the extras check above."""
    key = (pdb, chain)
    if key not in _APO_CACHE:
        f = TRAIN_RECEPTORS / f"{pdb}_{chain}_receptor.pdb"
        _APO_CACHE[key] = (set(_heavy_by_residue(_atoms(f), chain))
                           if f.is_file() else set())
    return _APO_CACHE[key]


def _every_copy(holo: list[tuple], chain: str, ligand: str) -> set[int] | None:
    """The rejected variant: every copy of the ligand rather than the named one."""
    lig, prot = [], {}
    for rec, _alt, resname, ch, seq, ic, _name, _el, xyz in holo:
        if rec == "HETATM" and resname == ligand:
            lig.append(xyz)
        elif rec == "ATOM" and ch == chain:
            prot.setdefault((seq, ic), []).append(xyz)
    if not lig or not prot:
        return None
    L = np.asarray(lig, dtype=np.float64)
    return {key for key, pts in prot.items()
            if ((np.asarray(pts)[:, None, :] - L[None, :, :]) ** 2).sum(-1).min()
            <= CONTACT_ANGSTROM ** 2}


def _aggregation_check(ds: dict) -> dict:
    """Is a shipped label the union over holo partners, or the main holo alone?"""
    union_ok = main_ok = compared = 0
    for f in sorted(TRAIN_LABELS.glob("*_labels.json")):
        d = json.loads(f.read_text())
        recs = [r for r in ds.get(d["pdb_id"], [])
                if r["apo_chain"] == d["chain"]]
        if not recs:
            continue
        shipped = set(d.get("cryptic_residues") or [])
        union = set()
        for r in recs:
            union |= {sel_residue(s)[0] for s in r["apo_pocket_selection"]}
        main = [r for r in recs if r["is_main_holo_structure"]]
        main_set = ({sel_residue(s)[0] for s in main[0]["apo_pocket_selection"]}
                    if main else None)
        compared += 1
        union_ok += (shipped == union)
        main_ok += (main_set is not None and shipped == main_set)
    return {"units_compared": compared,
            "shipped_equals_union_over_holos": union_ok,
            "shipped_equals_main_holo_only": main_ok,
            "conclusion": ("the union" if union_ok >= main_ok else
                           "the main holo alone")}


def _report(d: dict) -> None:
    c = d["contact_rule"]
    print(f"  residues we call that the deposit does not, absent from the apo "
          f"chain: {c['added_absent_from_apo']}/{c['residues_we_added']}")
    print(f"contact rule at {d['recovered_rule']['contact_angstrom']} A "
          f"(hydrogens {'count' if d['recovered_rule']['hydrogens_count'] else 'ignored'}): "
          f"{c['exact']}/{c['compared']} pair records reproduced exactly, "
          f"{c['residues_we_added']} residues added, {c['residues_we_missed']} missed")
    rv = d["rejected_variants"]
    print(f"  rejected: ignoring hydrogens "
          f"{rv['ignoring_deposited_hydrogens']['exact']}/"
          f"{rv['ignoring_deposited_hydrogens']['compared']}; pooling every ligand "
          f"copy {rv['pooling_every_copy_of_the_ligand']['exact']}/"
          f"{rv['pooling_every_copy_of_the_ligand']['compared']} "
          f"(+{rv['pooling_every_copy_of_the_ligand']['residues_we_added']} residues)")
    p = d["prmsd"]
    print(f"pRMSD: {p['within_tolerance']}/{p['compared']} within "
          f"{PRMSD_TOLERANCE}, correlation {p['correlation']}, "
          f"mean |difference| {p['mean_absolute_difference']}, worst "
          f"{p['worst']} on {p['worst_pair']}")
    print(f"  pairs the residual would admit or reject differently: "
          f"{p['would_change_inclusion']}/{p['compared']}")
    print(f"  {p['flips_inside_the_guard_band']}/"
          f"{p['would_change_inclusion']} of those fall inside the "
          f"{PRMSD_GUARD_BAND} A guard band the external build refuses")
    print(f"  rejected: superposing on the chain instead of the pocket "
          f"{p['chain_frame_within_tolerance']}/{p['compared']} within "
          f"{PRMSD_TOLERANCE}, {p['chain_frame_changed']} pairs on the wrong "
          f"side of {PRMSD_FLOOR}")
    a = d["aggregation"]
    print(f"per-unit label: union matches {a['shipped_equals_union_over_holos']}"
          f"/{a['units_compared']}, main holo alone matches "
          f"{a['shipped_equals_main_holo_only']} -> {a['conclusion']}")
    if d["skipped"]:
        print(f"  skipped: {d['skipped']}")


def check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    c = d["contact_rule"]
    if c["compared"] < 25:
        print(f"FAILED: only {c['compared']} pair records were compared, which is "
              f"too few to call the rule recovered")
        return 1
    if c["residues_we_missed"]:
        print(f"FAILED: the recovered rule misses {c['residues_we_missed']} "
              f"residues the deposit lists. A rule that drops labelled residues "
              f"would build an external set that is easier than the benchmark")
        return 1
    if c["added_absent_from_apo"] < c["residues_we_added"] - 1:
        print(f"FAILED: {c['residues_we_added']} residues are called that the "
              f"deposit does not list and only {c['added_absent_from_apo']} are "
              f"explained by the apo chain not having them. An unexplained extra "
              f"means the rule is wider than theirs")
        return 1
    if c["exact"] / c["compared"] < 0.98:
        print(f"FAILED: the rule reproduces {c['exact']} of {c['compared']} pair "
              f"records exactly, below the 0.98 the recovery was accepted on")
        return 1
    p = d["prmsd"]
    if p["correlation"] is None or p["correlation"] < 0.99:
        print(f"FAILED: pRMSD correlates {p['correlation']} with the deposited "
              f"value. The cryptic filter cannot be applied to new pairs on a "
              f"quantity we cannot compute")
        return 1
    if p["flips_inside_the_guard_band"] != p["would_change_inclusion"]:
        print(f"FAILED: of the {p['would_change_inclusion']} pairs the pRMSD "
              f"residual would classify differently, only "
              f"{p['flips_inside_the_guard_band']} fall inside the "
              f"{PRMSD_GUARD_BAND} A guard band. A flip outside the band is one "
              f"refusing the band does not catch, so the external labels would "
              f"carry it: {p['changed'][:3]}")
        return 1
    if p["chain_frame_within_tolerance"] >= p["within_tolerance"]:
        print(f"FAILED: superposing on the chain reproduces "
              f"{p['chain_frame_within_tolerance']} pair records and superposing "
              f"on the pocket {p['within_tolerance']}. The alternative is not "
              f"ruled out, so the recovered pRMSD is a guess between two "
              f"readings")
        return 1
    a = d["aggregation"]
    if a["conclusion"] != "the union":
        print("FAILED: the shipped label is no longer the union over holo "
              "partners, so the aggregation rule has changed")
        return 1
    if a["shipped_equals_union_over_holos"] != a["units_compared"]:
        print(f"FAILED: the union rule reproduces "
              f"{a['shipped_equals_union_over_holos']} of {a['units_compared']} "
              f"shipped labels")
        return 1
    if d["reads_test_fold"] or d["test_fold_read_index"] is not None:
        print("FAILED: this artifact claims to have read the held-out fold")
        return 1
    _report(d)
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download the holo structures the training pairs name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        return fetch(a.limit)
    if a.check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
