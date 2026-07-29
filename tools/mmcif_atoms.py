"""Read an mmCIF entry into exactly the tuples the recovered rule expects.

A quarter of the candidate structures for the external set have no legacy .pdb
file: the format cannot hold them, so the PDB does not offer one. Skipping them
would drop 283 apo-holo pairs, and the ones it drops are not a random sample --
they are the large assemblies, which is a bias in the direction of small, easy
proteins.

The rule this set is labelled with was validated against .pdb parsing, so reading
mmCIF is only safe if it produces the same thing. That is a testable claim rather
than an assumption: for every entry the PDB publishes in both formats, the two
readings must agree atom for atom. ``agreement`` measures it and the builder's gate
refuses to use this module until it does.

Only the first model is read, and the auth_* numbering is used throughout, because
that is the numbering the .pdb format carries and the numbering every label,
prediction and residue key in this repository is written in.
"""
from __future__ import annotations

import gzip
from pathlib import Path

WANT = ("group_PDB", "label_alt_id", "auth_comp_id", "auth_asym_id",
        "auth_seq_id", "pdbx_PDB_ins_code", "auth_atom_id", "type_symbol",
        "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num", "label_comp_id",
        "label_atom_id", "label_asym_id", "label_seq_id")


def _tokens(line: str) -> list[str]:
    """Split one mmCIF data row, honouring single and double quotes.

    Atom names like ``'O5''`` and residue names carrying spaces are why this
    cannot be ``line.split()``.
    """
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            j = i + 1
            while j < n:
                if line[j] == c and (j + 1 >= n or line[j + 1] in " \t"):
                    break
                j += 1
            out.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and line[j] not in " \t":
                j += 1
            out.append(line[i:j])
            i = j
    return out


def _loop(text: str, name: str) -> tuple[list[str], list[list[str]]]:
    """The named category's column names and rows, from a loop_ or a key-value block."""
    lines = text.splitlines()
    prefix = name + "."
    i = 0
    while i < len(lines):
        if lines[i].startswith(prefix):
            # A single-row category is written as key-value pairs, not a loop.
            cols, row = [], []
            while i < len(lines) and lines[i].startswith(prefix):
                parts = _tokens(lines[i])
                cols.append(parts[0][len(prefix):])
                row.append(parts[1] if len(parts) > 1 else "?")
                i += 1
            return cols, [row]
        if lines[i].strip() == "loop_":
            j, cols = i + 1, []
            while j < len(lines) and lines[j].startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith(prefix):
                names = [c[len(prefix):] for c in cols]
                rows = []
                while j < len(lines):
                    s = lines[j]
                    if not s or s[0] == "#" or s.startswith("loop_") \
                            or s.startswith("_") or s.startswith("data_"):
                        break
                    if s.startswith(";"):
                        j += 1
                        continue
                    t = _tokens(s)
                    if len(t) == len(names):
                        rows.append(t)
                    j += 1
                return names, rows
            i = j
            continue
        i += 1
    return [], []


def _text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="ignore")


def atoms(path: Path) -> list[tuple]:
    """(record, altloc, resname, chain, resseq, icode, name, element, xyz) per atom.

    Identical in shape and convention to the .pdb reader the rule was recovered
    with, so the two are interchangeable at the call site.
    """
    text = _text(path)
    names, rows = _loop(text, "_atom_site")
    if not rows:
        return []
    idx = {n: k for k, n in enumerate(names)}

    def col(row: list[str], *options: str) -> str:
        for o in options:
            k = idx.get(o)
            if k is not None and row[k] not in ("?", "."):
                return row[k]
        return ""

    out: list[tuple] = []
    model_key = idx.get("pdbx_PDB_model_num")
    first_model = rows[0][model_key] if model_key is not None else None
    for row in rows:
        if model_key is not None and row[model_key] != first_model:
            continue
        rec = col(row, "group_PDB")
        try:
            seq = int(col(row, "auth_seq_id", "label_seq_id"))
            xyz = (float(row[idx["Cartn_x"]]), float(row[idx["Cartn_y"]]),
                   float(row[idx["Cartn_z"]]))
        except (KeyError, ValueError):
            continue
        out.append((
            "ATOM" if rec == "ATOM" else "HETATM",
            col(row, "label_alt_id"),
            col(row, "auth_comp_id", "label_comp_id"),
            col(row, "auth_asym_id", "label_asym_id"),
            seq,
            col(row, "pdbx_PDB_ins_code"),
            col(row, "auth_atom_id", "label_atom_id"),
            col(row, "type_symbol"),
            xyz,
        ))
    return out


def seqres_names(path: Path) -> dict[str, set[str]]:
    """chain -> the residue names its polymer sequence lists, as SEQRES would.

    ``_pdbx_poly_seq_scheme`` carries the auth chain id next to every polymer
    residue name, which is the same fact SEQRES carries in a .pdb file, so a
    modified residue recorded as HETATM is still recognised as protein.
    """
    text = _text(path)
    names, rows = _loop(text, "_pdbx_poly_seq_scheme")
    out: dict[str, set[str]] = {}
    if rows:
        idx = {n: k for k, n in enumerate(names)}
        ch = idx.get("pdb_strand_id", idx.get("asym_id"))
        mon = idx.get("mon_id")
        if ch is not None and mon is not None:
            for row in rows:
                out.setdefault(row[ch], set()).add(row[mon])
    return out


COORDINATE_TOLERANCE = 0.002


def agreement(cif: Path, pdb: Path) -> dict:
    """Do the two formats of one entry read the same? Reported, not assumed.

    The discrete fields -- record type, residue name, chain, residue number,
    insertion code, atom name, element -- have to match exactly, because every one
    of them is a key that a label, a contact test or a superposition is indexed by.

    Coordinates are compared to a tolerance rather than exactly. The .pdb format
    stores three decimals and mmCIF stores the deposited precision, so the two
    disagree in the last digit on a few atoms per entry by as much as half a
    thousandth of an Angstrom. Requiring bit-equality there would fail a
    conversion that is in fact correct, and a tolerance of
    0.002 A is four orders of magnitude below the 4.5 A contact criterion it feeds.
    """
    def keyed(rows: list[tuple]) -> dict[tuple, tuple[float, float, float]]:
        return {(r[0], r[2], r[3], r[4], r[5], r[6], r[7].upper()): r[8]
                for r in rows}

    a, b = keyed(atoms(cif)), keyed(_pdb_atoms(pdb))
    shared = set(a) & set(b)
    worst = max((max(abs(a[k][i] - b[k][i]) for i in range(3))
                 for k in shared), default=0.0)
    return {"entry": pdb.stem, "n_cif": len(a), "n_pdb": len(b),
            "only_in_cif": len(set(a) - set(b)),
            "only_in_pdb": len(set(b) - set(a)),
            "worst_coordinate_difference": round(worst, 6),
            "identical": (set(a) == set(b)
                          and worst <= COORDINATE_TOLERANCE)}


def to_pdb_lines(rows: list[tuple]) -> tuple[list[str], list[str]]:
    """Render atoms as PDB ATOM/HETATM lines, plus the reasons any were refused.

    This exists so that an mmCIF entry and a .pdb entry reach the rest of the
    repository through one parser and one receptor writer. The alternative --
    teaching the writer a second input shape -- would leave two paths that could
    disagree about an insertion code or an alternate, and the receptor is the file
    every method reads.

    The legacy format cannot express everything mmCIF can. A chain identifier
    wider than one column or a residue number past four digits has no
    representation here, so the atom is refused with a reason rather than silently
    truncated into a different residue.
    """
    lines, refused = [], []
    seen: set[str] = set()
    for i, (rec, alt, resname, chain, seq, icode, name, el, xyz) in enumerate(
            rows, start=1):
        if len(chain) != 1:
            if f"chain {chain}" not in seen:
                seen.add(f"chain {chain}")
                refused.append(f"chain identifier {chain!r} needs more than the "
                               f"one column the PDB format has")
            continue
        if not -999 <= seq <= 9999:
            if "resseq" not in seen:
                seen.add("resseq")
                refused.append(f"residue number {seq} does not fit four columns")
            continue
        if len(resname) > 3 or len(name) > 4:
            if "wide" not in seen:
                seen.add("wide")
                refused.append(f"residue name {resname!r} or atom name {name!r} "
                               f"is wider than the format allows")
            continue
        # A one-letter element leaves column 13 empty; a two-letter one fills it.
        # Getting this wrong moves every atom name one column and the element is
        # then read from the wrong place.
        nm = name if len(name) == 4 else (
            f" {name:<3s}" if len(el.strip()) <= 1 else f"{name:<4s}")
        lines.append(
            f"{'ATOM  ' if rec == 'ATOM' else 'HETATM'}{i % 100000:5d} {nm}"
            f"{(alt or ' '):1s}{resname:>3s} {chain}{seq:4d}{(icode or ' '):1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          "
            f"{el.strip()[:2]:>2s}")
    return lines, refused


def as_pdb_text(path: Path) -> tuple[str, list[str]]:
    """One entry as PDB text, whichever format it is stored in."""
    if path.suffix != ".cif":
        return path.read_text(errors="ignore"), []
    lines, refused = to_pdb_lines(atoms(path))
    return "\n".join(lines) + "\nEND\n", refused


def round_trip(cif: Path) -> dict:
    """Does rendering mmCIF as PDB text and reparsing it preserve every atom?

    The receptor for an mmCIF-only entry is written from this rendering, so if the
    round trip loses or moves an atom the external inputs are quietly wrong. The
    comparison is against the mmCIF reading itself, not against another file.
    """
    src = atoms(cif)
    text, refused = as_pdb_text(cif)
    back = _parse_text(text)

    def keyed(rows: list[tuple]) -> dict[tuple, tuple]:
        return {(r[0], r[2], r[3], r[4], r[5], r[6]): r[8] for r in rows
                if len(r[3]) == 1 and -999 <= r[4] <= 9999
                and len(r[2]) <= 3 and len(r[6]) <= 4}

    a, b = keyed(src), keyed(back)
    shared = set(a) & set(b)
    worst = max((max(abs(a[k][i] - b[k][i]) for i in range(3))
                 for k in shared), default=0.0)
    return {"entry": cif.stem, "n_representable": len(a), "n_reparsed": len(b),
            "lost": len(set(a) - set(b)), "gained": len(set(b) - set(a)),
            "worst_coordinate_difference": round(worst, 6),
            "refused": refused,
            "identical": set(a) == set(b) and worst <= COORDINATE_TOLERANCE}


def _parse_text(text: str) -> list[tuple]:
    out = []
    for ln in text.splitlines():
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


def _pdb_atoms(path: Path) -> list[tuple]:
    """The .pdb reader, duplicated here so the comparison imports nothing circular."""
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
