"""Zero-leakage firewall for pocket predictors.

The predictor is physically blind to the label. Ligand coordinates can only
enter through the receptor PDB it is handed, so we make that structurally
impossible: every ``predict`` entrypoint is wrapped by :func:`ligand_leak_guard`,
which reads the receptor text and refuses (hard ``CRASH``) if it contains any
non-solvent HETATM or any non-polymer residue smuggled in as an ATOM record.

This is enforcement by boundary, not by discipline. A companion CI test
(``tests/test_leakage_firewall.py``) additionally asserts, via the import graph,
that no module under ``pocket_bench.methods`` imports the scorer
(``pocket_bench.metrics`` / ``pocket_bench.scoring``), so a predictor can never
reach label-joining code.
"""
from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from pocket_bench.paths import STATUS_CRASH

# Solvent / ion HETATM codes that are benign in a receptor file.
_SOLVENT_IONS = frozenset(
    {"HOH", "WAT", "DOD", "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE",
     "CU", "NI", "CO", "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "ACE", "IOD",
     "BR", "CD", "HG", "CS", "RB", "SR", "BA"}
)

# Standard polymer residues legal in an ATOM record (20 aa + common variants +
# nucleotides). Anything else appearing as ATOM is treated as a smuggled ligand.
_STANDARD_POLYMER = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # protonation / modification variants that crystallographers use
        "MSE", "SEC", "PYL", "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH",
        "LYN", "ARN", "TYM", "HSD", "HSE", "HSP", "SEP", "TPO", "PTR",
        # nucleotides
        "DA", "DC", "DG", "DT", "DU", "DI", "A", "C", "G", "U", "I", "N",
        "UNK",
    }
)


def _iter_residue_records(text: str):
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            yield rec, line[17:20].strip()


def has_hetatm(text: str) -> bool:
    """True if a non-solvent HETATM (i.e. a real ligand/cofactor) is present."""
    return any(
        rec == "HETATM" and resname not in _SOLVENT_IONS
        for rec, resname in _iter_residue_records(text)
    )


def has_ligand_resnames(text: str) -> bool:
    """True if a non-polymer residue is smuggled into ATOM records."""
    return any(
        rec == "ATOM" and resname not in _STANDARD_POLYMER
        for rec, resname in _iter_residue_records(text)
    )


def leak_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    het = sorted(
        {
            resname
            for rec, resname in _iter_residue_records(text)
            if rec == "HETATM" and resname not in _SOLVENT_IONS
        }
    )
    smuggled = sorted(
        {
            resname
            for rec, resname in _iter_residue_records(text)
            if rec == "ATOM" and resname not in _STANDARD_POLYMER
        }
    )
    if het:
        reasons.append(f"non_solvent_hetatm={het}")
    if smuggled:
        reasons.append(f"non_polymer_atom_resnames={smuggled}")
    return reasons


def receptor_provenance(path: Path, text: str) -> dict[str, Any]:
    """Firewall provenance stamped onto every prediction (proves atom-only input)."""
    return {
        "input_receptor_sha256": hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest(),
        "input_receptor_path": Path(path).name,
        "input_has_hetatm": has_hetatm(text),
        "input_has_ligand_resnames": has_ligand_resnames(text),
        "input_atom_only_verified": not (has_hetatm(text) or has_ligand_resnames(text)),
    }


def ligand_leak_guard(method: str | Callable[..., str]) -> Callable:
    """Decorator: refuse to run a predictor on a receptor that carries ligand atoms.

    Wraps ``predict(receptor_pdb, *, pdb_id, ...)``. On any leak the predictor is
    never executed; a fail-closed ``CRASH`` record is returned with the reason.
    On success the returned prediction is stamped with firewall provenance.

    ``method`` may be a callable over the call kwargs for a predictor that serves
    several registered method names, so a fail-closed record is attributed to the
    variant that was actually invoked rather than to a fixed label.
    """
    from pocket_bench.methods import prediction  # local import: avoids cycle

    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(receptor_pdb, *args, **kwargs):
            pdb_id = kwargs.get("pdb_id", "")
            name = method(**kwargs) if callable(method) else method
            path = Path(receptor_pdb)
            try:
                text = path.read_text(errors="ignore")
            except Exception as exc:  # noqa: BLE001
                return prediction(
                    method=name, pdb_id=pdb_id, status=STATUS_CRASH,
                    error=f"ligand_leak_guard:receptor_unreadable:{exc}",
                )
            reasons = leak_reasons(text)
            if reasons:
                return prediction(
                    method=name, pdb_id=pdb_id, status=STATUS_CRASH,
                    error="ligand_leak_guard:" + ";".join(reasons),
                    extra=receptor_provenance(path, text),
                )
            out = func(receptor_pdb, *args, **kwargs)
            if isinstance(out, dict):
                for k, v in receptor_provenance(path, text).items():
                    out.setdefault(k, v)
            return out

        wrapper.__ligand_firewalled__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorate
