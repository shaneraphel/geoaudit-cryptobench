"""Counting field: geometry_field columns plus nonlocal-seam family.

Separate module so GEOMETRY_FIELD.json's code digest stays untouched.
Development detector aimed at the short-chain / buried-cryptic deficit vs
pLM-NN. ``clinical_grade`` is false.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES,
    algebraic_residue_features,
)
from pocket_bench.methods.geometry_wires import (
    aggregate as geo_aggregate,
    geometry_columns,
    residue_rows as geo_residue_rows,
)
from pocket_bench.methods.nonlocal_seam import COLUMNS as SEAM_QTY, compute as seam_compute
from pocket_bench.methods.table_field import TableField
from pocket_bench.methods.wide_descriptors import build_wide
from pocket_bench.pdb_io import parse_pdb_atoms

_FIELD_PATH = (Path(__file__).resolve().parents[3]
               / "data/cryptobench_apo/SEAM_GEOMETRY_FIELD.json")
_CACHED: dict[str, "SeamGeometryField"] = {}
SKIP = frozenset({"HOH", "WAT", "DOD"})
AGGREGATIONS = ("own", "contact", "walk2")


def seam_column_names() -> tuple[str, ...]:
    return tuple(f"seam~{agg}~{q}" for agg in AGGREGATIONS for q in SEAM_QTY)


def seam_columns(receptor_pdb: Path, chain: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue seam block aligned to geometry_wires residue universe."""
    atoms = parse_pdb_atoms(Path(receptor_pdb).read_text())
    order, ctr, take = geo_residue_rows(atoms, chain)
    poly: dict[tuple[int, str], list[dict]] = {}
    for a in atoms:
        if a["chain"] != chain or a["element"] == "H" or a["resname"] in SKIP:
            continue
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
    atoms_by_res = [poly[k] for k in order]
    resseqs_full = [k[0] for k in order]
    raw = seam_compute(atoms_by_res, resseqs_full)[take]
    X = geo_aggregate(ctr, raw)
    names = seam_column_names()
    if X.shape[1] != len(names):
        raise AssertionError(f"seam width {X.shape[1]} != {len(names)}")
    resseq = np.asarray([r for r, _ic in
                         sorted({(r, "") for r, _ic in order})], dtype=np.int64)
    return resseq, X


class SeamGeometryField(TableField):
    def score_receptor(self, receptor_pdb: str | Path,
                       chain: str | None = None):
        path = Path(receptor_pdb)
        if chain is None:
            raise ValueError("seam_geometry_field needs an explicit chain")
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        Xw, _ = build_wide(F, codes, ctr, n_res_per,
                           tuple(FEATURE_NAMES), self.prop)
        g_resseq, Xg = geometry_columns(path, chain)
        s_resseq, Xs = seam_columns(path, chain)
        if (not np.array_equal(np.asarray(resseq), g_resseq)
                or not np.array_equal(np.asarray(resseq), np.asarray(s_resseq))):
            raise AssertionError(
                f"{path.name} chain {chain}: wire/geometry/seam residue "
                f"universes disagree")
        X = np.concatenate([np.asarray(Xw), Xg, Xs], axis=1)
        s = self.score_matrix(X, ctr, n_res_per)
        return resseq, s, self.positive_call(s)


def load_field(path: str | Path | None = None) -> SeamGeometryField:
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; run tools/compile_seam_geometry_field.py")
        _CACHED[key] = SeamGeometryField(json.loads(p.read_text()))
    return _CACHED[key]


def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            top_k: int = 5, **_ignored: Any) -> dict[str, Any]:
    import time
    from pocket_bench.methods import prediction
    from pocket_bench.paths import STATUS_CRASH, STATUS_OK

    t0 = time.perf_counter()
    try:
        field = load_field()
        resseq, s, call = field.score_receptor(receptor_pdb, chain)
        order = np.argsort(-s, kind="stable")
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(s[i]), "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        return prediction(
            method="seam_geometry_field", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(v)
                                   for r, v in zip(resseq, s)},
                "residue_positive": [int(r) for r, c in zip(resseq, call) if c],
                "n_residues": int(len(resseq)),
                "n_wires": int(field.doc["n_wires"]),
                "n_tables": int(len(field.tables)),
                "operating_q": field.q,
                "protocol": "quaternary_pair_tables_integer_fanout",
                "families": (
                    f"geometry_field 645+624 plus {len(SEAM_QTY)*3} "
                    "nonlocal-seam columns"
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(method="seam_geometry_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])
