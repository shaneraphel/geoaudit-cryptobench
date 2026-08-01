"""The counting field over the deployed wires plus the four geometry families.

What this is
------------
``table_field`` compiles integer tables over 645 wires and scores a receptor end
to end. This is the same construction over 645 + 624 columns: the deployed wires,
then the backbone, side-chain, void-topology and temperature-factor families
whose lift was measured at +0.0121 on twelve cluster-disjoint halvings of the
training fold, 12 of 12 positive, against a control arm that spends the same
table budget on already-deployed wires and lands at −0.0017.

Why it is a separate module and not an option on the old one
------------------------------------------------------------
``TABLE_FIELD.json`` carries a ``code_sha256`` over eight source files and
``table_field.py`` is one of them, so editing that file invalidates the compiled
field behind every published number. ``AGENTS.md`` states the rule and the reason:
a comment fixed in ``table_bank.py`` invalidates the compiled field. So the
scoring path is composed here rather than parameterised there --- the class is
subclassed, one method is overridden, and nothing under the old digest moves.

That also keeps the two detectors nameable. The paper distinguishes three
detectors that share a feature set and were being confused for one another; a
fourth that differed from ``table_field`` only by a flag would be the same
mistake with a different spelling. This one has its own name, its own compiled
artifact and its own column count.

What it does not change
-----------------------
The table topology, the quantisation, the fan-out cap, the ridge and the
operating-point rule are the deployed ones. Only the columns differ. That is what
makes the training-fold lift a statement about the columns, and it is why this
module contains no constants of its own.
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
    COLUMNS as GEOMETRY_COLUMNS,
    geometry_columns,
)
from pocket_bench.methods.table_field import TableField
from pocket_bench.methods.wide_descriptors import build_wide

_FIELD_PATH = (Path(__file__).resolve().parents[3]
               / "data/cryptobench_apo/GEOMETRY_FIELD.json")
_CACHED: dict[str, "GeometryField"] = {}


class GeometryField(TableField):
    """A counting field whose columns are the wires and the four families."""

    def score_receptor(self, receptor_pdb: str | Path,
                       chain: str | None = None):
        """``(resseq, score, positive)`` for one receptor chain, end to end.

        The residue universe is taken from the wire side and the geometry side is
        required to agree with it exactly. It is derived twice --- once by
        ``algebraic_residue_features`` over every non-hydrogen ATOM and once by
        ``geometry_wires.residue_rows`` over the polymer --- and if the two ever
        disagreed, every geometry column would be attached to the wrong residue
        while every lookup succeeded. That is the failure this assertion exists
        for; it is checked per receptor rather than argued once.
        """
        path = Path(receptor_pdb)
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        Xw, _names = build_wide(F, codes, ctr, n_res_per,
                                tuple(FEATURE_NAMES), self.prop)

        if chain is None:
            raise ValueError(
                "this field needs an explicit chain: the geometry families are "
                "computed per polymer chain and a whole-file universe would mix "
                "two contact graphs")
        g_resseq, Xg = geometry_columns(path, chain)
        if not np.array_equal(np.asarray(resseq), g_resseq):
            raise AssertionError(
                f"{path.name} chain {chain}: the wire side reports "
                f"{len(resseq)} residues and the geometry side {len(g_resseq)}, "
                f"or the same count in a different order. Every geometry column "
                f"would be attached to the wrong residue and nothing would "
                f"raise downstream")

        X = np.concatenate([np.asarray(Xw), Xg], axis=1)
        s = self.score_matrix(X, ctr, n_res_per)
        return resseq, s, self.positive_call(s)


def load_field(path: str | Path | None = None) -> GeometryField:
    """Load the compiled field, fail-closed with the command that builds it."""
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"compiled geometry field missing: {p}\n"
                f"  action: PYTHONPATH=src:tools python3.12 "
                f"tools/compile_geometry_field.py")
        # Constructed directly rather than through the inherited factory.
        # ``TableField.load`` is a staticmethod that names its own class, so
        # ``GeometryField.load(p)`` returns a base ``TableField`` -- which then
        # digitises 645 of the 1269 columns and addresses a table past the end
        # of its own digit matrix. It raised here; on a narrower field it would
        # have scored silently with two thirds of the columns missing. The
        # factory is not fixed in place because ``table_field.py`` sits under
        # ``TABLE_FIELD.json``'s ``code_sha256``.
        _CACHED[key] = GeometryField(json.loads(p.read_text()))
    return _CACHED[key]


def column_names() -> tuple[str, ...]:
    """The full column order the field is compiled over: wires, then families."""
    return tuple(FEATURE_NAMES) + GEOMETRY_COLUMNS


def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            top_k: int = 5, **_ignored: Any) -> dict[str, Any]:
    """Runner-facing entry point, emitting a natively residue-level prediction."""
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
            method="geometry_field", pdb_id=pdb_id, status=STATUS_OK,
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
                "families": "backbone 132 + sidechain 261 + void 135 + "
                            "displacement-B 96 over the 645 deployed wires",
            },
        )
    except AssertionError as exc:
        return prediction(method="geometry_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=f"alignment_or_leak_guard:{exc}")
    except Exception as exc:  # noqa: BLE001
        return prediction(method="geometry_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])
