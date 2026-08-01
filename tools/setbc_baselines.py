#!/usr/bin/env python3
"""Run pLM-NN and PocketMiner on the frozen cryo-EM sets, by redirecting paths.

Why the runners are redirected and not edited
---------------------------------------------
``plmnn_sequences.py``, ``plmnn_embed.py`` and ``pocketminer_run.py`` each carry
an ``--external`` mode whose paths point at Set A's receptors, and Set A's read is
spent and pinned. Adding a third mode to any of them would edit a tool inside a
frozen artifact's blast radius for a reason unrelated to that artifact, which
``AGENTS.md`` records refusing once already: a retry count was raised in a probe
by monkey-patching rather than in ``external_inventory.py``, because minimal blast
radius near a frozen artifact is worth more than tidiness.

So the module globals are reassigned from here and nothing in the runners moves.
What they compute --- the sequence a chain is turned into, the residues mapped to
X, the universe cross-check, the encoder layer, the head, the tie rule --- is
identical to what produced the numbers on the official fold and on Set A, which
is the property that makes a comparison across sets meaningful at all.

The one input that has to be built
----------------------------------
``plmnn_sequences`` cross-checks its sequences against a ``PER_STRUCTURE.json``
holding one ``n_universe`` per unit. Set A has one; the frozen sets do not,
because they were never scorable. It is derived here from the receptors the
manifest pins, which is the same quantity Set A's was derived from.

Nothing here compares a score against a label.
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
from pocket_bench.pdb_io import parse_pdb_atoms                    # noqa: E402

MANIFEST = ROOT / "data/external/setbc_manifest.json"
PER_STRUCTURE = ROOT / "results/external/SETBC_PER_STRUCTURE.json"
SEQS = ROOT / "results/baselines/PLMNN_SETBC_SEQUENCES.json"
PLM_OUT = ROOT / "results/baselines/PLMNN_SETBC_SCORES.json"
PLM_CKPT = ROOT / "results/baselines/_plmnn_setbc_checkpoint.jsonl"
PM_RECEPTORS = ROOT / "data/external/setbc_receptors"
PM_SCORE_DIR = ROOT / "data/baselines/pocketminer_setbc"
PM_OUT = ROOT / "results/baselines/POCKETMINER_SETBC_SCORES.json"
PREDS = ROOT / "results/external/setbc_predictions"

SKIP_RES = frozenset({"HOH", "WAT", "DOD"})


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _entries() -> list[dict]:
    return json.loads(MANIFEST.read_text())["entries"]


def build_per_structure() -> int:
    """One n_universe per unit, from the receptors the manifest pins."""
    rows = []
    for e in _entries():
        atoms = parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
        n = len({a["resseq"] for a in atoms
                 if a["chain"] == e["chain"] and a["element"] != "H"
                 and a["resname"] not in SKIP_RES})
        rows.append({"pdb": e["pdb"], "chain": e["chain"], "n_universe": n})
    PER_STRUCTURE.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {PER_STRUCTURE.relative_to(ROOT)}: {len(rows)} units, "
          f"{sum(r['n_universe'] for r in rows)} residues")
    return 0


def run_plmnn(limit: int = 0) -> int:
    import plmnn_sequences
    import plmnn_embed

    if not PER_STRUCTURE.is_file():
        build_per_structure()

    # Redirect, do not edit. Both modules read these at call time.
    plmnn_sequences.MANIFEST = MANIFEST
    plmnn_sequences.PER_STRUCTURE = PER_STRUCTURE
    plmnn_sequences.OUT = SEQS
    print("building sequences", flush=True)
    doc = plmnn_sequences.build()
    SEQS.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"  wrote {SEQS.relative_to(ROOT)}: {doc.get('n_units', '?')} units",
          flush=True)

    plmnn_embed.SEQS = SEQS
    plmnn_embed.PER_STRUCTURE = PER_STRUCTURE
    plmnn_embed.OUT = PLM_OUT
    plmnn_embed.CKPT = PLM_CKPT
    print("running the encoder and the head", flush=True)
    # run() returns the artifact dict; main() is what writes it. Called directly
    # so the redirected OUT is used and no argv is parsed.
    doc = plmnn_embed.run(limit)
    if limit:
        print("partial run, artifact not written")
        return 0
    PLM_OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"  wrote {PLM_OUT.relative_to(ROOT)}", flush=True)
    return 0


def run_pocketminer() -> int:
    import pocketminer_run
    pocketminer_run.EXTERNAL_RECEPTORS = PM_RECEPTORS
    pocketminer_run.EXTERNAL_SCORE_DIR = PM_SCORE_DIR
    pocketminer_run.EXTERNAL_OUT = PM_OUT
    # predict(external=True) is what main() dispatches to; it reads the three
    # module globals above at call time, which is what makes redirecting them
    # enough and editing the runner unnecessary.
    return pocketminer_run.predict(external=True)


def archive(method: str, src: Path) -> int:
    """Copy a baseline's scores into the read's archive shape."""
    if not src.is_file():
        raise SystemExit(f"{src.relative_to(ROOT)} does not exist yet")
    raw = json.loads(src.read_text())
    units = raw.get("units") or raw.get("scores") or {}
    PREDS.mkdir(parents=True, exist_ok=True)
    out = PREDS / f"{method}.json"
    out.write_text(json.dumps({
        "schema": f"geoaudit.setbc_prediction.{method}.v1",
        "clinical_grade": False,
        "method": method,
        "n_units": len(units),
        "manifest_sha256": _sha(MANIFEST),
        "source": src.relative_to(ROOT).as_posix(),
        "source_sha256": _sha(src),
        "units": units,
    }, indent=1) + "\n")
    print(f"archived {len(units)} units -> {out.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-structure", action="store_true")
    ap.add_argument("--plmnn", action="store_true")
    ap.add_argument("--pocketminer", action="store_true")
    ap.add_argument("--archive", choices=("plmnn", "pocketminer"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.per_structure:
        return build_per_structure()
    if a.plmnn:
        return run_plmnn(a.limit)
    if a.pocketminer:
        return run_pocketminer()
    if a.archive:
        return archive(a.archive,
                       PLM_OUT if a.archive == "plmnn" else PM_OUT)
    ap.error("choose --per-structure, --plmnn, --pocketminer or --archive")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
