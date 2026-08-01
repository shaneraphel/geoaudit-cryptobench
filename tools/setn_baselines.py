#!/usr/bin/env python3
"""Run the three published baselines on Set N, by redirecting rather than editing.

``setbc_baselines.py`` already solved this problem for the cryo-EM sets: pLM-NN and
PocketMiner read their inputs from module globals, so pointing those globals at a
different manifest runs them on a different set without touching either runner. Set
N needs the same thing at different paths, so this file redirects that file's
globals instead of copying its hundred lines. One redirection layer over another is
worth more than a second copy that can drift from the first.

Nothing here reads a label or compares a score against one.

Usage:
  PYTHONPATH=src:tools python3.12 tools/setn_baselines.py --per-structure
  PYTHONPATH=src:tools python3.12 tools/setn_baselines.py --plmnn
  PYTHONPATH=src:tools python3.12 tools/setn_baselines.py --pocketminer
  PYTHONPATH=src:tools python3.12 tools/setn_baselines.py --archive plmnn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

import setbc_baselines as base                                     # noqa: E402
from pocket_bench.paths import ROOT                                # noqa: E402

SETN = {
    "MANIFEST": ROOT / "data/external/setn_manifest.json",
    "PER_STRUCTURE": ROOT / "results/external/SETN_PER_STRUCTURE.json",
    "SEQS": ROOT / "results/baselines/PLMNN_SETN_SEQUENCES.json",
    "PLM_OUT": ROOT / "results/baselines/PLMNN_SETN_SCORES.json",
    "PLM_CKPT": ROOT / "results/baselines/_plmnn_setn_checkpoint.jsonl",
    "PM_RECEPTORS": ROOT / "data/external/setn_receptors",
    "PM_SCORE_DIR": ROOT / "data/baselines/pocketminer_setn",
    "PM_OUT": ROOT / "results/baselines/POCKETMINER_SETN_SCORES.json",
    "PREDS": ROOT / "results/external/setn_predictions",
}


def redirect() -> None:
    for name, path in SETN.items():
        setattr(base, name, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-structure", action="store_true")
    ap.add_argument("--plmnn", action="store_true")
    ap.add_argument("--pocketminer", action="store_true")
    ap.add_argument("--archive", choices=("plmnn", "pocketminer"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    redirect()
    if a.per_structure:
        return base.build_per_structure()
    if a.plmnn:
        return base.run_plmnn(a.limit)
    if a.pocketminer:
        return base.run_pocketminer()
    if a.archive:
        return base.archive(
            a.archive,
            SETN["PLM_OUT"] if a.archive == "plmnn" else SETN["PM_OUT"])
    ap.error("choose --per-structure, --plmnn, --pocketminer or --archive")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
