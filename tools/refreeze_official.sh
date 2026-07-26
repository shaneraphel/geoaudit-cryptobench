#!/usr/bin/env bash
# Rebuild every artifact that depends on the receptor parser, in dependency order.
#
# The conformer collapse in pdb_io changes the atom set of any entry carrying
# NMR models or alternate locations, so the training feature cache, both
# compiled fields, and every scored metric derived from them are stale together.
# Recompiling only some of them would leave the tables addressed by digits that
# the current extractor no longer produces -- a train/test preprocessing
# mismatch that no test would catch.
#
# P2Rank is deliberately NOT re-scored: it needs a Java runtime, and its rows
# are invariant under this change because it parses the PDB itself and is scored
# through the residue universe alone, which tests/test_conformer_collapse.py
# proves the collapse preserves. --merge carries those rows forward verbatim.
#
# Usage: bash tools/refreeze_official.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src
PY=/usr/local/bin/python3.12

step() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$1"; }

step "1/6 rebuild the 35-invariant feature cache (train + test)"
$PY tools/build_cascade_cache.py --fold both

step "2/6 recompile the algebraic field from the training fold"
$PY tools/compile_algebraic_field.py

step "3/6 recompile the quaternary resolution field, both tracks"
$PY tools/compile_resolution_field.py --jobs 1 --track both --refresh-cache

step "4/6 re-score the official test fold (P2Rank rows carried over)"
$PY tools/run_cryptobench_apo.py --dataset official --jobs 1 --merge \
  --methods geometric_foundation,fstar_pocket,sstar_pocket,ultrametric_shear_oracle,quaternary_lut,quaternary_lut_seq,algebraic_field,random_bbox

step "5/6 refreeze the two-method chance-control report"
$PY tools/run_official_fold.py

step "6/6 paired bootstrap against P2Rank"
$PY tools/paired_vs_p2rank.py

step "done"
