#!/usr/bin/env bash
# Build the optional native geometry kernels.
#
# This is a pure wall-clock optimisation. The kernels are an operation-for-operation
# port of the NumPy reference loops and are covered by a bit-identity test, so a run
# with the library present and a run without it produce the same numbers. If cargo
# is unavailable, skip this: the detectors fall back to NumPy automatically.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$here/native/geoaudit_kernels"

# A sandbox may export CARGO_TARGET_DIR to a scratch path; pin it to the crate so
# the loader in src/pocket_bench/native.py finds the artefact where it expects it.
export CARGO_TARGET_DIR="$crate/target"

cargo build --release --manifest-path "$crate/Cargo.toml"

for ext in dylib so; do
  lib="$CARGO_TARGET_DIR/release/libgeoaudit_kernels.$ext"
  [ -f "$lib" ] && echo "built $lib"
done
