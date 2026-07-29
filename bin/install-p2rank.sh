#!/usr/bin/env bash
# Install the exact P2Rank this repository's numbers were produced with.
#
# The version matters and is not a detail. Every P2Rank figure in the paper is a
# transformation of two CSV files written by one release under one JVM at one
# configuration, and all three are recorded per unit in the raw archive
# (results/*/p2rank_raw/*/run.json) alongside a SHA-256 of each CSV. A different
# release would produce different bytes and the archive gate would say so, which
# is the point of pinning rather than a formality.
#
# The distribution itself is deliberately not committed. It is 292 MB of
# third-party binaries, one file of which exceeds GitHub's size limit, and
# committing it would put that weight into every clone forever while adding
# nothing checkable: what makes the baseline auditable is the archived CSVs, the
# recorded command, the version banner and the digests, all of which are tracked.
#
# Usage: bash bin/install-p2rank.sh && export P2RANK_HOME=$PWD/.tools/p2rank
set -euo pipefail

VERSION="2.5.1"
# Pinned because the paper quotes this banner. If the download's version differs,
# the numbers in the archive were not produced by what was just installed.
EXPECTED_JVM_MAJOR="17"
URL="https://github.com/rdk/p2rank/releases/download/${VERSION}/p2rank_${VERSION}.tar.gz"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${ROOT}/.tools"
DEST="${TOOLS}/p2rank_${VERSION}"
TARBALL="${TOOLS}/p2rank_${VERSION}.tar.gz"

mkdir -p "${TOOLS}"

if [ ! -d "${DEST}" ]; then
  echo "fetching P2Rank ${VERSION}"
  # --retry because this download failed truncated once and tar then reported a
  # corrupt member 400 MB in, which reads like a bad release rather than a bad
  # connection. gzip -t is what tells the two apart.
  curl -fsSL --retry 5 --retry-all-errors -o "${TARBALL}" "${URL}"
  gzip -t "${TARBALL}" || {
    echo "the download is truncated; delete ${TARBALL} and retry" >&2
    exit 1
  }
  tar -xzf "${TARBALL}" -C "${TOOLS}"
  rm -f "${TARBALL}"
fi

ln -sfn "p2rank_${VERSION}" "${TOOLS}/p2rank"

# P2Rank is a JVM program and picks up whatever java is first on PATH, so a
# silently different JVM is the easiest way to get numbers that do not match the
# archive. Homebrew's openjdk is not linked into PATH by default on macOS, which
# is how this repository lost its P2Rank between sessions.
if ! command -v java >/dev/null 2>&1; then
  for candidate in \
      "/opt/homebrew/opt/openjdk@${EXPECTED_JVM_MAJOR}/bin" \
      "/usr/local/opt/openjdk@${EXPECTED_JVM_MAJOR}/bin"; do
    if [ -x "${candidate}/java" ]; then
      echo "note: java is not on PATH; add ${candidate}"
      break
    fi
  done
  echo "no java found. Install a JDK ${EXPECTED_JVM_MAJOR} and put it on PATH." >&2
  exit 1
fi

banner="$(java -version 2>&1 | head -1)"
echo "${banner}"
case "${banner}" in
  *"\"${EXPECTED_JVM_MAJOR}."*) ;;
  *) echo "warning: the archived runs used JVM ${EXPECTED_JVM_MAJOR}; this is" \
          "a different major version, so rescoring may not reproduce the" \
          "archived digests" >&2 ;;
esac

"${TOOLS}/p2rank/prank" 2>&1 | head -1
echo
echo "installed. Export it before rescoring:"
echo "  export P2RANK_HOME=${TOOLS}/p2rank"
