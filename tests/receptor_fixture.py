"""Training receptors for the tests that need real structures, or an honest skip.

Why this exists
---------------
``data/cryptobench_apo/train_receptors/`` is gitignored on purpose: 185 MB of
coordinates that are reproducible byte-for-byte from the SHA-256 pins already
committed in ``TRAIN_MANIFEST.json``. That trade is defensible, and it had one
consequence nobody had checked. In a fresh clone the tests that exercise
``void_topology``, ``sidechain_geometry`` and ``displacement`` on deposited
structures did not skip --- they **errored**, on ``FileNotFoundError``, because
they opened ``e["receptor_path"]`` with no guard. A reviewer cloning the
repository and running the suite got red tests and no statement of why.

So the rule from ``AGENTS.md`` applies here as much as to an artifact: report
"not measured" as not measured, print the reason, and print what to run. A skip
that names the command is a different object from a stack trace.

What it adds beyond a guard
----------------------------
Nothing verified these files. The manifest pins a ``receptor_sha256`` for every
one of the 770 and no tool or test read it back, so a truncated download, a
half-finished fetch or a hand-edited file would have been used as though it were
the deposited structure. Every receptor handed out here is checked against its
pin, and a mismatch fails rather than skips: absent is a missing input, wrong is
a broken one, and the two must not report the same way.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
RECEPTORS = ROOT / "data/cryptobench_apo/train_receptors"

# Copied from the tool's own usage line rather than remembered. The first
# version of this constant invented a --fetch flag that does not exist, which
# would have made the skip message worse than no message: a reviewer would have
# run it, seen argparse reject the flag, and concluded the instruction was stale.
FETCH = ("PYTHONPATH=src python3.12 tools/build_training_fold.py "
         "--folds train-0 train-1 train-2 train-3")

_SKIP = (
    "the training receptors are not materialised, so this test has no real "
    "structure to run on. They are 185 MB of coordinates, gitignored because "
    "TRAIN_MANIFEST.json pins a SHA-256 for every one of them and they are "
    "reproducible byte-for-byte from those pins. Materialise them with:\n"
    f"    {FETCH}\n"
    "This is a skip and not a pass: nothing about void topology, side-chain "
    "geometry or displacement has been checked against a deposited structure "
    "in this run."
)

_digest_cache: dict[str, str] = {}


def entries(n: int | None = None) -> list[dict]:
    """The first ``n`` manifest entries whose receptor is present and correct.

    Raises ``SkipTest`` when the receptors are absent, and ``AssertionError``
    when one is present with the wrong digest. The asymmetry is deliberate.
    """
    if not MANIFEST.is_file():
        raise unittest.SkipTest(
            "data/cryptobench_apo/TRAIN_MANIFEST.json is missing, which should "
            "not happen in a clone -- it is committed. Check the checkout.")
    all_entries = json.loads(MANIFEST.read_text())["entries"]
    if not RECEPTORS.is_dir() or not any(RECEPTORS.iterdir()):
        raise unittest.SkipTest(_SKIP)

    out: list[dict] = []
    for e in all_entries:
        path = ROOT / e["receptor_path"]
        if not path.is_file():
            continue
        rel = e["receptor_path"]
        if rel not in _digest_cache:
            _digest_cache[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        got = _digest_cache[rel]
        if got != e["receptor_sha256"]:
            raise AssertionError(
                f"{rel} is present and does not match the digest the manifest "
                f"pins for it: {got[:16]} against {e['receptor_sha256'][:16]}. "
                f"A present-but-wrong receptor is a broken input, not a missing "
                f"one, so this fails rather than skipping. Re-fetch with:\n"
                f"    {FETCH}")
        out.append(e)
        if n is not None and len(out) >= n:
            break
    if not out:
        raise unittest.SkipTest(_SKIP)
    return out


def n_materialised() -> int:
    """How many of the manifest's receptors are on disk. For reporting only."""
    if not MANIFEST.is_file() or not RECEPTORS.is_dir():
        return 0
    return sum(1 for e in json.loads(MANIFEST.read_text())["entries"]
               if (ROOT / e["receptor_path"]).is_file())
