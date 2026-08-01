#!/usr/bin/env python3
"""What a clone actually holds, and what a green suite therefore did not check.

Why this is a gate and not a note
----------------------------------
Several inputs here are gitignored because a committed SHA-256 makes them
reproducible byte-for-byte: 185 MB of training receptors, the official-fold
receptors, the derived caches, the vendored P2Rank release. That trade is
defensible and it has one consequence that was invisible. **A suite can be green
because everything passed, or green because the tests that would have exercised
the deposited structures skipped.** Those look identical in a summary line, and a
reviewer asking "is the verification actually green" is asking exactly which one
it was.

So this prints the state of each pinned-but-absent input, and how many tests
consequently have nothing to run on. It does not fail when an input is absent ---
that is the normal state of a fresh clone and failing on it would make the gate
useless. It fails when an input is **present and does not match its pin**, which
is a corrupt or hand-edited file being used as though it were the deposit.

The distinction is the whole point: absent is a missing input, wrong is a broken
one, and a checker that reports them the same way is worse than one that reports
neither.

Nothing here reads a label, a score or a fold.
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

SCHEMA = "geoaudit.materialised_inputs.v1"
OUT = ROOT / "results/ARTIFACT_INPUTS.json"

# Each entry: the manifest that pins the files, the key holding the relative
# path, the key holding the digest, and the command that materialises them.
PINNED = (
    {
        "name": "training receptors",
        "manifest": "data/cryptobench_apo/TRAIN_MANIFEST.json",
        "path_key": "receptor_path",
        "digest_key": "receptor_sha256",
        "materialise": ("PYTHONPATH=src python3.12 tools/build_training_fold.py"
                        " --folds train-0 train-1 train-2 train-3"),
        "what_depends_on_it": (
            "the real-structure tests for void_topology, sidechain_geometry and "
            "displacement, and every wire-family cache build"),
    },
    {
        "name": "official test-fold receptors",
        "manifest": "data/cryptobench_apo/official_manifest.json",
        "path_key": "receptor_path",
        "digest_key": "receptor_sha256",
        "materialise": ("PYTHONPATH=src python3.12 "
                        "tools/fetch_official_data.py"),
        "what_depends_on_it": (
            "rescoring the held-out fold from coordinates. The frozen per-unit "
            "telemetry is committed, so every published number is checkable "
            "without these"),
    },
)

# How many digests to verify when the files are present. Hashing 770 receptors
# costs a few seconds; hashing every pinned file in the repository on every
# `make verify` would not be paid for by what it catches, and the sample is
# taken from the front of the manifest so it is deterministic rather than lucky.
SAMPLE = 60


def _entries(manifest: Path, path_key: str) -> list[dict]:
    if not manifest.is_file():
        return []
    doc = json.loads(manifest.read_text())
    for key in ("entries", "units", "records"):
        v = doc.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict) \
                and path_key in v[0]:
            return v
    return []


def build(write: bool, sample: int) -> int:
    groups = []
    problems: list[str] = []
    for spec in PINNED:
        manifest = ROOT / spec["manifest"]
        entries = _entries(manifest, spec["path_key"])
        if not entries:
            # "0 of 0 present" reads as absent when the truth is that no entry
            # carrying this path key was found -- a manifest whose shape changed,
            # or a wrong filename here. That is not the same statement and must
            # not print as though it were; the first version of this tool said
            # "absent" for a manifest it had simply failed to parse.
            problems.append(
                f"{spec['name']}: no entry with a {spec['path_key']!r} key was "
                f"found in {spec['manifest']}, so nothing was checked. That is "
                f"a broken checker or a changed manifest, not an absent input")
            groups.append({
                "name": spec["name"], "manifest": spec["manifest"],
                "state": "manifest not understood", "n_pinned": 0,
                "n_present": 0, "n_absent": 0, "n_digests_verified": 0,
                "n_mismatched": 0, "mismatched": [],
                "materialise_with": spec["materialise"],
                "what_depends_on_it": spec["what_depends_on_it"],
            })
            continue
        present = [e for e in entries
                   if (ROOT / e[spec["path_key"]]).is_file()]
        checked, mismatched = 0, []
        for e in present[:sample]:
            rel = e[spec["path_key"]]
            want = e.get(spec["digest_key"])
            if not want:
                continue
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            checked += 1
            if got != want:
                mismatched.append({"path": rel, "pinned": want, "found": got})
        if mismatched:
            problems.append(
                f"{spec['name']}: {len(mismatched)} of {checked} sampled files "
                f"do not match the digest pinned for them")
        groups.append({
            "name": spec["name"],
            "manifest": spec["manifest"],
            "n_pinned": len(entries),
            "n_present": len(present),
            "n_absent": len(entries) - len(present),
            "n_digests_verified": checked,
            "n_mismatched": len(mismatched),
            "mismatched": mismatched[:5],
            "state": ("complete" if entries and len(present) == len(entries)
                      else "partial" if present else "absent"),
            "materialise_with": spec["materialise"],
            "what_depends_on_it": spec["what_depends_on_it"],
        })

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": (
            "which pinned-but-gitignored inputs does this checkout hold, and "
            "which tests therefore had nothing to run on"),
        "why_absent_does_not_fail": (
            "a fresh clone holds none of these by design; they are gitignored "
            "because a committed SHA-256 makes them reproducible byte-for-byte. "
            "Failing on absence would make this gate useless on the case it "
            "exists to describe"),
        "why_present_and_wrong_does_fail": (
            "a file that is present and does not match its pin is being used as "
            "though it were the deposit. Absent is a missing input and wrong is "
            "a broken one, and reporting them the same way is worse than "
            "reporting neither"),
        "digest_sample": sample,
        "why_a_sample": (
            "hashing every pinned file on every make verify is not paid for by "
            "what it catches. The sample is the front of each manifest, so it is "
            "deterministic rather than lucky, and the test fixture verifies "
            "every receptor it actually hands to a test"),
        "groups": groups,
    }

    for g in groups:
        print(f"{g['name']}: {g['state']}  "
              f"{g['n_present']}/{g['n_pinned']} present, "
              f"{g['n_digests_verified']} digests verified, "
              f"{g['n_mismatched']} mismatched")
        if g["state"] != "complete":
            print(f"    materialise with: {g['materialise_with']}")
            print(f"    without them: {g['what_depends_on_it']}")
    if problems:
        print("\nFAILED: " + "; ".join(problems))

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sample", type=int, default=SAMPLE)
    a = ap.parse_args(argv)
    return build(a.write, a.sample)


if __name__ == "__main__":
    raise SystemExit(main())
