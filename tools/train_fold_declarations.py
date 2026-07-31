#!/usr/bin/env python3
"""Every training-fold artifact must say so, and the code must back the claim.

The rule and where it came from
-------------------------------
``AGENTS.md``: *any artifact produced on training folds must say
``"reads_test_fold": false`` and mean it.* The rule arrived partway through this
repository's life and the artifacts that predate it were never brought up to it.
An audit of ``results/architecture_sweep/`` -- which is the training-fold
directory by definition -- found **30 artifacts carrying no such declaration**,
including the whole ``COUNTERATTACK_*`` series, and three that also carry no
``clinical_grade`` although ``AGENTS.md`` says that flag is on every artifact and
is not decoration.

That is a silent gap rather than a wrong number. A reader auditing this
repository has to decide, per artifact, whether it touched the held-out fold, and
the answer is currently "look at the tool that wrote it and work it out". This
file makes the answer a field, and makes adding the field require evidence.

Why the declaration is derived and not asserted
-----------------------------------------------
Stamping thirty files with ``"reads_test_fold": false`` would be writing a claim
nobody checked, which is the failure this repository keeps a list of. So the
claim is derived from the generator instead: the tool that assigns the artifact's
path is located, and its source is searched for any reference to a test-fold or
external input -- the official manifest, the official prediction directories, the
test-side caches, the external set. A generator that mentions none of them cannot
have read the fold, whatever it does internally, because it never names a path
into it.

That is a conservative test in the right direction. It can refuse a
training-fold artifact whose generator merely mentions an official path in a
comment, and it will do that rather than assert something it cannot support; such
an artifact is reported as needing a human decision. It cannot pass an artifact
whose generator opens the fold, because opening it requires naming it.

Three modes
-----------
``--check``   the gate: every artifact under the training-fold directory carries
              both declarations. This is what ``make verify`` runs.
``--audit``   report what is missing and, for each, whether the generator
              supports the claim.
``--fix``     insert the declarations the generators support, leaving the rest
              untouched and named.

Nothing here reads a label, a fold, or an external unit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from pocket_bench.paths import ROOT

SWEEP = ROOT / "results/architecture_sweep"
TOOLS = ROOT / "tools"

# A generator naming any of these has a path into the held-out fold or the
# external set, so this file will not derive a "no" from it. The list is written
# as substrings of paths rather than of prose so that a tool discussing the fold
# in its docstring is not confused with one that opens it -- with the exception
# of the bare directory names, which are checked only inside quoted paths.
TEST_FOLD_MARKERS = (
    "results/official_fold",
    "results/cryptobench_official",
    "data/cryptobench_apo/official_manifest.json",
    "official_receptors",
    "official_labels",
    "_wide_cache_test",
    "_expanded_cache_test",
    "_cascade_cache_test",
    "data/external",
    "results/external",
)

REQUIRED = ("clinical_grade", "reads_test_fold")

# The same statement is spelled six ways across the tree, and a gate that knew
# only the first spelling would report five honest artifacts as silent. The
# audit that found this counted eighteen distinct field names carrying some form
# of "did this touch a held-out set"; these are the ones that answer *this*
# question, and the canonical spelling is the first. New artifacts should use it;
# the rest are accepted so that a rename is not forced on files whose bytes other
# things may depend on.
READ_SYNONYMS = (
    "reads_test_fold",
    "test_fold_touched",
    "test_fold_read",
    "test_fold_reads",
    "reads_our_test_fold",
    "reads_cryptobench_test_fold",
)

# Artifacts whose generator names a fold path for a reason that is not a read,
# recorded by name with the reason rather than being silently skipped. Empty
# until the audit produces one; an entry here is a human decision and should
# carry the reason it was made.
EXEMPT: dict[str, str] = {}


def _artifacts() -> list[Path]:
    out = []
    for p in sorted(glob.glob(str(SWEEP / "*.json"))):
        if os.path.basename(p).startswith("_"):
            continue
        out.append(Path(p))
    return out


def generators_of(artifact: Path) -> list[Path]:
    """The tools that write this artifact, as opposed to reading it.

    Naming the path is not enough. ``emit_frozen_numbers.py`` assigns a constant
    for nearly every artifact in the repository because it reads them all, and
    counting it as a generator made seventeen artifacts look as though their
    producer touched the held-out fold. A writer is a tool that binds the path to
    a name *and* writes through that name, so both are required here.
    """
    name = artifact.name
    out = []
    for t in sorted(TOOLS.glob("*.py")):
        src = t.read_text(errors="ignore")
        if name not in src:
            continue
        bound = set()
        for line in src.splitlines():
            if name not in line or line.strip().startswith("#"):
                continue
            head, _, _tail = line.partition("=")
            if _ and head.strip().isidentifier():
                bound.add(head.strip())
        if any(f"{b}.write_text" in src or f"{b}.open(" in src
               or f"savez_compressed({b}" in src for b in bound):
            out.append(t)
    return out


def touches_the_fold(tool: Path) -> list[str]:
    src = tool.read_text(errors="ignore")
    return [m for m in TEST_FOLD_MARKERS if m in src]


def audit() -> list[dict]:
    rows = []
    for a in _artifacts():
        d = json.loads(a.read_text())
        missing = []
        if "clinical_grade" not in d:
            missing.append("clinical_grade")
        if not any(s in d for s in READ_SYNONYMS):
            missing.append("reads_test_fold")
        if not missing:
            continue
        gens = generators_of(a)
        marks = sorted({m for g in gens for m in touches_the_fold(g)})
        rows.append({
            "artifact": str(a.relative_to(ROOT)),
            "missing": missing,
            "generators": [str(g.relative_to(ROOT)) for g in gens],
            "generator_names_a_fold_path": marks,
            "claim_is_supported": bool(gens) and not marks,
            "declared_elsewhere_in_the_file": {
                k: d[k] for k in ("reads_any_external_unit", "reads_a_label",
                                  "fold", "protocol")
                if k in d and isinstance(d[k], (str, bool))},
        })
    return rows


LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"


def in_the_ledger(artifact: Path) -> bool:
    """Whether the ledger lists this artifact *as an access*.

    The ledger is the repository's record of every look at the held-out fold. If
    it lists an artifact, the honest declaration is ``true`` and writing
    ``false`` would be the worst outcome this file could produce -- a claim of
    innocence on a file the ledger has already recorded as an access.

    Membership is read from the probe list rather than by searching the file's
    text, because a training-fold artifact can appear inside a probe's
    ``selection_provenance``: it is the sweep that *chose* the architecture the
    probe then read on the fold, and it read nothing itself.
    ``COUNTERATTACK_QUOTIENT.json`` is exactly that, correctly declares
    ``false``, and a substring test calls it a liar.
    """
    if not LEDGER.is_file():
        return False
    led = json.loads(LEDGER.read_text())
    rel = str(artifact.relative_to(ROOT))
    return any(r.get("artifact") == rel
               for r in led.get("standalone_probe_artifacts", []))


def _indent_of(path: Path) -> int:
    for line in path.read_text().splitlines()[1:]:
        if line.startswith(" "):
            return len(line) - len(line.lstrip(" "))
    return 1


def fix(rows: list[dict]) -> tuple[list[str], list[str]]:
    done, left = [], []
    for r in rows:
        p = ROOT / r["artifact"]
        listed = in_the_ledger(p)
        if not r["claim_is_supported"] and not listed:
            left.append(r["artifact"])
            continue
        value = bool(listed)
        d = json.loads(p.read_text())
        out: dict = {}
        for k, v in d.items():
            out[k] = v
            if k == "schema":
                if "clinical_grade" in r["missing"]:
                    out["clinical_grade"] = False
                if "reads_test_fold" in r["missing"]:
                    out["reads_test_fold"] = value
        if "clinical_grade" not in out:
            out = {"clinical_grade": False, **out}
        if "reads_test_fold" not in out and "reads_test_fold" in r["missing"]:
            out = {"reads_test_fold": value, **out}
        p.write_text(json.dumps(out, indent=_indent_of(p),
                                allow_nan=False) + "\n")
        done.append(f"{r['artifact']}  reads_test_fold={str(value).lower()}")
    return done, left


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args(argv)

    rows = [r for r in audit() if os.path.basename(r["artifact"]) not in EXEMPT]

    if a.fix:
        done, left = fix(rows)
        print(f"declared {len(done)} artifacts as training-fold")
        for n in done:
            print(f"  + {n}")
        if left:
            print(f"\n{len(left)} left undeclared because the generator names a "
                  f"fold path, or no generator was found; each needs a human "
                  f"decision and an entry in EXEMPT with the reason:")
            for r in rows:
                if r["artifact"] in left:
                    print(f"  ? {r['artifact']}  generators="
                          f"{r['generators'] or 'none found'}  "
                          f"names={r['generator_names_a_fold_path']}")
        return 0

    if not rows:
        print(f"OK every artifact under {SWEEP.relative_to(ROOT)} declares "
              f"{' and '.join(REQUIRED)}")
        return 0

    print(f"{len(rows)} artifacts under {SWEEP.relative_to(ROOT)} are missing a "
          f"declaration")
    for r in rows:
        flag = "supported" if r["claim_is_supported"] else "NEEDS A DECISION"
        print(f"  {os.path.basename(r['artifact']):44s} missing "
              f"{', '.join(r['missing']):32s} {flag}")
        if not r["claim_is_supported"]:
            print(f"      generators {r['generators'] or 'none found'}; "
                  f"names {r['generator_names_a_fold_path']}")
    if a.check:
        print("\nFAILED: run tools/train_fold_declarations.py --fix, then decide "
              "the rest by hand")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
