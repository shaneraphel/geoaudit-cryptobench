"""Hold the two test runners to each other, and hold the README to both.

This repository runs its tests two ways. ``make test`` discovers them with
``unittest``, which is what CI executes; contributors and reviewers usually run
``pytest``. The README once recorded that the two had diverged, that five checks
guarding the neighbour-search kernel had therefore never run in CI, and that the
two "now collect the same set".

They had diverged again. ``unittest`` collected 529 and ``pytest`` 614, and the
85 in the gap were not incidental: every test guarding the external validation
set, its preregistration, its confirmatory read, the recovered labelling rule,
the PocketMiner read and the threshold curve. ``unittest`` collects methods of
``TestCase`` subclasses and nothing else, so a module written as plain
``test_*`` functions is discovered by one runner and silently skipped by the
other. The claim that the sets agreed was true when written and rotted quietly,
which is what claims do when nothing checks them.

So the agreement is checked here rather than asserted in prose: pytest's
collection must cover unittest's, unittest must import every module without
error, and the README must state the counts both runners actually produce.

Usage: PYTHONPATH=src python3.12 tools/count_tests.py [--check]
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
README = ROOT / "README.md"


def unittest_ids() -> set[str]:
    """Every test unittest's discovery finds, as ``module.Class.method``."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS), top_level_dir=str(TESTS))
    if loader.errors:
        first = str(loader.errors[0]).strip().splitlines()[-1]
        raise SystemExit(
            f"unittest could not import {len(loader.errors)} test module(s), so "
            f"its count would be an undercount rather than a disagreement: "
            f"{first}")
    out: set[str] = set()

    def walk(s: unittest.TestSuite) -> None:
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            else:
                out.add(t.id())

    walk(suite)
    return out


def pytest_ids() -> set[str]:
    """Every test pytest collects, normalised to unittest's naming."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--collect-only"],
        capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"pytest could not collect:\n{proc.stdout[-2000:]}")
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        path, *rest = line.split("::")
        out.add(".".join([Path(path).stem, *rest]))
    return out


def counts() -> dict[str, object]:
    u, p = unittest_ids(), pytest_ids()
    missed = sorted(p - u)
    by_module: dict[str, int] = {}
    for t in missed:
        by_module[t.split(".")[0]] = by_module.get(t.split(".")[0], 0) + 1
    return {
        "unittest": len(u),
        "pytest": len(p),
        "collected_by_pytest_only": len(missed),
        "modules_in_the_gap": dict(sorted(by_module.items(),
                                          key=lambda kv: -kv[1])),
        "collected_by_unittest_only": sorted(u - p),
    }


def check() -> int:
    c = counts()
    if c["collected_by_unittest_only"]:
        print(f"FAILED: pytest does not collect "
              f"{len(c['collected_by_unittest_only'])} tests that unittest "
              f"does, e.g. {c['collected_by_unittest_only'][:3]}")
        return 1
    text = README.read_text()
    want = re.search(r"`make test` runs (\d+) tests", text)
    if not want:
        print("FAILED: the README no longer states what `make test` runs, in "
              "the form '`make test` runs <n> tests'")
        return 1
    if int(want.group(1)) != c["pytest"]:
        print(f"FAILED: the README says `make test` runs {want.group(1)} "
              f"tests; pytest collects {c['pytest']}")
        return 1
    gap = re.search(r"unittest\W*'?s discovery finds (\d+) of them", text)
    if not gap or int(gap.group(1)) != c["unittest"]:
        print(f"FAILED: the README puts unittest's share at "
              f"{gap.group(1) if gap else 'nothing'}; it discovers "
              f"{c['unittest']}")
        return 1
    print(f"OK `make test` runs {c['pytest']} tests, covering all "
          f"{c['unittest']} unittest discovers; {c['collected_by_pytest_only']} "
          f"are function-style and reachable only through pytest")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    c = counts()
    print(f"unittest {c['unittest']}, pytest {c['pytest']}, "
          f"pytest-only {c['collected_by_pytest_only']}")
    for m, n in c["modules_in_the_gap"].items():
        print(f"  {n:3d}  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
