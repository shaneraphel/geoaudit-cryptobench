#!/usr/bin/env python3
"""No published JSON may contain a bare ``NaN`` or ``Infinity`` token.

``json.dump`` emits those tokens by default. They are not JSON: RFC 8259 has no
non-finite literal, so a strict parser -- Rust's serde_json, Go's
encoding/json, PostgreSQL's jsonb, and ``json.loads(..., parse_constant=raise)``
in Python itself -- rejects the file outright. A reviewer who cannot load an
artifact cannot audit it, and "it opens in my editor" is not a standard.

The absent value in these reports is a metric that could not be computed, and
its correct encoding is ``null``. This walks every JSON in the tree and parses
it with a hook that refuses the non-finite constants.

Usage: PYTHONPATH=src python3.12 tools/check_strict_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pocket_bench.paths import ROOT

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".tools"}


def _refuse(token: str) -> float:
    raise ValueError(f"non-finite JSON token {token!r}")


def main() -> int:
    bad: list[tuple[Path, str]] = []
    n = 0
    for path in sorted(ROOT.rglob("*.json")):
        if SKIP_DIRS & set(path.parts):
            continue
        n += 1
        try:
            json.loads(path.read_text(), parse_constant=_refuse)
        except ValueError as exc:
            bad.append((path.relative_to(ROOT), str(exc)[:120]))
        except OSError as exc:
            bad.append((path.relative_to(ROOT), f"unreadable: {exc}"))

    print(f"strict-JSON scan: {n} files")
    if bad:
        print(f"NON-STRICT ({len(bad)}):", file=sys.stderr)
        for p, why in bad:
            print(f"  - {p}: {why}", file=sys.stderr)
        return 1
    print("every JSON parses under a strict reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
