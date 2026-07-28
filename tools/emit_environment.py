#!/usr/bin/env python3
"""Record the environment that actually produced the frozen numbers.

This exists because the declared environment and the real one had drifted, in a
way that would have wasted a reviewer's afternoon. ``pyproject.toml`` pinned
``numpy==1.26.4``; the numbers in this repository were produced under numpy
2.5.1, because 1.26.4 segfaults on import in this interpreter and had to be
removed. A reader following the declared pin would either fail to start or, if
they got it working, would be running a different linear-algebra stack than the
one behind the tables.

So the lock is measured, not asserted: interpreter, platform, the versions of
every library the numbers depend on, the BLAS that numpy actually bound to, the
external tools and the JVM behind them. ``environment_sha256`` is a digest over
the whole record, and the frozen artifacts carry it, so a reader can tell at a
glance whether a given table came out of the same stack as another.

The BLAS matters more than it looks. Accelerate and OpenBLAS do not agree to the
last bit on the symmetric solve behind the integer fan-out, and the fan-out is
then rounded to integers, so a coefficient sitting near a rounding boundary can
land differently. The effect is small and bounded, and it is recorded rather
than hidden.

Usage:
  PYTHONPATH=src python3.12 tools/emit_environment.py
  PYTHONPATH=src python3.12 tools/emit_environment.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ENVIRONMENT.json"
REQ = ROOT / "requirements.txt"

# Any other file in the tree that names a version a reader might install from.
# Two of these used to exist -- requirements.lock and configs/environment.yml --
# both pinning the numpy that segfaults here, both unread by the Makefile, by CI
# and by every tool in the repository. Nothing caught them because the audit
# only compared the record against requirements.txt. An unread declaration is
# not harmless: it is the one a reader follows when the generated file looks
# machine-made. So the audit now covers every dependency declaration in the
# tree, and a pin that contradicts the measurement has to be fixed or deleted.
PIN_FILE_PATTERNS = ("requirements", "constraints", "environment", "pyproject",
                     "setup.cfg", "Pipfile")
# By suffix, because a reader installs from a manifest, not from source. Without
# this the scan reads its own docstring, which quotes the bad pin in order to
# explain it, and reports the explanation as the offence.
PIN_FILE_SUFFIXES = (".txt", ".lock", ".yml", ".yaml", ".toml", ".cfg", "")
PIN_RE = re.compile(r"\b(numpy|scipy)\s*(==|>=|=)\s*([0-9][0-9.]*[0-9]|[0-9])")


def _pkg_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("numpy", "scipy"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _blas() -> dict[str, str | None]:
    try:
        import numpy as np
        cfg = np.__config__.show_config("dicts")
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        return {"name": blas.get("name"), "version": blas.get("version")}
    except Exception:  # noqa: BLE001
        return {"name": None, "version": None}


def _p2rank() -> dict[str, str | None]:
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from pocket_bench.methods.p2rank_wrap import (  # noqa: PLC0415
            _find_p2rank,
            _jvm_version,
            _version,
        )
        exe = _find_p2rank()
        # Deliberately not the absolute path: it is a home directory, it says
        # nothing a reader can use, and the repository's own scope gate rejects
        # local absolute paths in published files.
        return {"found": bool(exe), "version": _version() or None,
                "jvm": _jvm_version() if exe else None}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "version": None, "jvm": None,
                "error": str(exc)[:200]}


def _git_commit() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=20)
        return p.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build() -> dict:
    rec = {
        "schema": "geoaudit.environment.v1",
        "clinical_grade": False,
        "note": "measured on the machine that produced the frozen artifacts; "
                "not a declaration of intent",
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            # Not sys.executable: on a venv that is a home directory, which the
            # scope gate rejects and which tells a reader nothing.
            "prefix_is_venv": sys.prefix != sys.base_prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _pkg_versions(),
        "blas": _blas(),
        "tools": {"p2rank": _p2rank()},
        "git_commit": _git_commit(),
    }
    # The digest deliberately excludes the git commit, the interpreter path and
    # whether the P2Rank launcher happened to be on this machine:
    # the same stack on another checkout or another user's home is the same
    # stack, and a digest that changed with every commit would say nothing.
    core = {k: v for k, v in rec.items()
            if k in ("python", "platform", "packages", "blas", "tools")}
    core["python"] = {k: v for k, v in core["python"].items()
                      if k != "prefix_is_venv"}
    core["tools"] = {"p2rank": {k: v for k, v in core["tools"]["p2rank"].items()
                                if k != "found"}}
    rec["environment_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True).encode()).hexdigest()
    return rec


def requirements(rec: dict) -> str:
    lines = [
        "# GENERATED FILE -- regenerate with tools/emit_environment.py",
        "# Pinned to the versions that produced the frozen artifacts, which is",
        "# not the same thing as the oldest versions that would work. numpy",
        "# 1.26.4, which an earlier pyproject demanded, segfaults on import in",
        "# this interpreter and is not usable here.",
    ]
    for name, ver in sorted(rec["packages"].items()):
        if ver:
            lines.append(f"{name}=={ver}")
    return "\n".join(lines) + "\n"


def _tracked_pin_files() -> list[Path]:
    """Every committed file that could tell a reader which version to install.

    Tracked files only: an uncommitted scratch file is not something a reader
    can follow, and walking the tree would drag in the data dump and the vendored
    build trees for nothing.
    """
    try:
        p = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        names = p.stdout.split("\0") if p.returncode == 0 else []
    except Exception:  # noqa: BLE001
        names = []
    out = []
    for name in names:
        if not name or name.startswith("_local/"):
            continue
        stem = Path(name).name.lower()
        if (Path(name).suffix.lower() in PIN_FILE_SUFFIXES
                and any(pat in stem for pat in PIN_FILE_PATTERNS)):
            out.append(ROOT / name)
    return out


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(".") if x.isdigit())


def _rival_pins(measured: dict[str, str | None],
                paths: list[Path] | None = None) -> list[str]:
    """Declarations elsewhere in the tree that contradict the measurement.

    Exact pins must equal the measured version. Lower bounds only have to admit
    it, because ``numpy>=2.0`` is a statement about what the code needs rather
    than a claim about what produced the numbers. Comments are stripped first:
    this file and ``pyproject.toml`` both discuss the bad pin in prose, and a
    gate that cannot tell a warning from a requirement is a gate that gets
    switched off.
    """
    problems: list[str] = []
    for path in (_tracked_pin_files() if paths is None else paths):
        if path == REQ or not path.exists():
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0]
            for name, op, ver in PIN_RE.findall(line):
                have = measured.get(name)
                if not have:
                    continue
                if op in ("==", "=") and ver != have:
                    problems.append(
                        f"{rel}:{lineno} pins {name}{op}{ver} but the numbers "
                        f"were produced under {name} {have}")
                elif op == ">=" and _version(have) < _version(ver):
                    problems.append(
                        f"{rel}:{lineno} requires {name}>={ver} but the "
                        f"measured stack has {name} {have}")
    return problems


def _audit() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    if not REQ.exists():
        print(f"MISSING {REQ.relative_to(ROOT)}")
        return 1
    rec = json.loads(OUT.read_text())
    problems: list[str] = []
    if not rec.get("environment_sha256"):
        problems.append("no environment_sha256")
    for field in ("python", "platform", "packages", "blas", "tools"):
        if not rec.get(field):
            problems.append(f"no {field}")
    if not (rec.get("packages") or {}).get("numpy"):
        problems.append("no numpy version recorded")
    p2 = (rec.get("tools") or {}).get("p2rank") or {}
    if not p2.get("version") or not p2.get("jvm"):
        problems.append("p2rank version or JVM not recorded")

    pinned = {}
    for line in REQ.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, ver = line.split("==", 1)
            pinned[name.strip()] = ver.strip()
    for name, ver in (rec.get("packages") or {}).items():
        if ver and pinned.get(name) != ver:
            problems.append(f"requirements.txt pins {name}=={pinned.get(name)} "
                            f"but the record says {ver}")
    problems.extend(_rival_pins(rec.get("packages") or {}))

    if problems:
        print("ENVIRONMENT.json audit failed:")
        for p in problems:
            print(f"  - {p}")
        print("  regenerate: PYTHONPATH=src python3.12 "
              "tools/emit_environment.py")
        return 1
    pkgs = ", ".join(f"{k} {v}" for k, v in sorted(
        (rec.get("packages") or {}).items()) if v)
    print(f"environment record complete: python "
          f"{rec['python']['version']}, {pkgs}, blas {rec['blas']['name']}, "
          f"p2rank {p2['version']}; sha {rec['environment_sha256'][:16]}")
    scanned = [p for p in _tracked_pin_files() if p != REQ and p.exists()]
    print(f"no rival pin in {len(scanned)} other dependency "
          f"declaration(s): "
          + ", ".join(str(p.relative_to(ROOT)) for p in scanned))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless THIS machine matches the record")
    ap.add_argument("--audit", action="store_true",
                    help="machine-independent: the record is present, complete "
                         "and agrees with requirements.txt")
    args = ap.parse_args(argv)

    rec = build()
    text = json.dumps(rec, indent=2, allow_nan=False) + "\n"
    req = requirements(rec)

    if args.audit:
        # What CI can meaningfully assert. CI runs on a different machine by
        # construction, so demanding a byte-identical stack there would make the
        # gate permanently red and it would be switched off within a week. What
        # is checkable anywhere: the record exists, carries every field a reader
        # needs, and does not contradict the pins shipped beside it.
        return _audit()

    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        old = json.loads(OUT.read_text())
        if old.get("environment_sha256") != rec["environment_sha256"]:
            print("STALE ENVIRONMENT.json: this machine's stack differs from "
                  "the recorded one")
            print(f"  recorded {old.get('environment_sha256', '')[:16]}  "
                  f"here {rec['environment_sha256'][:16]}")
            for k in ("packages", "blas"):
                if old.get(k) != rec.get(k):
                    print(f"  {k}: {old.get(k)} -> {rec.get(k)}")
            return 1
        print("ENVIRONMENT.json matches this machine")
        return 0

    OUT.write_text(text)
    REQ.write_text(req)
    p2 = rec["tools"]["p2rank"]
    print(f"python {rec['python']['version']} on {rec['platform']['system']} "
          f"{rec['platform']['machine']}")
    print(f"numpy {rec['packages']['numpy']}, scipy {rec['packages']['scipy']}, "
          f"blas {rec['blas']['name']}")
    print(f"p2rank {p2['version']} under {p2['jvm']}")
    print(f"environment_sha256 {rec['environment_sha256'][:16]}...")
    print(f"wrote {OUT.relative_to(ROOT)} and {REQ.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
