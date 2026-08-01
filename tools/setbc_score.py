#!/usr/bin/env python3
"""Score five methods on the frozen cryo-EM sets, under the committed plan.

What runs here
--------------
``geometry_field`` (the 1269-column stack this read exists to confirm),
``table_field`` (the deployed 645-wire detector it is compared against), and the
three published baselines. All five on the same 45 units, so that a difference
between two of them is a difference between the methods and not between two
residue universes or two metric implementations.

Why the plan is verified before anything is scored
--------------------------------------------------
This tool refuses unless ``PREREGISTERED_SETBC.json`` is committed, clean in the
working tree, and pins the digests the sets and the compiled field currently
have. The order ``AGENTS.md`` fixes is build, freeze, hash, preregister, read
once, and a scorer that runs without checking the plan is the step where that
order stops being enforced and becomes a description of what somebody remembers
doing.

Why the baseline runners are monkey-patched rather than edited
--------------------------------------------------------------
``plmnn_embed.py`` and ``pocketminer_run.py`` have ``--external`` modes whose
paths point at Set A's receptors, and Set A's read is spent and pinned. Adding a
flag to either would edit a tool in the blast radius of a frozen artifact for a
reason unrelated to it, which ``AGENTS.md`` records refusing once already --- a
retry count was raised in a probe by monkey-patching rather than in
``external_inventory.py``. The same judgement applies here: the paths are
redirected from this file, the runners themselves are untouched, and what they
compute is unchanged.

Resumable per method
--------------------
Each method writes its own archive and is skipped if that archive already covers
every unit. A pLM-NN pass is an encoder run and a PocketMiner pass needs a second
interpreter; losing either to a crash in the fifth method would be a waste and
would tempt somebody to run the read on four.

Nothing here compares a score against a label. That is the read, and it is a
separate tool so that a mistake in the comparison does not cost the scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.setbc_scores.v1"
PLAN = ROOT / "results/external/PREREGISTERED_SETBC.json"
MANIFEST = ROOT / "data/external/setbc_manifest.json"
PREDS = ROOT / "results/external/setbc_predictions"
P2RANK_RAW = ROOT / "results/external/setbc_p2rank_raw"
OUT = ROOT / "results/external/SETBC_SCORES.json"

OURS = ("geometry_field", "table_field")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a],
                          capture_output=True, text=True).stdout.strip()


def _verify_plan() -> dict:
    """Refuse to score unless the plan is committed, clean and still accurate."""
    if not PLAN.is_file():
        raise SystemExit(
            f"{PLAN.relative_to(ROOT)} does not exist. Write it with "
            f"tools/preregister_setbc.py and commit it before scoring anything")
    plan = json.loads(PLAN.read_text())
    rel = PLAN.relative_to(ROOT).as_posix()

    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is dirty in the working tree. A plan that can still be "
            f"edited is not a plan")
    added = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
    if not added.strip():
        raise SystemExit(
            f"{rel} is not committed. The whole value of the plan is that it "
            f"precedes the scores in the history")

    for key, path in (("set_b", ROOT / plan["sets"]["set_b"]["artifact"]),
                      ("set_c", ROOT / plan["sets"]["set_c"]["artifact"])):
        want = plan["sets"][key]["sha256"]
        got = _sha(path)
        if got != want:
            raise SystemExit(
                f"{path.name} hashes to {got[:16]} and the plan pinned "
                f"{want[:16]}. The set moved after the plan was written")
    if _sha(MANIFEST) != plan["manifest"]["sha256"]:
        raise SystemExit(
            "the manifest moved after the plan pinned it, so the unit set being "
            "scored is not the one the plan declares")
    for name in OURS:
        spec = plan["methods"][name]
        got = _sha(ROOT / spec["artifact"])
        if got != spec["sha256"]:
            raise SystemExit(
                f"{spec['artifact']} hashes to {got[:16]} and the plan pinned "
                f"{spec['sha256'][:16]}. A detector recompiled after the plan is "
                f"a different experiment wearing the plan's licence")
    print(f"plan verified: committed, clean, and every pinned digest matches")
    return plan


def _entries() -> list[dict]:
    man = json.loads(MANIFEST.read_text())
    for e in man["entries"]:
        p = ROOT / e["receptor_path"]
        if _sha(p) != e["receptor_sha256"]:
            raise SystemExit(
                f"{e['receptor_path']} does not match the digest the manifest "
                f"pins. A receptor edited after the plan is a different input")
    return man["entries"]


def _archive(method: str) -> Path:
    return PREDS / f"{method}.json"


def _covered(method: str, units: set[str]) -> bool:
    p = _archive(method)
    if not p.is_file():
        return False
    have = set(json.loads(p.read_text()).get("units") or {})
    return units <= have


def _run_ours(method: str, entries: list[dict]) -> dict:
    from pocket_bench.methods import geometry_field, table_field
    from pocket_bench.paths import STATUS_OK
    mod = {"geometry_field": geometry_field, "table_field": table_field}[method]
    out, t0 = {}, time.perf_counter()
    for i, e in enumerate(entries, start=1):
        uid = f"{e['pdb']}_{e['chain']}"
        pred = mod.predict(ROOT / e["receptor_path"], pdb_id=e["pdb"],
                           chain=e["chain"])
        if pred.get("status") != STATUS_OK:
            raise SystemExit(
                f"{method} failed on {uid}: status {pred.get('status')!r}, "
                f"error {pred.get('error')}. A read with a method missing on "
                f"some units is not the read the plan declares")
        # prediction() flattens its extras to the top level rather than nesting
        # them under "extra"; reading the nested key gave a KeyError that the
        # broad except in predict() turned into a crash status with no message.
        out[uid] = {"residue_scores": pred["residue_scores"],
                    "residue_positive": pred["residue_positive"],
                    "n_residues": pred["n_residues"],
                    "runtime_s": round(pred["runtime_s"], 3)}
        if i % 10 == 0 or i == len(entries):
            print(f"  {method}: {i}/{len(entries)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def _run_p2rank(entries: list[dict]) -> dict:
    from pocket_bench.methods import p2rank_wrap
    from pocket_bench.paths import STATUS_OK
    P2RANK_RAW.mkdir(parents=True, exist_ok=True)
    out, t0 = {}, time.perf_counter()
    for i, e in enumerate(entries, start=1):
        uid = f"{e['pdb']}_{e['chain']}"
        pred = p2rank_wrap.predict(ROOT / e["receptor_path"], pdb_id=e["pdb"],
                                   chain=e["chain"], archive_dir=P2RANK_RAW)
        if pred.get("status") != STATUS_OK:
            raise SystemExit(f"p2rank failed on {uid}: "
                             f"status {pred.get('status')!r}, "
                             f"error {pred.get('error')}")
        out[uid] = {"residue_scores": pred.get("residue_scores") or {},
                    "residue_positive": pred.get("residue_positive") or [],
                    "n_residues": pred.get("n_residues"),
                    "runtime_s": round(pred["runtime_s"], 3)}
        if i % 5 == 0 or i == len(entries):
            print(f"  p2rank: {i}/{len(entries)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def _write(method: str, units: dict) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    _archive(method).write_text(json.dumps({
        "schema": f"geoaudit.setbc_prediction.{method}.v1",
        "clinical_grade": False,
        "method": method,
        "n_units": len(units),
        "manifest_sha256": _sha(MANIFEST),
        "units": units,
    }, indent=1) + "\n")
    print(f"  wrote {_archive(method).relative_to(ROOT)}")


def build(methods: list[str], write: bool) -> int:
    plan = _verify_plan()
    entries = _entries()
    ids = {f"{e['pdb']}_{e['chain']}" for e in entries}
    print(f"{len(entries)} units, digests verified against the manifest\n")

    done = []
    for method in methods:
        if _covered(method, ids):
            print(f"{method}: archive already covers every unit, skipping")
            done.append(method)
            continue
        print(f"{method}: scoring", flush=True)
        if method in OURS:
            units = _run_ours(method, entries)
        elif method == "p2rank":
            units = _run_p2rank(entries)
        else:
            raise SystemExit(
                f"{method} is not scored by this tool yet. pLM-NN and "
                f"PocketMiner need their own runners redirected at "
                f"data/external/setbc_receptors; run them and archive under "
                f"{PREDS.relative_to(ROOT)}")
        if write:
            _write(method, units)
        done.append(method)

    if write:
        OUT.write_text(json.dumps({
            "schema": SCHEMA,
            "clinical_grade": False,
            "reads_test_fold": False,
            "reads_any_external_unit": True,
            "what_this_is": (
                "raw per-residue scores for five methods on the 45 units of the "
                "frozen cryo-EM sets, under the plan committed before it ran"),
            "no_comparison_is_made_here": (
                "not one score is compared against a label in this file. The "
                "comparison is tools/setbc_read.py, kept separate so that a "
                "mistake in it does not cost the scoring"),
            "plan": {"artifact": PLAN.relative_to(ROOT).as_posix(),
                     "sha256": _sha(PLAN)},
            "manifest_sha256": _sha(MANIFEST),
            "n_units": len(entries),
            "methods_archived": sorted(done),
            "archives": {m: {"path": _archive(m).relative_to(ROOT).as_posix(),
                             "sha256": _sha(_archive(m))}
                         for m in sorted(done) if _archive(m).is_file()},
            "head": _git("rev-parse", "HEAD"),
        }, indent=2) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+",
                    default=["geometry_field", "table_field", "p2rank"])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    return build(a.methods, a.write)


if __name__ == "__main__":
    raise SystemExit(main())
