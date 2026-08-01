#!/usr/bin/env python3
"""The read of Set B and Set C, once, under the plan committed before it.

What this settles
-----------------
Whether the four wire families' +0.0121 over the deployed detector --- measured on
twelve cluster-disjoint halvings of the training fold, positive on 12 of 12 ---
survives on units no part of their development could have seen. Sets B and C were
built, frozen and hashed before any of those families existed and have never been
read.

Everything about how this is read was fixed in ``PREREGISTERED_SETBC.json``: the
statistic, the four co-primary comparisons, the Bonferroni level, the two
secondary analyses, the coverage, and the sentence to write under each of six
outcomes, four of which are outcomes where the work does not hold up. This tool
applies that table rather than interpreting the numbers.

What it refuses
---------------
Every refusal the plan asked for. A set, manifest or compiled field whose digest
has moved. A plan that is uncommitted or dirty. A method missing on any unit. An
existing read, unless a reason is recorded --- because the one thing that cannot
be undone here is reading twice and keeping the better answer.

One thing it will not do
------------------------
It will not report a comparison the plan did not name, and it will not promote a
secondary to a headline. ``SETBC_DIFFICULTY.json``, committed before this ran,
records the confound these sets carry --- chains 1.6 times the official fold's at
the median, and ``geometry_field`` reads further than ``table_field`` --- together
with P2Rank as its control. That artifact is quoted here so that the reading
arrives with it rather than after it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from plmnn_read import _interval                                   # noqa: E402
from pocket_bench.metrics import residue_auc_pr                    # noqa: E402
from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.setbc_read.v1"
PLAN = ROOT / "results/external/PREREGISTERED_SETBC.json"
MANIFEST = ROOT / "data/external/setbc_manifest.json"
PREDS = ROOT / "results/external/setbc_predictions"
DIFFICULTY = ROOT / "results/external/SETBC_DIFFICULTY.json"
OUT = ROOT / "results/external/SETBC_READ.json"

METHODS = ("geometry_field", "table_field", "p2rank", "pocketminer", "plmnn")
TIE_ATOL = 1e-12


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a],
                          capture_output=True, text=True).stdout.strip()


def _verify(plan: dict) -> None:
    rel = PLAN.relative_to(ROOT).as_posix()
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is dirty; a plan that can still be edited is "
                         f"not a plan")
    if not _git("log", "--diff-filter=A", "--format=%H", "--", rel).strip():
        raise SystemExit(f"{rel} is not committed. Its whole value is that it "
                         f"precedes this read in the history")
    for key in ("set_b", "set_c"):
        p = ROOT / plan["sets"][key]["artifact"]
        if _sha(p) != plan["sets"][key]["sha256"]:
            raise SystemExit(f"{p.name} moved after the plan pinned it")
    if _sha(MANIFEST) != plan["manifest"]["sha256"]:
        raise SystemExit("the manifest moved after the plan pinned it")
    for name in ("geometry_field", "table_field"):
        spec = plan["methods"][name]
        if _sha(ROOT / spec["artifact"]) != spec["sha256"]:
            raise SystemExit(
                f"{spec['artifact']} moved after the plan pinned it. A detector "
                f"recompiled between plan and read is a different experiment "
                f"wearing the plan's licence")


def _truth() -> dict[str, set[int]]:
    out = {}
    for e in json.loads(MANIFEST.read_text())["entries"]:
        lab = json.loads((ROOT / e["label_path"]).read_text())
        out[f"{e['pdb']}_{e['chain']}"] = set(lab["cryptic_residues"])
    return out


def _archive(method: str) -> dict[str, dict]:
    p = PREDS / f"{method}.json"
    if not p.is_file():
        raise SystemExit(
            f"{p.relative_to(ROOT)} does not exist. The plan names four "
            f"co-primary comparisons and this read will not run on a subset: "
            f"computing some, looking, and then computing the rest is "
            f"sequential peeking and the Bonferroni level is over four")
    return json.loads(p.read_text())["units"]


def _auc(keys: list[int], scores: dict[int, float], pos: set[int],
         called: list[int]) -> float | None:
    """One per-unit ROC-AUC, always through the harness's own function."""
    res = residue_auc_pr(
        [], sorted(pos), sorted(keys),
        {"residue_scores": {str(k): scores[k] for k in keys},
         "residue_positive": called})
    v = res.get("residue_auc")
    return None if v is None else float(v)


def _shared(truth: dict) -> dict:
    """Per unit: the residues every method scored, with labels and scores."""
    arch = {m: _archive(m) for m in METHODS}
    sets = {f"{e['pdb']}_{e['chain']}": e["set"]
            for e in json.loads(MANIFEST.read_text())["entries"]}
    units, lost, skipped = {}, [], []
    for uid in sorted(truth):
        s = {}
        for m in METHODS:
            raw = (arch[m].get(uid) or {}).get("residue_scores") or {}
            s[m] = {int(k): float(v) for k, v in raw.items()}
        keys = sorted(set.intersection(*(set(s[m]) for m in METHODS)))
        widest = max(len(s[m]) for m in METHODS)
        if widest - len(keys):
            lost.append({"unit": uid, "n_shared": len(keys),
                         **{f"n_{m}": len(s[m]) for m in METHODS}})
        pos = truth[uid]
        if not keys or not (pos & set(keys)):
            skipped.append({
                "unit": uid, "n_shared": len(keys),
                "why": ("no residue shared by all five methods" if not keys else
                        "no labelled residue survives the intersection, so no "
                        "ROC-AUC exists on the shared universe")})
            continue
        units[uid] = {
            "keys": keys, "pos": pos, "set": sets[uid],
            "n_cryptic_shared": len(pos & set(keys)),
            **{m: s[m] for m in METHODS},
            **{f"{m}_called": [int(k) for k in
                               ((arch[m].get(uid) or {}).get("residue_positive")
                                or [])] for m in METHODS},
        }
    return {"units": units, "residues_not_shared": lost,
            "units_skipped": skipped}


def _per_unit(units: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for uid, u in units.items():
        for m in METHODS:
            v = _auc(u["keys"], u[m], u["pos"], u[f"{m}_called"])
            if v is not None:
                out[m][uid] = v
    return out


def _paired(a: dict, b: dict, seed: int, n_boot: int, corrected: float) -> dict:
    shared = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in shared])
    out = _interval(d, seed, n_boot, corrected)
    out["n"] = len(shared)
    out["n_first_ahead"] = int((d > TIE_ATOL).sum())
    out["n_second_ahead"] = int((d < -TIE_ATOL).sum())
    out["level_first"] = round(float(np.mean([a[u] for u in shared])), 6)
    out["level_second"] = round(float(np.mean([b[u] for u in shared])), 6)
    return out


def _verdict(block: dict, spec: dict) -> str:
    """The plan's own table, applied rather than interpreted."""
    lo, hi = block["ci"]
    if "transfers_if" not in spec:
        return ("leads" if lo > 0 else "behind" if hi < 0 else "unresolved")
    if lo <= 0 <= hi:
        return "does_not_resolve"
    if hi < 0:
        return "negative"
    pred = spec["predicted"]
    return "transfers_and_replicates" if lo <= pred <= hi else \
        "transfers_but_smaller" if hi < pred else "transfers_and_larger"


def build(reason: str | None, write: bool) -> int:
    plan = json.loads(PLAN.read_text())
    _verify(plan)
    print("plan verified: committed, clean, every pinned digest matches\n")
    if OUT.exists() and not reason:
        raise SystemExit(
            f"{OUT.relative_to(ROOT)} already exists. Re-reading a spent set and "
            f"keeping the better answer is the one thing that cannot be undone "
            f"here, so pass --reason with why this read is being taken again")

    truth = _truth()
    sh = _shared(truth)
    units = sh["units"]
    aucs = _per_unit(units)
    st = plan["statistic"]
    seed, n_boot = st["seed"], st["n_boot"]
    corrected = 1.0 - (1.0 - st["ci_level"]) / len(plan["co_primary_comparisons"])

    co = {}
    for i, spec in enumerate(plan["co_primary_comparisons"]):
        key = spec["key"]
        _, other = key.split("geometry_field_minus_")
        block = _paired(aucs["geometry_field"], aucs[other], seed + i, n_boot,
                        corrected)
        block["predicted"] = spec["predicted"]
        block["verdict"] = _verdict(block, spec)
        co[key] = block

    per_set = {}
    for name in ("set_b", "set_c"):
        keep = {u for u, d in units.items() if d["set"] == name}
        if len(keep) < 3:
            per_set[name] = {"n": len(keep),
                             "why_omitted": "fewer than three units"}
            continue
        a = {u: v for u, v in aucs["geometry_field"].items() if u in keep}
        b = {u: v for u, v in aucs["table_field"].items() if u in keep}
        per_set[name] = _paired(a, b, seed + 100, n_boot, corrected)

    strata = {}
    for label, pred in (("under_ten_cryptic", lambda d: d["n_cryptic_shared"] < 10),
                        ("ten_or_more", lambda d: d["n_cryptic_shared"] >= 10)):
        keep = {u for u, d in units.items() if pred(d)}
        if len(keep) < 3:
            strata[label] = {"n": len(keep),
                             "why_omitted": "fewer than three units"}
            continue
        a = {u: v for u, v in aucs["geometry_field"].items() if u in keep}
        b = {u: v for u, v in aucs["table_field"].items() if u in keep}
        strata[label] = _paired(a, b, seed + 200, n_boot, corrected)

    primary = co["geometry_field_minus_table_field"]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": True,
        "status": "confirmatory" if primary["verdict"].startswith("transfers")
                  else "read taken, outcome negative or unresolved",
        "plan": {"artifact": PLAN.relative_to(ROOT).as_posix(),
                 "sha256": _sha(PLAN),
                 "committed_in": _git("log", "--diff-filter=A", "--format=%H",
                                      "--",
                                      PLAN.relative_to(ROOT).as_posix()
                                      ).splitlines()[-1]},
        "sets": {k: {"sha256": v["sha256"],
                     "frozen_at_prefix": v["frozen_at_prefix"]}
                 for k, v in plan["sets"].items()},
        "field_sha256": plan["methods"]["geometry_field"]["sha256"],
        "coverage": {
            "n_units_in_the_manifest": len(truth),
            "n_units_compared": len(units),
            "units_skipped": sh["units_skipped"],
            "n_units_losing_residues": len(sh["residues_not_shared"]),
            "residues_not_shared": sh["residues_not_shared"][:10],
            "why": st["residue_universe"],
        },
        "levels": {m: round(float(np.mean(list(aucs[m].values()))), 6)
                   for m in METHODS},
        "co_primary": co,
        "multiplicity": {"n_comparisons": len(co),
                         "corrected_ci_level": round(corrected, 6),
                         "correction": plan["multiplicity"]["correction"]},
        "secondary_per_set": per_set,
        "secondary_by_pocket_size": strata,
        "the_confound_named_before_this_read": {
            "artifact": DIFFICULTY.relative_to(ROOT).as_posix(),
            "sha256": _sha(DIFFICULTY) if DIFFICULTY.exists() else None,
            "what": ("these chains are 1.6x the official fold's at the median "
                     "and geometry_field reads further than table_field, so the "
                     "difference could be inflated by set shape rather than by "
                     "the families generalising"),
            "control": "p2rank's movement, which was +0.0035 between the "
                       "official fold and Set A",
        },
        "sentence_the_plan_fixed": plan["sentences_fixed_before_the_read"].get(
            primary["verdict"], plan["sentences_fixed_before_the_read"].get(
                "does_not_resolve")),
        "reread_reason": reason,
    }

    print(f"{len(units)} of {len(truth)} units compared "
          f"({len(sh['units_skipped'])} skipped)\n")
    print("mean per-unit ROC-AUC on the shared universe")
    for m in METHODS:
        print(f"  {m:<16} {doc['levels'][m]:.6f}  (n={len(aucs[m])})")
    print(f"\nco-primary, Bonferroni level {corrected:.4f}")
    for key, b in co.items():
        print(f"  {key}")
        print(f"    {b['mean']:+.6f}  95% [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"
              f"  bonf [{b['ci_bonferroni'][0]:+.4f}, "
              f"{b['ci_bonferroni'][1]:+.4f}]"
              f"  {b['n_first_ahead']}/{b['n_second_ahead']}"
              f"  predicted {b['predicted']:+.4f}  -> {b['verdict']}")
    print(f"\nsentence fixed by the plan:\n  {doc['sentence_the_plan_fixed']}")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--reason", type=str, default=None,
                    help="why an existing read is being taken again")
    a = ap.parse_args(argv)
    return build(a.reason, a.write)


if __name__ == "__main__":
    raise SystemExit(main())
