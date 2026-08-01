#!/usr/bin/env python3
"""Read the resolution stratification once, under the plan, and take what comes back.

Everything that could be chosen is in ``PREREGISTERED_RESOLUTION.json``, committed
before this ran: the cut, the statistic, the endpoint, the seed, the bootstrap count,
the control and the sentence to write under each outcome. This tool applies them and
looks the verdict up rather than interpreting it.

No method is re-run and no metric is recomputed. The per-unit ROC-AUCs come from
``RECOVERY_READ.json``, where all four were computed through one harness, so a
difference between two methods here cannot be a difference in how the metric was
formed.

Usage: PYTHONPATH=src:tools python3.12 tools/resolution_read.py [--check] [--rerun R]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_RESOLUTION.json"
RES = ROOT / "results/official_fold/OFFICIAL_RESOLUTIONS.json"
RECOVERY = ROOT / "results/official_fold/RECOVERY_READ.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/official_fold/RESOLUTION_READ.json"
SCHEMA = "geoaudit.resolution_read.v1"


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan() -> tuple[dict, dict]:
    rel = str(PLAN.relative_to(ROOT))
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is modified or untracked. A plan that can still "
                         f"be edited is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused.")
    plan = json.loads(PLAN.read_text())
    for key, path in (("per_unit_aucs", RECOVERY), ("resolutions", RES)):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != plan["inputs"][key]["sha256"]:
            raise SystemExit(f"{path.name} moved after the plan was written")
    return plan, {"artifact": rel, "commit": sha,
                  "committed_at": _git("log", "-1", "--format=%cI", sha),
                  "subject": _git("log", "-1", "--format=%s", sha)}


def _covariates() -> dict[str, dict]:
    universe = {f"{r['pdb']}_{r['chain']}": r["n_universe"]
                for r in json.loads(PER_STRUCTURE.read_text())}
    out = {}
    for e in json.loads(MANIFEST.read_text())["entries"]:
        uid = f"{e['pdb']}_{e['chain']}"
        lab = json.loads((ROOT / e["label_path"]).read_text())
        pos = len({int(r) for r in (lab.get("cryptic_residues") or [])})
        n = universe.get(uid)
        out[uid] = {"length": n, "n_positive": pos,
                    "positive_rate": (pos / n) if n else None}
    return out


def _mean_difference(rows: list[dict], a: str, b: str) -> float:
    return float(np.mean([r[a] - r[b] for r in rows]))


def _endpoint(high: list[dict], low: list[dict], a: str, b: str,
              seed: int, n_boot: int, alpha: float) -> dict:
    """High-stratum margin minus low-stratum margin, resampled within strata."""
    dh = np.array([r[a] - r[b] for r in high])
    dl = np.array([r[a] - r[b] for r in low])
    obs = float(dh.mean() - dl.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for t in range(n_boot):
        draws[t] = (rng.choice(dh, dh.size, replace=True).mean()
                    - rng.choice(dl, dl.size, replace=True).mean())
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"high_margin": round(float(dh.mean()), 6),
            "low_margin": round(float(dl.mean()), 6),
            "endpoint": round(obs, 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_high": int(dh.size), "n_low": int(dl.size)}


def read(plan: dict, prov: dict) -> dict:
    cut = plan["strata"]["cut_angstrom"]
    seed = plan["primary"]["seed"]
    n_boot = plan["primary"]["n_bootstrap"]
    alpha = plan["primary"]["alpha"]

    resolution = {u["unit"]: u["resolution"]
                  for u in json.loads(RES.read_text())["units"]}
    cov = _covariates()
    high, low, unplaced = [], [], []
    for r in json.loads(RECOVERY.read_text())["per_unit"]:
        res = resolution.get(r["unit"])
        if res is None:
            unplaced.append(r["unit"])
            continue
        row = dict(r, resolution=res, **cov.get(r["unit"], {}))
        (high if res <= cut else low).append(row)

    ours = _endpoint(high, low, "ours", "plmnn", seed, n_boot, alpha)
    control = _endpoint(high, low, "p2rank", "plmnn", seed + 1, n_boot, alpha)

    def _levels(rows):
        return {m: round(float(np.mean([r[m] for r in rows])), 6)
                for m in ("ours", "p2rank", "plmnn", "pocketminer")}

    def _cov(rows):
        L = [r["length"] for r in rows if r.get("length")]
        P = [r["positive_rate"] for r in rows if r.get("positive_rate") is not None]
        return {"mean_chain_length": round(float(np.mean(L)), 2) if L else None,
                "mean_positive_rate": round(float(np.mean(P)), 5) if P else None,
                "mean_resolution": round(float(np.mean(
                    [r["resolution"] for r in rows])), 3)}

    strata = {"high": {"definition": f"resolution <= {cut} A", "n": len(high),
                       "mean_per_unit_auc": _levels(high), "covariates": _cov(high)},
              "low": {"definition": f"resolution > {cut} A", "n": len(low),
                      "mean_per_unit_auc": _levels(low), "covariates": _cov(low)}}

    lc, hc = strata["low"]["covariates"], strata["high"]["covariates"]
    length_ratio = (hc["mean_chain_length"] / lc["mean_chain_length"]
                    if lc["mean_chain_length"] else None)
    rate_ratio = (hc["mean_positive_rate"] / lc["mean_positive_rate"]
                  if lc["mean_positive_rate"] else None)
    covariate_dominates = bool(
        (length_ratio is not None and not 0.7 <= length_ratio <= 1.43)
        or (rate_ratio is not None and not 0.7 <= rate_ratio <= 1.43))

    if covariate_dominates:
        headline = "a_covariate_explains_it"
    elif not ours["excludes_zero"]:
        headline = "does_not_resolve"
    elif ours["endpoint"] < 0:
        headline = "widens"
    elif control["endpoint"] > 0:
        headline = "narrows_and_the_control_agrees"
    else:
        headline = "narrows_but_the_control_disagrees"

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "declared": plan["declared"],
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        # Named explicitly because the ledger falls back to "table field variant"
        # for an artifact that does not say, and counted this read as a twelfth
        # architecture evaluated on the fold. It evaluates none: it re-strata a
        # per-unit number an earlier read already froze.
        "method": "table_field",
        "status": "exploratory",
        "question": plan["question"],
        "hypothesis": plan["hypothesis_and_its_source"]["statement"],
        "plan": prov,
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "rescored_anything": False,
        "n_units": len(high) + len(low),
        "units_without_a_resolution": unplaced,
        "strata": strata,
        "primary_ours_minus_plmnn": ours,
        "control_p2rank_minus_plmnn": control,
        "covariate_screen": {
            "high_over_low_chain_length": (round(length_ratio, 3)
                                           if length_ratio else None),
            "high_over_low_positive_rate": (round(rate_ratio, 3)
                                            if rate_ratio else None),
            "band_that_counts_as_comparable": [0.7, 1.43],
            "dominates": covariate_dominates,
            "why_a_band_and_not_a_test": (
                "the question is whether the two strata are similar enough for a "
                "stratum effect to be about resolution. A significance test on a "
                "covariate answers a different question and would pass or fail on "
                "the stratum sizes"),
        },
        "outcome": headline,
        "conclusion": plan["outcome_sentences_written_in_advance"][headline],
        "what_this_cannot_show": plan["what_this_cannot_show"],
    }


def report(d: dict) -> None:
    print(f"\nresolution read, {d['declared']}: {d['n_units']} units")
    for name in ("high", "low"):
        s = d["strata"][name]
        lv, c = s["mean_per_unit_auc"], s["covariates"]
        print(f"  {name:4s} {s['definition']:22s} n={s['n']:3d}  "
              f"res {c['mean_resolution']:.2f} A  len {c['mean_chain_length']:.0f}  "
              f"pos {c['mean_positive_rate']:.4f}")
        print(f"       ours {lv['ours']:.4f}  p2rank {lv['p2rank']:.4f}  "
              f"plmnn {lv['plmnn']:.4f}  pocketminer {lv['pocketminer']:.4f}")
    p, c = d["primary_ours_minus_plmnn"], d["control_p2rank_minus_plmnn"]
    print(f"\n  ours - plmnn   high {p['high_margin']:+.4f}  "
          f"low {p['low_margin']:+.4f}  endpoint {p['endpoint']:+.4f} "
          f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] "
          f"{'resolved' if p['excludes_zero'] else 'crosses zero'}")
    print(f"  p2rank - plmnn high {c['high_margin']:+.4f}  "
          f"low {c['low_margin']:+.4f}  endpoint {c['endpoint']:+.4f} "
          f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] "
          f"{'resolved' if c['excludes_zero'] else 'crosses zero'}   <- control")
    cs = d["covariate_screen"]
    print(f"  covariates high/low: length {cs['high_over_low_chain_length']}, "
          f"positive rate {cs['high_over_low_positive_rate']}"
          f"{'  DOMINATES' if cs['dominates'] else ''}")
    print(f"\n  outcome: {d['outcome']}")
    print(f"  {d['conclusion']}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rerun", metavar="REASON")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        report(json.loads(OUT.read_text()))
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    if OUT.exists() and not a.rerun:
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists. This read runs once.")
    plan, prov = _plan()
    d = read(plan, prov)
    if a.rerun:
        d["reread"] = {"reason": a.rerun}
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    report(d)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
