#!/usr/bin/env python3
"""The eighth read: where on the fold the two detectors differ, and the tie count.

Runs the plan in results/architecture_sweep/PREREGISTERED_SUBGROUPS.json and
refuses to start until that plan is committed, clean and an ancestor of HEAD, and
until the covariate artifact still hashes to what the plan recorded. Both guards
exist for the same reason: a subgroup analysis whose groups can be redrawn after
the numbers are seen is not an analysis.

Nothing is rescored. The per-unit ROC-AUCs were frozen at read one and this
partitions them. Partitioning frozen numbers and testing inside the parts is a
new inferential use of the fold, so the ledger indexes it as read eight.

The plan declares the result exploratory before any number is produced, caps what
may be concluded, and writes the sentence for each outcome including the two
unfavourable ones. This file is not allowed to reach a conclusion the plan did
not already name.

Usage: PYTHONPATH=src:tools python3.12 tools/subgroup_read.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_SUBGROUPS.json"
COV = ROOT / "results/official_fold/SUBGROUP_COVARIATES.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
PREREG_READ = ROOT / "results/official_fold/PREREGISTERED_READ.json"
OUT = ROOT / "results/official_fold/SUBGROUP_READ.json"

SCHEMA = "geoaudit.subgroup_read.v1"
READ_INDEX = 8
METHOD = "table_field"
BASELINE = "p2rank"
BANDS = ("low", "mid", "high")
# A tie in a per-residue ROC-AUC computed on the same universe is exact equality
# of two rank statistics, not approximate agreement. The tolerance is here only
# to absorb the last bit of float representation, not to widen what "tie" means.
TIE_ATOL = 1e-12


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan_provenance() -> dict:
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the commit that fixed the subgroups "
            "may not be present and the ordering cannot be checked. Fetch the "
            "full history (actions/checkout with fetch-depth: 0) and retry.")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is modified or untracked. A plan that can still be edited "
            f"is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(
            f"{rel} has no commit. The subgroups have to be in the history "
            f"before the fold is partitioned by them; this read is refused.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    return {"artifact": rel, "committed_in": sha,
            "committed_at": _git("log", "-1", "--format=%cI", sha),
            "subject": _git("log", "-1", "--format=%s", sha),
            "is_ancestor_of_head": True}


def _paired() -> tuple[np.ndarray, list[str]]:
    rows = json.loads(TELEMETRY.read_text())["rows"]
    by = {m: {} for m in (METHOD, BASELINE)}
    for r in rows:
        if r["method"] in by and r.get("residue_auc") is not None:
            by[r["method"]][r["unit_id"]] = float(r["residue_auc"])
    shared = sorted(set(by[METHOD]) & set(by[BASELINE]))
    return (np.array([by[METHOD][u] - by[BASELINE][u] for u in shared]),
            shared)


def _ci(d: np.ndarray, rng: np.random.Generator, n_boot: int,
        level: float) -> dict:
    """Paired bootstrap mean with an interval at an arbitrary level.

    Taking the level as an argument rather than fixing 95% is what lets the
    corrected and uncorrected intervals come from the same resampling, so the
    only thing that differs between them is the quantile read off it.
    """
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None], "excludes_zero": None}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return {"n": int(n), "mean": round(float(d.mean()), 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with ties averaged, without pulling in scipy.

    scipy is a dependency here, but the two-line rank version keeps this file
    readable next to the bootstrap and gives the same number; ties are averaged
    the same way scipy does.
    """
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1, dtype=np.float64)
        # Average the ranks inside each tied run so a covariate with many
        # repeats (pocket size especially) is not ordered by array position.
        s = v[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return 0.0 if den == 0 else float((rx * ry).sum() / den)


def build() -> dict:
    prov = _plan_provenance()
    plan = json.loads(PLAN.read_text())
    if plan["status_declared_in_advance"] != "exploratory":
        raise SystemExit("the plan no longer declares itself exploratory")

    digest = hashlib.sha256(COV.read_bytes()).hexdigest()
    if digest != plan["covariate_artifact"]["sha256"]:
        raise SystemExit(
            f"the covariate artifact hashes to {digest[:12]} but the plan was "
            f"written against {plan['covariate_artifact']['sha256'][:12]}. The "
            f"bands would not be the ones that were preregistered, so this "
            f"read is refused.")

    cov = json.loads(COV.read_text())
    by_unit = {r["unit_id"]: r for r in cov["rows"]}
    d, units = _paired()
    missing = [u for u in units if u not in by_unit]
    if missing:
        raise SystemExit(f"{len(missing)} scored units have no covariate row: "
                         f"{missing[:5]}")

    n_boot = plan["statistic"]["n_boot"]
    rng = np.random.default_rng(plan["statistic"]["seed"])
    lvl_band = 1.0 - plan["multiplicity"]["corrected_level"]

    # Calibration first. If the overall numbers do not reproduce what read five
    # published, the partition is being computed on a different vector and no
    # subgroup number below would mean anything.
    read5 = json.loads(PREREG_READ.read_text())
    shape = read5["shape_of_the_differences"]
    ahead = int((d > TIE_ATOL).sum())
    behind = int((d < -TIE_ATOL).sum())
    ties = int(len(d) - ahead - behind)
    overall = _ci(d, np.random.default_rng(plan["statistic"]["seed"]),
                  n_boot, 0.95)
    calib = {
        "mean": overall["mean"],
        "mean_published_at_read_five": read5["mean_reported_beside_it"]["point"],
        "n_field_ahead": ahead,
        "n_field_ahead_published": shape["n_field_ahead"],
        "n_baseline_ahead": behind,
        "n_baseline_ahead_published": shape["n_baseline_ahead"],
        "reproduces": (
            abs(overall["mean"] - read5["mean_reported_beside_it"]["point"])
            < 1e-6
            and ahead == shape["n_field_ahead"]
            and behind == shape["n_baseline_ahead"]),
        "what_this_is": "a check that the arithmetic matches the published "
                        "read, not a finding",
    }
    if not calib["reproduces"]:
        raise SystemExit(
            f"the paired differences do not reproduce read five: mean "
            f"{overall['mean']} against "
            f"{read5['mean_reported_beside_it']['point']}, "
            f"{ahead}/{behind} ahead/behind against "
            f"{shape['n_field_ahead']}/{shape['n_baseline_ahead']}")

    # The full per-chain distribution, which the plan lists as unread: read five
    # published quantiles and counts but never the tie count nor the deciles.
    q = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    distribution = {
        "n": int(len(d)),
        "n_field_ahead": ahead,
        "n_baseline_ahead": behind,
        "n_tied": ties,
        "tie_definition": f"|difference| <= {TIE_ATOL:g} in per-residue ROC-AUC",
        "quantiles": {str(p): round(float(np.quantile(d, p)), 6) for p in q},
        "mean": overall["mean"],
        "sd": round(float(d.std(ddof=1)), 6),
        "n_losses_worse_than_5pct": int((d < -0.05).sum()),
        "n_wins_better_than_5pct": int((d > 0.05).sum()),
        "how_to_read_it": (
            "the field is ahead on more chains than it is behind, and its "
            "losses reach further than its wins. Those two facts are the whole "
            "reason the mean and every robust summary of the same vector "
            "disagree, and reporting the vector's shape is what lets a reader "
            "pick the summary their application calls for"),
    }

    results, per_cov = [], {}
    for spec in plan["covariates"]:
        name = spec["id"]
        lo_cut, hi_cut = spec["cuts"]
        vals = np.array([by_unit[u][name] if by_unit[u][name] is not None
                         else np.nan for u in units], dtype=np.float64)
        defined = ~np.isnan(vals)
        masks = {"low": defined & (vals < lo_cut),
                 "mid": defined & (vals >= lo_cut) & (vals < hi_cut),
                 "high": defined & (vals >= hi_cut)}
        sizes = [int(masks[b].sum()) for b in BANDS]
        if sizes != spec["group_sizes"]:
            raise SystemExit(
                f"{name}: the bands hold {sizes} units but the plan "
                f"preregistered {spec['group_sizes']}; the groups have been "
                f"redrawn and this read is refused")
        if int(sum(masks[b] for b in BANDS).max()) > 1:
            raise SystemExit(f"{name}: a unit falls in more than one band")

        bands = []
        for b in BANDS:
            db = d[masks[b]]
            unc = _ci(db, np.random.default_rng(plan["statistic"]["seed"]),
                      n_boot, 0.95)
            cor = _ci(db, np.random.default_rng(plan["statistic"]["seed"]),
                      n_boot, lvl_band)
            bands.append({
                "band": b, "n": unc["n"], "mean": unc["mean"],
                "median": round(float(np.median(db)), 6) if len(db) else None,
                "n_field_ahead": int((db > TIE_ATOL).sum()),
                "n_baseline_ahead": int((db < -TIE_ATOL).sum()),
                "ci95": unc["ci"],
                "excludes_zero_uncorrected": unc["excludes_zero"],
                "ci_bonferroni": cor["ci"],
                "excludes_zero_corrected": cor["excludes_zero"],
            })
            results.append({"covariate": name, "band": b, **bands[-1]})

        # A weighted reconstruction of the overall mean from the three band
        # means. It cannot fail unless a unit was dropped or double-counted,
        # which is exactly the silent failure worth a guard.
        recon = sum(x["mean"] * x["n"] for x in bands) / sum(
            x["n"] for x in bands)
        if abs(recon - overall["mean"]) > 1e-6:
            raise SystemExit(
                f"{name}: the band means weighted by band size give {recon:.6f} "
                f"but the overall mean is {overall['mean']:.6f}")

        rho = _spearman(vals[defined], d[defined])
        # Permutation rather than a normal approximation: 192 paired differences
        # with the tails this fold has are not a case for asymptotics.
        perm = np.array([_spearman(vals[defined],
                                   rng.permutation(d[defined]))
                         for _ in range(2000)])
        p = float((np.abs(perm) >= abs(rho)).mean())
        per_cov[name] = {
            "bands": bands,
            "monotone_in_the_mean": bool(
                bands[0]["mean"] <= bands[1]["mean"] <= bands[2]["mean"]
                or bands[0]["mean"] >= bands[1]["mean"] >= bands[2]["mean"]),
            "spearman_rho": round(rho, 4),
            "spearman_p_permutation": round(p, 4),
            "n_permutations": 2000,
            "trend_survives_correction": bool(
                p < plan["trend_test"]["corrected_level"]),
            "spread_between_bands": round(
                max(x["mean"] for x in bands) - min(x["mean"] for x in bands),
                6),
        }

    surviving = [r for r in results if r["excludes_zero_corrected"]]
    # A band that clears a corrected threshold while its own covariate shows no
    # trend is the shape noise takes when fifteen partitions are examined: there
    # is no dose-response to explain it, and the neighbouring bands do not order
    # around it. Computing that here rather than saying it in prose means the
    # caveat travels with the number.
    for r in surviving:
        v = per_cov[r["covariate"]]
        r["covariate_is_monotone"] = v["monotone_in_the_mean"]
        r["covariate_trend_survives"] = v["trend_survives_correction"]
        r["covariate_spearman_rho"] = v["spearman_rho"]
        r["covariate_spearman_p"] = v["spearman_p_permutation"]
        r["supported_by_a_trend"] = bool(v["monotone_in_the_mean"]
                                         and v["trend_survives_correction"])

    uncorrected_only = [r for r in results
                        if r["excludes_zero_uncorrected"]
                        and not r["excludes_zero_corrected"]]
    trends = [k for k, v in per_cov.items() if v["trend_survives_correction"]]
    favours_p2rank = [r for r in results
                      if r["excludes_zero_uncorrected"] and r["mean"] < 0]

    if surviving:
        key = "a_band_survives_correction"
    elif trends:
        key = "a_trend_survives_correction"
    else:
        key = "nothing_survives_correction"
    if favours_p2rank:
        key = "a_band_or_trend_favours_p2rank"

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "status": plan["status_declared_in_advance"],
        "question": plan["question"],
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "the per-unit numbers were frozen by read one, but partitioning "
            "them by a covariate and testing inside the parts draws new "
            "inferences from the fold, which is what the ledger counts"),
        "provenance_of_the_plan": prov,
        "covariate_artifact_sha256": digest,
        "telemetry_source": str(TELEMETRY.relative_to(ROOT)),
        "method": METHOD,
        "baseline": BASELINE,
        "n_paired_units": int(len(d)),
        "calibration_against_read_five": calib,
        "per_chain_distribution": distribution,
        "by_covariate": per_cov,
        "n_band_tests": len(results),
        "bonferroni_level": plan["multiplicity"]["corrected_level"],
        "bands_excluding_zero_after_correction": [
            {"covariate": r["covariate"], "band": r["band"], "mean": r["mean"],
             "ci": r["ci_bonferroni"], "n": r["n"],
             "supported_by_a_trend": r["supported_by_a_trend"],
             "covariate_is_monotone": r["covariate_is_monotone"],
             "covariate_spearman_rho": r["covariate_spearman_rho"],
             "covariate_spearman_p": r["covariate_spearman_p"]}
            for r in surviving],
        "n_surviving_bands_supported_by_a_trend": sum(
            1 for r in surviving if r["supported_by_a_trend"]),
        "how_to_read_a_surviving_band": (
            "a band that clears a corrected threshold while its own covariate "
            "shows no monotone trend has no dose-response behind it: the two "
            "neighbouring thirds of the fold do not order around it. That is "
            "the shape a false positive takes when fifteen partitions are "
            "examined, and it is why the trend test is reported beside every "
            "band rather than after them"),
        "bands_excluding_zero_only_before_correction": [
            {"covariate": r["covariate"], "band": r["band"], "mean": r["mean"],
             "ci95": r["ci95"], "n": r["n"]} for r in uncorrected_only],
        "trends_surviving_correction": trends,
        "bands_favouring_p2rank": [
            {"covariate": r["covariate"], "band": r["band"], "mean": r["mean"],
             "ci95": r["ci95"]} for r in favours_p2rank],
        "outcome_key": key,
        "outcome": plan["what_will_be_written_under_each_outcome"][key],
        "what_may_not_be_concluded": plan["decision_rules"][
            "no_subgroup_may_become_a_claim"],
    }


def _report(d: dict) -> None:
    p = d["per_chain_distribution"]
    print(f"read {d['test_fold_read_index']} ({d['status']}): "
          f"{p['n']} chains, {p['n_field_ahead']} ahead, "
          f"{p['n_baseline_ahead']} behind, {p['n_tied']} tied")
    print(f"  mean {p['mean']:+.4f}  sd {p['sd']:.4f}  "
          f"losses worse than 5pt: {p['n_losses_worse_than_5pct']}, "
          f"wins better: {p['n_wins_better_than_5pct']}")
    for name, v in d["by_covariate"].items():
        print(f"  {name:<14s} rho {v['spearman_rho']:+.3f} "
              f"p={v['spearman_p_permutation']:.4f}"
              f"{'  TREND' if v['trend_survives_correction'] else ''}")
        for b in v["bands"]:
            mark = ("  survives" if b["excludes_zero_corrected"]
                    else ("  uncorrected only" if b["excludes_zero_uncorrected"]
                          else ""))
            print(f"      {b['band']:<5s} n={b['n']:<4d} {b['mean']:+.4f} "
                  f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
                  f"{b['n_field_ahead']}/{b['n_baseline_ahead']}{mark}")
    for b in d["bands_excluding_zero_after_correction"]:
        print(f"\n  survives correction: {b['covariate']} {b['band']} "
              f"{b['mean']:+.4f} {b['ci']}, but its covariate's trend is "
              f"rho {b['covariate_spearman_rho']:+.3f} "
              f"p={b['covariate_spearman_p']:.3f} and the bands are "
              f"{'monotone' if b['covariate_is_monotone'] else 'not monotone'}")
    print(f"\n  {d['outcome_key']}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") != READ_INDEX:
        bad.append(f"read index {d.get('test_fold_read_index')}")
    if d.get("status") != "exploratory":
        bad.append(f"the read reports itself as {d.get('status')}; the plan "
                   f"declared it exploratory before any number was seen")
    if not (d.get("calibration_against_read_five") or {}).get("reproduces"):
        bad.append("the read no longer reproduces read five's published mean "
                   "and win/loss counts")
    plan = json.loads(PLAN.read_text())
    if d.get("outcome") != plan["what_will_be_written_under_each_outcome"].get(
            d.get("outcome_key")):
        bad.append("the stated outcome is not the sentence preregistered for "
                   "the outcome key it reports")
    if d.get("covariate_artifact_sha256") != hashlib.sha256(
            COV.read_bytes()).hexdigest():
        bad.append("the covariate artifact has changed since the read, so the "
                   "bands reported are not the ones the fold was cut into")
    p = d.get("per_chain_distribution") or {}
    if p.get("n_field_ahead", 0) + p.get("n_baseline_ahead", 0) + p.get(
            "n_tied", 0) != p.get("n"):
        bad.append("wins, losses and ties do not account for every chain")
    # Every band's mean must still reconstruct the overall mean, which is the
    # cheap way to catch a unit dropped from a partition.
    for name, v in (d.get("by_covariate") or {}).items():
        tot = sum(b["n"] for b in v["bands"])
        if tot != d["n_paired_units"]:
            bad.append(f"{name}: bands hold {tot} of {d['n_paired_units']} units")
        recon = sum(b["mean"] * b["n"] for b in v["bands"]) / tot
        if abs(recon - p["mean"]) > 1e-6:
            bad.append(f"{name}: band means reconstruct {recon:.6f}, not "
                       f"{p['mean']:.6f}")
    # The one claim this artifact must never make.
    if d.get("bands_excluding_zero_after_correction") and d.get(
            "status") != "exploratory":
        bad.append("a surviving band has been promoted out of exploratory")
    # A surviving band must carry its own trend context, or the number travels
    # into the manuscript without the thing that qualifies it.
    for b in d.get("bands_excluding_zero_after_correction") or []:
        if "supported_by_a_trend" not in b:
            bad.append(f"{b['covariate']} {b['band']} survives correction but "
                       f"does not record whether a trend supports it")
        elif b["supported_by_a_trend"] != (
                b["covariate_is_monotone"]
                and b["covariate"] in (d.get("trends_surviving_correction")
                                       or [])):
            bad.append(f"{b['covariate']} {b['band']}: its trend support flag "
                       f"disagrees with the trend results beside it")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
