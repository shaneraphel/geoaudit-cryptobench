#!/usr/bin/env python3
"""The seventh read: where the matched margin sits, and what it survives.

Arithmetic on per-residue scores that have been committed since the first read,
so nothing is rescored. Indexed anyway, for the reason the sixth read was: a
frozen score re-binarised at a new threshold, or summarised by a metric nobody
had computed, is a new statement about the held-out fold, and a ledger that
counted only re-runs would make an unlimited number of such statements free.

What is read is fixed by ``PREREGISTERED_MATCHED_FULL.json``: four conventions
(as deployed, a common budget, each tuned for F1, each tuned for MCC), four
metrics on each (precision, recall, positive-class F1, MCC), and three
resampling units (chain, PDB entry, sequence cluster).

Four guards stand before the arithmetic, all from the plan. The plan's commit
must be an ancestor of HEAD. The deployment-rule F1 and MCC must reproduce the
frozen bootstrap. The matched F1 delta must reproduce read six exactly, because
that number is published and this file has no licence to move it. And precision
and recall must move in the same direction under a matched budget, because at a
common calling fraction they are two views of one confusion count; if they
disagree, the alignment is wrong and no number here means what it says.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/cryptobench_apo/official_labels"
PREDS = ROOT / "results/cryptobench_official/predictions"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_FULL.json"
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
READ6 = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"
OUT = ROOT / "results/official_fold/MATCHED_FULL_READ.json"

SCHEMA = "geoaudit.matched_full_read.v1"
READ_INDEX = 7
N_BOOT = 10000
SEED = 20260725
CI = 0.95
TRIM = 0.20
REPRO_PLACES = 4
METRICS = ("precision", "recall", "positive_class_f1", "mcc")


def _resnum(x) -> int | None:
    """``pocket_bench.residue_id``'s convention, duplicated on purpose.

    A recomputation that imports the harness's parser cannot detect the
    harness's parsing bugs, which is the same reason ``recompute_from_raw``
    carries its own copy.
    """
    if isinstance(x, int):
        return x
    s, digits, negative = str(x), "", False
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


def load_truth() -> dict[str, set[int]]:
    truth = {}
    for f in sorted(LABELS.glob("*_labels.json")):
        d = json.loads(f.read_text())
        res = d.get("cryptic_residues") or d.get("binding_residues") or []
        truth[f"{d['pdb_id']}_{d['chain']}"] = {
            r for r in (_resnum(r) for r in res) if r is not None}
    return truth


def aligned(unit: dict, positives: set[int]):
    raw = unit.get("residue_scores")
    if not raw:
        return None
    by_res = {}
    for k, v in raw.items():
        r = _resnum(k)
        if r is not None:
            by_res[r] = float(v)
    if not by_res or not (positives & set(by_res)):
        return None
    keys = sorted(by_res)
    call = {r for r in (_resnum(r) for r in (unit.get("residue_positive") or []))
            if r is not None}
    return (np.array([by_res[k] for k in keys], dtype=np.float64),
            np.array([k in call for k in keys], dtype=bool),
            np.array([k in positives for k in keys], dtype=bool))


def top_q_call(s: np.ndarray, q: float) -> np.ndarray:
    n = len(s)
    k = max(1, int(round(q * n)))
    out = np.zeros(n, dtype=bool)
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def confusion(call: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    tp = int((call & truth).sum())
    fp = int((call & ~truth).sum())
    fn = int((~call & truth).sum())
    tn = int((~call & ~truth).sum())
    return tp, fp, tn, fn


def metric_of(name: str, c: tuple[int, int, int, int]) -> float | None:
    """One metric on one chain, or None where it is genuinely undefined.

    None is not zero. A chain on which P2Rank's pocket assignment calls no
    residue has no precision -- the quantity asks what fraction of the calls
    were right and there were no calls -- and imputing zero there would score
    a method for abstaining as though it had been wrong.
    """
    tp, fp, tn, fn = c
    if name == "precision":
        return (tp / (tp + fp)) if (tp + fp) else None
    if name == "recall":
        return (tp / (tp + fn)) if (tp + fn) else None
    if name == "positive_class_f1":
        d = 2 * tp + fp + fn
        return (2 * tp / d) if d else None
    if name == "mcc":
        d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return ((tp * tn - fp * fn) / d) if d > 0 else None
    raise KeyError(name)


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _trimmed(v: list[float], frac: float = TRIM) -> float:
    s = sorted(v)
    k = int(len(s) * frac)
    core = s[k:len(s) - k] or s
    return sum(core) / len(core)


def paired(a: list[float | None], b: list[float | None], stat,
           groups: list[int], n_groups: int) -> dict:
    """Bootstrap a paired difference, resampling whole groups.

    ``groups[i]`` is the resampling group of unit ``i``; drawing a group takes
    all of its units. With one unit per group this is the chain bootstrap every
    published interval used. The pairing is on the units where both arms have a
    value, because differencing two means taken over different subsets would
    fold "which chains were scorable" into a quantity read as "which method is
    better".
    """
    shared = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if not shared:
        return {"delta_point": None, "delta_ci_low": None, "delta_ci_high": None,
                "p_two_sided_bootstrap": None, "crosses_zero": None,
                "n_paired_units": 0, "n_paired_groups": 0}
    by_group: dict[int, list[int]] = {}
    for i in shared:
        by_group.setdefault(groups[i], []).append(i)
    keys = sorted(by_group)
    rng = random.Random(SEED)
    deltas = []
    for _ in range(N_BOOT):
        pick: list[int] = []
        for _ in range(len(keys)):
            pick.extend(by_group[keys[rng.randrange(len(keys))]])
        deltas.append(stat([a[i] for i in pick]) - stat([b[i] for i in pick]))
    deltas.sort()
    lo = deltas[int((1 - CI) / 2 * N_BOOT)]
    hi = deltas[min(N_BOOT - 1, int((1 + CI) / 2 * N_BOOT))]
    point = stat([a[i] for i in shared]) - stat([b[i] for i in shared])
    side = min(sum(1 for d in deltas if d <= 0), sum(1 for d in deltas if d >= 0))
    return {"delta_point": round(point, 6),
            "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6),
            "ci_width": round(hi - lo, 6),
            "p_two_sided_bootstrap": round(min(1.0, 2 * side / N_BOOT), 6),
            "crosses_zero": bool(lo <= 0.0 <= hi),
            "n_paired_units": len(shared),
            "n_paired_groups": len(keys),
            "of_groups_available": n_groups}


def plan_precedes_this_read() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    rel = str(PLAN.relative_to(ROOT))
    if git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the ordering of the plan against this "
            "read cannot be checked. Fetch the full history and run it again")
    if git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is modified or untracked. A plan that can still be edited "
            "is not one, so this read is refused")
    sha = git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused")
    head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, head],
                      cwd=ROOT, capture_output=True).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD {head[:12]}")
    return {"artifact": rel, "plan_commit": sha,
            "committed_at": git("log", "-1", "--format=%cI", sha),
            "subject": git("log", "-1", "--format=%s", sha),
            "head_at_read": head,
            "plan_is_an_ancestor_of_the_read": True,
            "checked_with": "git log -1 -- <plan>, then merge-base "
                            "--is-ancestor against HEAD"}


def _pairs():
    truth = load_truth()
    tf = json.loads((PREDS / "table_field.json").read_text())["units"]
    p2 = json.loads((PREDS / "p2rank.json").read_text())["units"]
    units, data = [], {}
    for u in sorted(set(tf) & set(p2)):
        pos = truth.get(u) or set()
        a, b = aligned(tf[u], pos), aligned(p2[u], pos)
        if a is None or b is None:
            continue
        units.append(u)
        data[u] = (a, b)
    return units, data


def _groupings(units: list[str]) -> dict[str, dict]:
    """Chain, PDB entry and sequence cluster, as integer group labels."""
    e = json.loads(MANIFEST.read_text())["entries"]
    pdb_of = {f"{x['pdb']}_{x['chain']}": x["pdb"] for x in e}
    clu_of = {f"{x['pdb']}_{x['chain']}": x["cluster_id"] for x in e}
    missing = [u for u in units if u not in pdb_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} scored units are absent from the official "
            f"manifest ({missing[:3]}), so they cannot be assigned a cluster "
            f"and a cluster bootstrap over them would be undefined")
    out = {}
    for name, mapping in (("chain", {u: u for u in units}),
                          ("pdb_entry", pdb_of), ("uniprot_cluster", clu_of)):
        keys = sorted({mapping[u] for u in units})
        idx = {k: i for i, k in enumerate(keys)}
        out[name] = {"groups": [idx[mapping[u]] for u in units],
                     "n_groups": len(keys)}
    return out


def _calls(data, units, convention: dict) -> dict[str, list[np.ndarray]]:
    """The two arms' positive calls under one convention."""
    cid = convention["id"]
    ours, theirs = [], []
    for u in units:
        (s_a, native_a, _), (s_b, native_b, _) = data[u]
        if cid == "D1_as_deployed":
            ours.append(native_a)
            theirs.append(native_b)
        else:
            ours.append(top_q_call(s_a, convention["q_ours"]))
            theirs.append(top_q_call(s_b, convention["q_p2rank"]))
    return {"table_field": ours, "p2rank": theirs}


def _per_unit(calls, data, units, which: int) -> dict[str, list[float | None]]:
    out = {m: [] for m in METRICS}
    for i, u in enumerate(units):
        c = confusion(calls[i], data[u][which][2])
        for m in METRICS:
            out[m].append(metric_of(m, c))
    return out


def build() -> dict:
    order = plan_precedes_this_read()
    plan = json.loads(PLAN.read_text())
    shipped_q = plan["conventions_to_be_read"][1]["q"]
    d4 = next(c for c in plan["conventions_to_be_read"]
              if c["id"] == "D4_each_tuned_for_mcc")
    d3 = next(c for c in plan["conventions_to_be_read"]
              if c["id"] == "D3_each_tuned_for_f1")
    conventions = [
        {"id": "D1_as_deployed", "q_ours": None, "q_p2rank": None},
        {"id": "D2_common_budget", "q_ours": shipped_q, "q_p2rank": shipped_q},
        {"id": "D3_each_tuned_for_f1", "q_ours": d3["q_table_field"],
         "q_p2rank": d3["q_p2rank"]},
        {"id": "D4_each_tuned_for_mcc", "q_ours": d4["q_table_field"],
         "q_p2rank": d4["q_p2rank"]},
    ]

    units, data = _pairs()
    grouping = _groupings(units)

    rows = {}
    for conv in conventions:
        calls = _calls(data, units, conv)
        ours = _per_unit(calls["table_field"], data, units, 0)
        theirs = _per_unit(calls["p2rank"], data, units, 1)
        entry = {
            "q_ours": conv["q_ours"], "q_p2rank": conv["q_p2rank"],
            "table_field": {m: round(_mean([v for v in ours[m] if v is not None]), 6)
                            for m in METRICS},
            "p2rank": {m: round(_mean([v for v in theirs[m] if v is not None]), 6)
                       for m in METRICS},
            "n_units_scored": {
                m: {"table_field": sum(1 for v in ours[m] if v is not None),
                    "p2rank": sum(1 for v in theirs[m] if v is not None)}
                for m in METRICS},
            "n_residues_called": {
                "table_field": int(sum(int(c.sum()) for c in calls["table_field"])),
                "p2rank": int(sum(int(c.sum()) for c in calls["p2rank"]))},
            "paired": {},
        }
        for m in METRICS:
            entry["paired"][m] = {
                name: paired(ours[m], theirs[m], _mean,
                             g["groups"], g["n_groups"])
                for name, g in grouping.items()}
            entry["paired"][m]["chain_trimmed_mean"] = paired(
                ours[m], theirs[m], _trimmed,
                grouping["chain"]["groups"], grouping["chain"]["n_groups"])
        rows[conv["id"]] = entry

    # Guard: the deployment rules must give back the frozen bootstrap.
    frozen = json.loads(BOOT.read_text())["metrics"]
    repro = {}
    for met, key in (("residue_f1", "positive_class_f1"), ("residue_mcc", "mcc")):
        for arm in ("table_field", "p2rank"):
            ref = frozen[met]["per_method"][arm]["point"]
            mine = rows["D1_as_deployed"][arm][key]
            repro[f"{met}/{arm}"] = {
                "recomputed": mine, "committed": round(ref, 6),
                "agrees": round(mine, REPRO_PLACES) == round(ref, REPRO_PLACES)}
    if not all(r["agrees"] for r in repro.values()):
        raise SystemExit(
            f"the deployment-rule recomputation does not reproduce the frozen "
            f"bootstrap: {json.dumps(repro)}. This file has no standing to "
            f"report matched numbers it cannot calibrate")

    # Guard: the matched F1 delta is published and may not move here.
    r6 = json.loads(READ6.read_text())["matched"]["A_common_q"]["primary"]
    mine6 = rows["D2_common_budget"]["paired"]["positive_class_f1"]["chain"]
    f1_agrees = abs(mine6["delta_point"] - r6["delta_point"]) < 1e-6
    if not f1_agrees:
        raise SystemExit(
            f"the matched F1 delta here is {mine6['delta_point']} but read six "
            f"published {r6['delta_point']}. One of the two is wrong and this "
            f"read is not entitled to decide which")

    # Guard: at a matched budget, precision and recall are one confusion count
    # seen twice, so their deltas cannot disagree in sign.
    sign_check = {}
    for cid in ("D2_common_budget", "D3_each_tuned_for_f1"):
        p = rows[cid]["paired"]["precision"]["chain"]["delta_point"]
        r = rows[cid]["paired"]["recall"]["chain"]["delta_point"]
        ok = (p == 0 and r == 0) or (p > 0) == (r > 0)
        sign_check[cid] = {"precision_delta": p, "recall_delta": r,
                           "same_sign": ok}
        if not ok:
            raise SystemExit(
                f"under {cid} the precision delta is {p:+.6f} and the recall "
                f"delta is {r:+.6f}. At a matched calling budget these are two "
                f"views of one confusion count and cannot disagree in sign; "
                f"the residue alignment is wrong")

    governing = rows["D2_common_budget"]["paired"]
    verdicts = {}
    for m in ("positive_class_f1", "mcc"):
        c = governing[m]["chain"]
        if c["delta_point"] is None:
            verdicts[m] = "undefined"
        elif not c["crosses_zero"] and c["delta_point"] > 0:
            verdicts[m] = "excludes_zero"
        elif c["delta_point"] > 0:
            verdicts[m] = "positive_but_unresolved"
        else:
            verdicts[m] = "zero_or_reversed"

    said = plan["what_will_be_written_under_each_outcome"]
    f1v, mccv = verdicts["positive_class_f1"], verdicts["mcc"]
    if "zero_or_reversed" in (f1v, mccv):
        key = "either_reverses"
    elif f1v == "excludes_zero" and mccv == "excludes_zero":
        key = "both_survive"
    elif mccv == "excludes_zero":
        key = "f1_unresolved_mcc_survives"
    else:
        key = "f1_and_mcc_both_unresolved"

    # Multiplicity. Four metrics on four conventions under three resamplings is
    # 48 intervals, and the plan attached a decision rule to two of them. Any
    # of the other 46 that happens to exclude zero is an observation and not a
    # test, and the only way to keep it from being read as one is to say so
    # here, with the arithmetic, rather than in a caveat further downstream.
    governed = ("positive_class_f1", "mcc")
    n_metrics = len(METRICS)
    multiplicity = {
        "intervals_examined": len(rows) * len(METRICS) * 3,
        "intervals_a_decision_rule_governs": len(governed),
        "governed": list(governed),
        "bonferroni_alpha_over_the_four_metrics_of_one_convention": round(
            0.05 / n_metrics, 4),
        "under_the_governing_convention": {
            m: {
                "p_nominal": governing[m]["chain"]["p_two_sided_bootstrap"],
                "excludes_zero_nominally":
                    governing[m]["chain"]["crosses_zero"] is False,
                "survives_bonferroni_over_the_four_metrics": (
                    governing[m]["chain"]["p_two_sided_bootstrap"] is not None
                    and governing[m]["chain"]["p_two_sided_bootstrap"]
                    < 0.05 / n_metrics),
                "a_decision_rule_governs_it": m in governed,
            } for m in METRICS},
        "what_this_means_for_the_ungoverned_intervals": (
            "an ungoverned interval that excludes zero is reported as an "
            "observation about this fold. It is not evidence at the strength a "
            "preregistered test would carry, because the metric that excluded "
            "zero was chosen for emphasis after the numbers existed, and "
            "because it is one of several intervals computed on the same "
            "confusion counts"),
    }

    widths = {}
    for m in METRICS:
        base = governing[m]["chain"]["ci_width"]
        widths[m] = {
            name: {"width": governing[m][name]["ci_width"],
                   "ratio_to_chain": (round(governing[m][name]["ci_width"] / base, 4)
                                      if base else None)}
            for name in ("chain", "pdb_entry", "uniprot_cluster")}

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "method": "table_field",
        "baseline": "p2rank",
        "question": plan["question"],
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "re-binarising a frozen score at a new threshold, or summarising "
            "it with a metric nobody had computed, is a new statement about "
            "the held-out fold"),
        "selection_provenance": str(PLAN.relative_to(ROOT)),
        "ordering": order,
        "n_units": len(units),
        "resampling_units": {k: v["n_groups"] for k, v in grouping.items()},
        "reproduces_the_frozen_bootstrap": repro,
        "reproduces_read_six_matched_f1": {
            "recomputed": mine6["delta_point"],
            "read_six": r6["delta_point"],
            "agrees": f1_agrees,
            "status": "a calibration, not a finding; the plan says so and the "
                      "paper does not cite it as a result"},
        "precision_recall_sign_agreement": sign_check,
        "conventions": rows,
        "governing_convention": "D2_common_budget",
        "governing_resampling_unit": "chain",
        "verdicts": verdicts,
        "multiplicity": multiplicity,
        "where_the_deployment_rule_margin_came_from": {
            "residues_called_under_the_deployment_rules": rows[
                "D1_as_deployed"]["n_residues_called"],
            "p2rank_calls_this_many_times_as_many": round(
                rows["D1_as_deployed"]["n_residues_called"]["p2rank"]
                / rows["D1_as_deployed"]["n_residues_called"]["table_field"], 3),
            "precision_delta_as_deployed": rows["D1_as_deployed"]["paired"][
                "precision"]["chain"]["delta_point"],
            "recall_delta_as_deployed": rows["D1_as_deployed"]["paired"][
                "recall"]["chain"]["delta_point"],
            "reading": (
                "under its own pocket assignment P2Rank calls far more "
                "residues than our rule does, which buys it recall and costs "
                "it precision. Both of those differences have intervals "
                "excluding zero and they point opposite ways, so the published "
                "F1 and MCC margins were summarising a trade between two "
                "calling budgets rather than a uniformly better detector"),
        },
        "ci_width_by_resampling_unit": widths,
        "clustering_changes_nothing_materially": all(
            w["uniprot_cluster"]["ratio_to_chain"] is None
            or w["uniprot_cluster"]["ratio_to_chain"] < 1.10
            for w in widths.values()),
        "forecast_vs_outcome": {
            "f1": {"forecast": plan["forecast"]["f1"]["predicted_matched_delta"],
                   "outcome": governing["positive_class_f1"]["chain"]["delta_point"]},
            "mcc": {"forecast": plan["forecast"]["mcc"]["predicted_matched_delta"],
                    "outcome": governing["mcc"]["chain"]["delta_point"],
                    "predicted_to_cross_zero":
                        plan["forecast"]["mcc"]["predicted_to_cross_zero"],
                    "does_cross_zero": governing["mcc"]["chain"]["crosses_zero"]},
        },
        "outcome_key": key,
        "conclusion": said[key],
        "decision_rules": plan["decision_rules"],
    }


def _report(d: dict) -> None:
    print(f"\nmatched, full metric set, {d['n_units']} held-out units "
          f"(read {d['test_fold_read_index']})")
    for cid, e in d["conventions"].items():
        q = ("native" if e["q_ours"] is None
             else f"{e['q_ours']:.2f}/{e['q_p2rank']:.2f}")
        print(f"\n  {cid}  q {q}   called "
              f"{e['n_residues_called']['table_field']} vs "
              f"{e['n_residues_called']['p2rank']}")
        for m in METRICS:
            p = e["paired"][m]["chain"]
            if p["delta_point"] is None:
                print(f"    {m:<18} undefined")
                continue
            star = "" if p["crosses_zero"] else "  *"
            print(f"    {m:<18} ours {e['table_field'][m]:.4f}  "
                  f"P2Rank {e['p2rank'][m]:.4f}  "
                  f"delta {p['delta_point']:+.4f} "
                  f"[{p['delta_ci_low']:+.4f}, {p['delta_ci_high']:+.4f}] "
                  f"p={p['p_two_sided_bootstrap']:.3f} n={p['n_paired_units']}{star}")
    print("\n  CI width by resampling unit (governing convention):")
    for m, w in d["ci_width_by_resampling_unit"].items():
        print(f"    {m:<18} chain {w['chain']['width']:.4f}  "
              f"pdb {w['pdb_entry']['width']:.4f} "
              f"(x{w['pdb_entry']['ratio_to_chain']})  "
              f"cluster {w['uniprot_cluster']['width']:.4f} "
              f"(x{w['uniprot_cluster']['ratio_to_chain']})")
    f = d["forecast_vs_outcome"]
    print(f"\n  forecast F1  {f['f1']['forecast']:+.4f} -> "
          f"{f['f1']['outcome']:+.4f}")
    print(f"  forecast MCC {f['mcc']['forecast']:+.4f} -> "
          f"{f['mcc']['outcome']:+.4f}   crosses zero: "
          f"predicted {f['mcc']['predicted_to_cross_zero']}, "
          f"observed {f['mcc']['does_cross_zero']}")
    mp = d["multiplicity"]
    print(f"\n  {mp['intervals_examined']} intervals examined, "
          f"{mp['intervals_a_decision_rule_governs']} governed by a rule; "
          f"Bonferroni alpha {mp['bonferroni_alpha_over_the_four_metrics_of_one_convention']}")
    for m, v in mp["under_the_governing_convention"].items():
        if v["excludes_zero_nominally"]:
            print(f"    {m:<18} p={v['p_nominal']:.3f} excludes zero; "
                  f"survives Bonferroni: "
                  f"{v['survives_bonferroni_over_the_four_metrics']}; "
                  f"governed by a rule: {v['a_decision_rule_governs_it']}")
    print(f"  verdicts: {d['verdicts']}")
    print(f"  outcome: {d['outcome_key']}")
    print(f"\n  {d['conclusion']}")


def calibrate() -> int:
    """Recompute only what the frozen bootstrap already published."""
    units, data = _pairs()
    calls = _calls(data, units, {"id": "D1_as_deployed"})
    ours = _per_unit(calls["table_field"], data, units, 0)
    theirs = _per_unit(calls["p2rank"], data, units, 1)
    frozen = json.loads(BOOT.read_text())["metrics"]
    ok = True
    print(f"{len(units)} units aligned")
    for met, key in (("residue_f1", "positive_class_f1"), ("residue_mcc", "mcc")):
        for arm, vals in (("table_field", ours), ("p2rank", theirs)):
            got = _mean([v for v in vals[key] if v is not None])
            ref = frozen[met]["per_method"][arm]["point"]
            agrees = round(got, REPRO_PLACES) == round(ref, REPRO_PLACES)
            ok &= agrees
            print(f"  {met:<12} {arm:<12} recomputed {got:.6f}  "
                  f"committed {ref:.6f}  {'ok' if agrees else 'MISMATCH'}")
    print("no matched threshold was applied and no new statement about the "
          "held-out fold was produced")
    return 0 if ok else 1


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("test_fold_read_index") != READ_INDEX:
        print(f"FAILED: read index {d.get('test_fold_read_index')}")
        return 1
    if not d["ordering"]["plan_is_an_ancestor_of_the_read"]:
        print("FAILED: the plan is not recorded as preceding the read")
        return 1
    if not all(r["agrees"] for r in d["reproduces_the_frozen_bootstrap"].values()):
        print("FAILED: the deployment rules do not reproduce the frozen bootstrap")
        return 1
    if not d["reproduces_read_six_matched_f1"]["agrees"]:
        print("FAILED: the matched F1 delta disagrees with read six")
        return 1
    for cid, s in d["precision_recall_sign_agreement"].items():
        if not s["same_sign"]:
            print(f"FAILED: {cid} has precision and recall deltas of opposite sign")
            return 1
    plan = json.loads(PLAN.read_text())
    if d["conclusion"] != plan["what_will_be_written_under_each_outcome"][
            d["outcome_key"]]:
        print("FAILED: the conclusion is not the sentence preregistered for "
              "this outcome")
        return 1
    for cid, e in d["conventions"].items():
        for m, byunit in e["paired"].items():
            for name, p in byunit.items():
                if p["delta_point"] is None:
                    continue
                if p["crosses_zero"] != (p["delta_ci_low"] <= 0.0
                                         <= p["delta_ci_high"]):
                    print(f"FAILED: {cid}/{m}/{name} mislabels whether its "
                          f"interval crosses zero")
                    return 1
    # The verdict on each governed metric must follow from its own interval.
    gov = d["conventions"][d["governing_convention"]]["paired"]
    for m, v in d["verdicts"].items():
        p = gov[m][d["governing_resampling_unit"]]
        want = ("undefined" if p["delta_point"] is None
                else "excludes_zero" if (not p["crosses_zero"]
                                         and p["delta_point"] > 0)
                else "positive_but_unresolved" if p["delta_point"] > 0
                else "zero_or_reversed")
        if v != want:
            print(f"FAILED: {m} is recorded as {v} but its interval says {want}")
            return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="audit the committed read artifact")
    ap.add_argument("--calibrate", action="store_true",
                    help="recompute only the already-published deployment-rule "
                         "numbers, to test the plumbing without making a new "
                         "statement about the held-out fold")
    a = ap.parse_args(argv)
    if a.check:
        return _check()
    if a.calibrate:
        return calibrate()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
