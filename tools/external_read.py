"""Read the external validation set once, under the plan, and take whatever comes back.

This is the only confirmatory comparison in the repository. Everything else was
computed on a fold that had been read twelve times while eleven architectures were
compared on it, which is disclosed and which is why none of those numbers can carry
a claim on their own.

What makes this one different is not care, it is order: the set was built and
hashed, the plan was committed naming three co-primary comparisons and the sentence
to write under each outcome, and only then was anything scored. This tool checks
that order against the commit graph rather than asserting it, refuses to run if the
plan is editable or the set has moved, and writes the result once.

Nothing here chooses anything. The statistic, the correction, the binarisation
conventions, the seed and the number of resamples all come out of the plan. The
verdict is looked up in the plan's own table from the numbers this read produces.

Usage: PYTHONPATH=src:tools python3.12 tools/external_read.py [--check] [--rerun]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from matched_full_read import confusion, metric_of, top_q_call
from plmnn_read import _interval
from pocket_bench.metrics import residue_auc_pr
from pocket_bench.paths import ROOT

PLAN = ROOT / "results/external/PREREGISTERED_EXTERNAL.json"
SET = ROOT / "results/external/EXTERNAL_SET.json"
RULE = ROOT / "results/external/CRYPTOBENCH_RULE.json"
LABELS = ROOT / "data/external/labels"
PREDS = ROOT / "results/external/predictions"
PLMNN = ROOT / "results/baselines/PLMNN_EXTERNAL_SCORES.json"
POCKETMINER = ROOT / "data/baselines/pocketminer_external"
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
OUT = ROOT / "results/external/EXTERNAL_READ.json"

SCHEMA = "geoaudit.external_read.v1"
TIE_ATOL = 1e-12
METRICS = ("precision", "recall", "positive_class_f1", "mcc")
BASELINES = ("p2rank", "plmnn", "pocketminer")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan() -> tuple[dict, dict]:
    """The plan, and the evidence that it was fixed before this ran.

    A preregistration that could have been written afterwards is worth nothing, and
    no field inside the file can establish that it was not. The commit graph can.
    """
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "shallow clone: the commit that fixed the plan may be absent, so the "
            "ordering cannot be checked. Fetch full history and retry.")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is modified or untracked. A plan that can still "
                         f"be edited is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")

    plan = json.loads(PLAN.read_text())
    for path, want, what in ((SET, plan["the_set"]["sha256"], "external set"),
                             (RULE, plan["the_set"]["labelling_rule_sha256"],
                              "labelling rule")):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(
                f"the {what} hashes to {got[:12]} but the plan pinned "
                f"{want[:12]}; it was rebuilt after the plan was written, so this "
                f"is no longer the set the plan was written for")
    return plan, {"artifact": rel, "committed_in": sha,
                  "committed_at": _git("log", "-1", "--format=%cI", sha),
                  "subject": _git("log", "-1", "--format=%s", sha),
                  "is_ancestor_of_head": True}


def _truth() -> dict[str, set[int]]:
    out = {}
    for f in sorted(LABELS.glob("*_labels.json")):
        d = json.loads(f.read_text())
        out[f"{d['pdb_id']}_{d['chain']}"] = {int(r) for r in
                                              d["cryptic_residues"]}
    return out


def _scores(path) -> dict[str, dict]:
    return json.loads(path.read_text())["units"]


def _plmnn() -> dict[str, dict[int, float]]:
    return {u["unit_id"]: {int(k): float(v) for k, v in u["scores"].items()}
            for u in json.loads(PLMNN.read_text())["units"]}


def _pocketminer() -> dict[str, dict[int, float]]:
    out = {}
    for f in sorted(POCKETMINER.glob("*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = {int(k): float(v) for k, v in d["residue_scores"].items()}
    return out


def _auc(keys: list[int], scores: dict[int, float], pos: set[int],
         called: list[int]) -> float | None:
    """One per-unit ROC-AUC, always through the harness's own function.

    Every method's number comes through this call, so a difference between two
    methods cannot be a difference in how the metric was computed.
    """
    res = residue_auc_pr(
        [], sorted(pos), sorted(keys),
        {"residue_scores": {str(k): scores[k] for k in keys},
         "residue_positive": called})
    v = res.get("residue_auc")
    return None if v is None else float(v)


def _shared() -> dict:
    """Per unit: the residues all four methods scored, with labels and scores.

    The intersection, not any one method's own set. A paired difference taken over
    two different residue universes is not paired, and the four methods disagree
    about which residues exist: the counting field reads every ATOM residue,
    PocketMiner needs a complete backbone, and pLM-NN scores one position per
    sequence row.
    """
    truth = _truth()
    ours = _scores(PREDS / "table_field.json")
    p2 = _scores(PREDS / "p2rank.json")
    plm, pm = _plmnn(), _pocketminer()
    units, lost, skipped = {}, [], []
    for uid in sorted(ours):
        pos = truth.get(uid) or set()
        s_o = {int(k): float(v) for k, v in
               (ours[uid].get("residue_scores") or {}).items()}
        s_p = {int(k): float(v) for k, v in
               ((p2.get(uid) or {}).get("residue_scores") or {}).items()}
        s_l, s_m = plm.get(uid) or {}, pm.get(uid) or {}
        keys = sorted(set(s_o) & set(s_p) & set(s_l) & set(s_m))
        widest = max(len(s_o), len(s_p), len(s_l), len(s_m))
        if widest - len(keys):
            lost.append({"unit": uid, "n_shared": len(keys),
                         "n_table_field": len(s_o), "n_p2rank": len(s_p),
                         "n_plmnn": len(s_l), "n_pocketminer": len(s_m)})
        if not keys or not (pos & set(keys)):
            skipped.append({"unit": uid, "n_shared": len(keys),
                            "why": ("no residue shared by all four methods"
                                    if not keys else
                                    "no labelled residue survives the "
                                    "intersection, so no ROC-AUC exists")})
            continue
        units[uid] = {
            "keys": keys,
            "y": np.array([k in pos for k in keys], dtype=bool),
            "pos": pos,
            "table_field": s_o, "p2rank": s_p, "plmnn": s_l, "pocketminer": s_m,
            "table_field_called": [int(k) for k in
                                   (ours[uid].get("residue_positive") or [])],
            "p2rank_called": [int(k) for k in
                              ((p2.get(uid) or {}).get("residue_positive") or [])],
        }
    return {"units": units, "residues_not_shared": lost, "units_skipped": skipped}


def _per_unit_auc(units: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {"table_field": {}, "p2rank": {},
                                        "plmnn": {}, "pocketminer": {}}
    for uid, u in units.items():
        for m in out:
            called = u.get(f"{m}_called")
            if called is None:
                called = [k for k, s in u[m].items() if s >= 0.5]
            v = _auc(u["keys"], u[m], u["pos"], called)
            if v is not None:
                out[m][uid] = v
    return out


def _paired(a: dict[str, float], b: dict[str, float], seed: int, n_boot: int,
            corrected: float) -> dict:
    shared = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in shared])
    out = _interval(d, seed, n_boot, corrected)
    out["n_first_ahead"] = int((d > TIE_ATOL).sum())
    out["n_second_ahead"] = int((d < -TIE_ATOL).sum())
    out["n_tied"] = int((np.abs(d) <= TIE_ATOL).sum())
    out["level_first"] = round(float(np.mean([a[u] for u in shared])), 6)
    out["level_second"] = round(float(np.mean([b[u] for u in shared])), 6)
    # Reported because the plan says it stays exploratory, so that a reader can see
    # it did not become the endpoint the moment the mean failed to resolve.
    k = max(1, int(round(0.1 * len(d))))
    out["trimmed_mean_exploratory"] = (
        round(float(np.sort(d)[k:len(d) - k].mean()), 6) if len(d) > 2 * k
        else None)
    return out


def _verdict(external: dict, pinned: dict) -> str:
    """The plan's own replication table, applied rather than interpreted."""
    lo, hi = external["ci"]
    if lo <= 0.0 <= hi:
        return "unresolved_externally"
    d, (plo, phi) = external["mean"], pinned["ci"]
    if d * pinned["delta"] < 0:
        return "fails_to_replicate"
    if plo <= d <= phi:
        return "replicates"
    return "same_direction_but_outside"


def _thresholded(units: dict, q: float, seed: int, n_boot: int,
                 corrected: float) -> dict:
    """F1 and MCC against P2Rank under the two conventions the plan named.

    Both arms give the counting field the same top-q budget, because that budget
    was fixed on the training fold and is what it deploys. The arms differ only in
    how P2Rank is binarised: its own pocket assignment, or the same budget.
    """
    arms = {"as_deployed": {"ours": [], "theirs": []},
            "common_budget": {"ours": [], "theirs": []}}
    abstained = []
    for uid, u in units.items():
        y = u["y"]
        own = confusion(top_q_call(np.array([u["table_field"][k]
                                             for k in u["keys"]]), q), y)
        s_p = np.array([u["p2rank"][k] for k in u["keys"]])
        called = set(u["p2rank_called"])
        if not called:
            abstained.append({"unit": uid, "n_pockets_predicted": 0})
        arms["as_deployed"]["ours"].append(own)
        arms["as_deployed"]["theirs"].append(confusion(
            np.array([k in called for k in u["keys"]], dtype=bool), y))
        arms["common_budget"]["ours"].append(own)
        arms["common_budget"]["theirs"].append(confusion(top_q_call(s_p, q), y))
    out = {"q": q, "n_units": len(units),
           "p2rank_predicted_no_pocket_at_all": {
               "units": abstained,
               "n": len(abstained),
               "consequence": ("with no call made, precision is 0/0 and MCC's "
                               "denominator carries a zero factor, so both are "
                               "undefined and the unit leaves those two "
                               "comparisons. Recall and F1 are defined and equal "
                               "zero, so it stays in those"),
               "which_way_this_cuts": ("it removes a unit where P2Rank found "
                                       "nothing from P2Rank's precision and MCC "
                                       "averages, so both figures below flatter "
                                       "P2Rank rather than us. The F1 and recall "
                                       "figures, which keep the unit, do not"),
               "the_asymmetry_itself": ("a fixed per-chain budget always answers; "
                                        "native pocket assignment can decline to. "
                                        "That is a difference in deployment "
                                        "behaviour, not in score quality")}}
    for i, (arm, sides) in enumerate(arms.items()):
        block = {}
        for k, name in enumerate(METRICS):
            vals = {side: [metric_of(name, c) for c in cs]
                    for side, cs in sides.items()}
            block[name] = {
                side: round(float(np.mean([v for v in vs if v is not None])), 6)
                for side, vs in vals.items()}
            block[name]["n_undefined"] = {
                side: int(sum(v is None for v in vs))
                for side, vs in vals.items()}
            d = np.array([o - t for o, t in zip(vals["ours"], vals["theirs"])
                          if o is not None and t is not None])
            block[name]["paired_difference_ours_minus_theirs"] = _interval(
                d, seed + 1000 + 100 * i + k, n_boot, corrected)
        out[arm] = block
    return out


def build() -> dict:
    plan, prov = _plan()
    st = plan["statistic"]
    seed, n_boot = st["seed"], st["n_boot"]
    co_level = plan["co_primary"]["corrected_level"]
    sec_level = plan["secondary"]["corrected_level"]
    pinned = plan["what_cryptobench_said_these_numbers_would_be"]

    sh = _shared()
    units = sh["units"]
    if not units:
        raise SystemExit("no unit survived the four-way intersection")
    aucs = _per_unit_auc(units)

    co: dict[str, dict] = {}
    for i, b in enumerate(BASELINES):
        blk = _paired(aucs["table_field"], aucs[b], seed + i, n_boot, co_level)
        blk["cryptobench_predicted"] = pinned[
            f"counting_field_minus_{b}_roc_auc"]
        blk["verdict"] = _verdict(blk, blk["cryptobench_predicted"])
        co[f"table_field_minus_{b}"] = blk
    # The qualification the plan requires next to the PocketMiner result: if
    # P2Rank's margin over PocketMiner is similar to ours, the comparison is
    # measuring the two label definitions rather than ranking the methods.
    co["p2rank_minus_pocketminer_for_context"] = _paired(
        aucs["p2rank"], aucs["pocketminer"], seed + 90, n_boot, co_level)

    # The budget comes out of the frozen field, where it was written when it was
    # chosen on the training fold, rather than being restated here. The plan named
    # a figure independently; if the two ever disagreed, one of them would be a
    # number typed into a script, so the read refuses instead of picking.
    op = json.loads(FIELD.read_text())["operating_point"]
    q = float(op["q"])
    declared = [t for t in plan["secondary"]["tests"] if "top-" in t]
    if not all(f"top-{q:.0%}" in t for t in declared):
        raise SystemExit(
            f"the frozen field deploys top-{q:.0%} but the plan's secondary tests "
            f"name a different budget: {declared}")
    thresholded = _thresholded(units, q, seed, n_boot, sec_level)
    thresholded["q_came_from"] = {
        "artifact": str(FIELD.relative_to(ROOT)),
        "selected_on": op["selected_on"], "rule": op["rule"]}

    p2 = co["table_field_minus_p2rank"]
    plm = co["table_field_minus_plmnn"]
    n = len(units)
    if not any(co[f"table_field_minus_{b}"]["excludes_zero"]
               for b in BASELINES):
        headline = "underpowered"
    elif p2["excludes_zero"] and p2["mean"] > 0:
        headline = "p2rank_advantage_appears"
    elif p2["excludes_zero"] and p2["mean"] < 0:
        headline = "p2rank_advantage_reverses"
    else:
        headline = "p2rank_parity_replicates"
    plmnn_sentence = ("plmnn_deficit_replicates" if plm["excludes_zero"]
                      else "plmnn_deficit_unresolved")

    def _fill(key: str, blk: dict) -> str:
        pin = blk.get("cryptobench_predicted") or {}
        out = (plan["what_will_be_written_under_each_outcome"][key]
               .replace("{n}", str(n))
               .replace("{d}", f"{blk['mean']:+.4f}")
               .replace("{ci}", f"[{blk['ci'][0]:+.4f}, {blk['ci'][1]:+.4f}]")
               .replace("{old}", f"{pin['delta']:+.4f}" if pin else "{old}"))
        if "{" in out:
            raise SystemExit(
                f"the preregistered sentence for {key} still has an unfilled "
                f"placeholder: {out[out.index('{'):][:40]}. A sentence the read "
                f"cannot fill is one the plan and the read disagree about.")
        return out

    return {
        "schema": SCHEMA, "clinical_grade": False,
        "reads_the_external_set": True,
        "reads_cryptobench_test_fold": False,
        "status": plan["status_declared_in_advance"],
        "plan": prov,
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "set_sha256": hashlib.sha256(SET.read_bytes()).hexdigest(),
        "n_units_in_set": json.loads(SET.read_text())["n_units_with_a_cryptic_pocket"],
        "n_units_compared": n,
        "coverage": {
            "units_skipped": sh["units_skipped"],
            "residues_not_shared_by_all_four": sh["residues_not_shared"],
            "n_units_losing_residues": len(sh["residues_not_shared"]),
            "why": ("a paired difference needs one residue universe, so every "
                    "comparison runs on the residues all four methods scored"),
        },
        "levels": {m: round(float(np.mean(list(v.values()))), 6)
                   for m, v in aucs.items()},
        "co_primary": co,
        "secondary_thresholded_against_p2rank": thresholded,
        "what_the_paper_must_now_say": {
            "headline": headline,
            "headline_sentence": _fill(headline, p2),
            "plmnn": plmnn_sentence,
            "plmnn_sentence": _fill(plmnn_sentence, plm),
            "chosen_by": ("the plan's own decision table, applied to the numbers "
                          "above; the sentences were written before the read"),
        },
        "the_commitment": plan["the_commitment_that_matters"],
    }


def _numbers(d: dict) -> str:
    """A digest of everything in the artifact that is a result rather than prose.

    Its only job is to answer one question about a rerun: did any number move? A
    rerun that fixes a sentence and a rerun that changes a finding look the same in
    a shell history and completely different here.
    """
    return hashlib.sha256(json.dumps(
        {"levels": d.get("levels"), "co_primary": d.get("co_primary"),
         "secondary": d.get("secondary_thresholded_against_p2rank"),
         "n": d.get("n_units_compared")}, sort_keys=True).encode()).hexdigest()


def _report(d: dict) -> None:
    print(f"\nexternal read, declared {d['status']}: {d['n_units_compared']} of "
          f"{d['n_units_in_set']} units compared")
    print("  plan fixed in {} ({})".format(d["plan"]["committed_in"][:12],
                                           d["plan"]["committed_at"][:10]))
    lv = d["levels"]
    print("  mean per-unit ROC-AUC: " + ", ".join(
        f"{m} {lv[m]:.4f}" for m in ("table_field", "p2rank", "plmnn",
                                     "pocketminer")))
    for name, blk in d["co_primary"].items():
        pin = blk.get("cryptobench_predicted")
        line = (f"  {name}: {blk['mean']:+.4f} "
                f"[{blk['ci'][0]:+.4f}, {blk['ci'][1]:+.4f}]"
                f"{'  resolved' if blk['excludes_zero'] else '  crosses zero'}")
        if pin:
            line += (f"; CryptoBench said {pin['delta']:+.4f} -> "
                     f"{blk['verdict']}")
        print(line + f"; per-unit {blk['n_first_ahead']}/{blk['n_second_ahead']}")
    for arm in ("as_deployed", "common_budget"):
        b = d["secondary_thresholded_against_p2rank"][arm]
        f1 = b["positive_class_f1"]["paired_difference_ours_minus_theirs"]
        mc = b["mcc"]["paired_difference_ours_minus_theirs"]
        print(f"  {arm}: F1 {f1['mean']:+.4f} [{f1['ci'][0]:+.4f}, "
              f"{f1['ci'][1]:+.4f}], MCC {mc['mean']:+.4f} "
              f"[{mc['ci'][0]:+.4f}, {mc['ci'][1]:+.4f}]")
    w = d["what_the_paper_must_now_say"]
    print(f"\n  outcome: {w['headline']} / {w['plmnn']}")
    print(f"  {w['headline_sentence']}")
    print(f"  {w['plmnn_sentence']}")


def check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != d["plan_sha256"]:
        bad.append("the plan changed after the read, so it no longer governs it")
    if hashlib.sha256(SET.read_bytes()).hexdigest() != d["set_sha256"]:
        bad.append("the external set changed after the read")
    if d["status"] != "confirmatory":
        bad.append(f"status is {d['status']}")
    plan = json.loads(PLAN.read_text())
    for b in BASELINES:
        blk = d["co_primary"][f"table_field_minus_{b}"]
        if blk["verdict"] not in plan["replication_verdicts_defined_now"]:
            bad.append(f"{b}: verdict {blk['verdict']} is not one the plan defined")
    w = d["what_the_paper_must_now_say"]
    if w["headline"] not in plan["what_will_be_written_under_each_outcome"]:
        bad.append(f"the headline {w['headline']} has no preregistered sentence")
    if bad:
        for x in bad:
            print(f"FAILED: {x}")
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rerun", metavar="REASON",
                    help="overwrite an existing result. The reason is recorded in "
                         "the artifact, together with whether any number changed, "
                         "because a confirmatory read that ran twice is not one "
                         "read and the difference has to be visible")
    a = ap.parse_args()
    if a.check:
        return check()
    if OUT.is_file() and not a.rerun:
        raise SystemExit(
            f"{OUT.relative_to(ROOT)} exists. This read is confirmatory and is "
            f"meant to happen once; pass --rerun REASON if you intend to "
            f"overwrite it, and the artifact will say so.")
    before = json.loads(OUT.read_text()) if OUT.is_file() else None
    d = build()
    if a.rerun:
        d["reruns"] = (before or {}).get("reruns", []) + [{
            "reason": a.rerun,
            "at": _git("log", "-1", "--format=%cI") or None,
            "numbers_identical_to_the_previous_run": (
                _numbers(before) == _numbers(d) if before else None),
            "previous_numeric_digest": _numbers(before) if before else None,
            "why_this_is_recorded": (
                "a confirmatory read that ran more than once is no longer one "
                "read. Keeping the reason and whether any number moved in the "
                "artifact is what lets a reader tell a corrected sentence from a "
                "second look at the data"),
        }]
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
