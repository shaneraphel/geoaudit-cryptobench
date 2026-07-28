"""The sixth read: is the F1 margin a property of the scores or the threshold?

Both methods' per-residue scores for the held-out fold have been committed since
the first read, so this is arithmetic on numbers already in the repository rather
than new inference. It is indexed anyway. Re-binarising a frozen score at a new
threshold is a new statement about the held-out fold, and a ledger that only
counted re-runs would let an unlimited number of such statements be made for
free -- which is precisely the loophole a fold-access ledger exists to close.

What is read, and nothing else, is fixed by
``PREREGISTERED_MATCHED_OPERATING_POINT.json``: rule A puts both methods at our
shipped q, rule B puts each at the q a single grid search chose for it on the
training receptors, and rule C reports P2Rank's whole F1-against-q curve so that
the best threshold any oracle could hand it is visible rather than hidden. The
statistic is the one the published F1 headline uses, so the matched delta and
the native-call delta are the same kind of number.

Two guards stand before the arithmetic. The preregistration's commit must be an
ancestor of HEAD, checked with git rather than asserted in prose. And
re-binarising each method at its own native call must reproduce the committed
per-method F1; if this file cannot recover the published numbers it has no
standing to report new ones, and it exits instead.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/cryptobench_apo/official_labels"
PREDS = ROOT / "results/cryptobench_official/predictions"
PREREG = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_OPERATING_POINT.json"
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
OUT = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"

SCHEMA = "geoaudit.matched_operating_point_read.v1"
READ_INDEX = 6
N_BOOT = 10000
SEED = 20260725
CI = 0.95
TRIM = 0.20
Q_GRID = [round(q, 2) for q in np.arange(0.02, 0.41, 0.01)]
# The published per-method F1 is stored to full float precision; four decimals
# is the precision the paper quotes and the precision at which a faithful
# reimplementation must agree.
REPRO_PLACES = 4


def _resnum(x) -> int | None:
    """The residue-number convention of ``pocket_bench.residue_id``.

    Duplicated rather than imported for the same reason ``recompute_from_raw``
    duplicates it: a recomputation that shares the harness's parser cannot
    detect the harness's parsing bugs.
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


def load_unit(method: str) -> dict[str, dict]:
    return json.loads((PREDS / f"{method}.json").read_text())["units"]


def aligned(unit: dict, positives: set[int]):
    """Scores, native call and truth over one unit's residues, or None.

    The alignment is the scorer's: residue numbers ascending, a residue the
    method did not mention scoring zero, the native call restricted to the
    universe. A unit with no positives inside the universe is not scorable and
    is dropped, which is the rule the published run used.
    """
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
    """Our shipped rule, applied to whichever method's scores are passed."""
    n = len(s)
    k = max(1, int(round(q * n)))
    out = np.zeros(n, dtype=bool)
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def f1_of(call: np.ndarray, truth: np.ndarray) -> float:
    tp = int((call & truth).sum())
    fp = int((call & ~truth).sum())
    fn = int((~call & truth).sum())
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _trimmed(v, frac=TRIM):
    s = sorted(v)
    k = int(len(s) * frac)
    core = s[k:len(s) - k] or s
    return sum(core) / len(core)


def paired(a: list[float], b: list[float], stat) -> dict:
    """Bootstrap the paired difference of a summary, resampling units."""
    n = len(a)
    rng = random.Random(SEED)
    deltas = []
    for _ in range(N_BOOT):
        pick = [rng.randrange(n) for _ in range(n)]
        deltas.append(stat([a[i] for i in pick]) - stat([b[i] for i in pick]))
    deltas.sort()
    lo = deltas[int((1 - CI) / 2 * N_BOOT)]
    hi = deltas[min(N_BOOT - 1, int((1 + CI) / 2 * N_BOOT))]
    point = stat(a) - stat(b)
    # Two-sided bootstrap p: the mass on the far side of zero, doubled.
    side = min(sum(1 for d in deltas if d <= 0), sum(1 for d in deltas if d >= 0))
    return {"delta_point": round(point, 6),
            "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6),
            "p_two_sided_bootstrap": round(min(1.0, 2 * side / N_BOOT), 6),
            "crosses_zero": bool(lo <= 0.0 <= hi),
            "n_paired_units": n}


def preregistration_precedes_this_read() -> dict:
    """Refuse to read the fold unless the plan is already in the history.

    The check is the one ``preregistered_read`` uses, for the same reason: the
    commit that last touched the plan is what dates it, and a plan still
    editable in the working tree is not a plan. Trusting a commit hash the
    plan writes about itself would let a rewritten plan carry an old date.
    """
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    rel = str(PREREG.relative_to(ROOT))
    if git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the ordering of the preregistration "
            "against this read cannot be checked. Fetch the full history "
            "(actions/checkout with fetch-depth: 0) and run it again")
    if git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is modified or untracked. A plan that can still be edited "
            "is not one, so this read is refused")
    sha = git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(
            f"{rel} has no commit. The matched rules have to be in the history "
            "before the fold is read under them; this read is refused")
    head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, head],
                      cwd=ROOT, capture_output=True).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD {head[:12]}")
    return {"artifact": rel,
            "preregistration_commit": sha,
            "committed_at": git("log", "-1", "--format=%cI", sha),
            "subject": git("log", "-1", "--format=%s", sha),
            "clean_in_working_tree": True,
            "head_at_read": head,
            "preregistration_is_an_ancestor_of_the_read": True,
            "checked_with": "git log -1 -- <plan>, then merge-base "
                            "--is-ancestor against HEAD"}


def _reproduces_published(native: dict[str, float]) -> dict:
    """Recover the committed per-method F1 before claiming anything new."""
    want = json.loads(BOOT.read_text())["metrics"]["residue_f1"]["per_method"]
    rows = {}
    for m, mine in native.items():
        ref = want[m]["point"]
        rows[m] = {"recomputed": round(mine, 6), "committed": round(ref, 6),
                   "agrees_to_places": REPRO_PLACES,
                   "agrees": round(mine, REPRO_PLACES) == round(ref, REPRO_PLACES)}
    return rows


def _pairs():
    """The units both methods scored, aligned to the labels."""
    truth = load_truth()
    tf, p2 = load_unit("table_field"), load_unit("p2rank")
    units, data = [], {}
    for u in sorted(set(tf) & set(p2)):
        pos = truth.get(u) or set()
        a, b = aligned(tf[u], pos), aligned(p2[u], pos)
        if a is None or b is None:
            continue
        units.append(u)
        data[u] = (a, b)
    return units, data


def _arm_factory(units, data):
    def arm(which: int, mode: str, q: float | None = None) -> list[float]:
        out = []
        for u in units:
            s, native, t = data[u][which]
            call = native if mode == "native" else top_q_call(s, q)
            out.append(f1_of(call, t))
        return out
    return arm


def calibrate() -> int:
    """Exercise the whole path, but recompute only what is already published.

    Wiring a recomputation of the held-out fold together is fiddly -- residue
    conventions, dropped units, tie-breaks -- and getting it wrong is the most
    likely way for this analysis to be silently false. Debugging it against the
    matched deltas would mean reading the fold repeatedly before the plan was
    even committed. So this mode computes the two native-call F1 values, which
    the first read already published, and refuses to compute anything new.
    """
    units, data = _pairs()
    arm = _arm_factory(units, data)
    repro = _reproduces_published({"table_field": _mean(arm(0, "native")),
                                   "p2rank": _mean(arm(1, "native"))})
    print(f"{len(units)} units aligned")
    for m, r in repro.items():
        flag = "ok" if r["agrees"] else "MISMATCH"
        print(f"  {m:<14} recomputed {r['recomputed']:.6f}  "
              f"committed {r['committed']:.6f}  {flag}")
    print("no matched threshold was applied and no new statement about the "
          "held-out fold was produced")
    return 0 if all(r["agrees"] for r in repro.values()) else 1


def build() -> dict:
    order = preregistration_precedes_this_read()
    plan = json.loads(PREREG.read_text())
    q_ours = float(plan["our_rule_is_unchanged"]["q"])
    q_p2 = float(next(r for r in plan["rules_to_be_read"]
                      if r["id"] == "B_each_tuned_on_train")["q_p2rank"])

    units, data = _pairs()
    arm = _arm_factory(units, data)

    ours_native = arm(0, "native")
    p2_native = arm(1, "native")
    repro = _reproduces_published({"table_field": _mean(ours_native),
                                   "p2rank": _mean(p2_native)})
    if not all(r["agrees"] for r in repro.values()):
        raise SystemExit(
            "re-binarising at the native calls does not reproduce the "
            f"committed per-method F1: {json.dumps(repro)}. This file has no "
            "standing to report matched numbers it cannot calibrate")

    ours_q = arm(0, "topq", q_ours)
    # Our native call is already the per-chain top-q at this q, so recomputing
    # it here must land on the same residues. A disagreement would mean the
    # harness and this file order or align residues differently, and every
    # matched number below would inherit that difference silently.
    same = sum(1 for u in units
               if np.array_equal(data[u][0][1],
                                 top_q_call(data[u][0][0], q_ours)))
    drift = abs(_mean(ours_q) - _mean(ours_native))
    our_rule_recovers_our_call = {
        "units_where_the_recomputed_call_is_identical": same,
        "of": len(units),
        "identical_everywhere": same == len(units),
        "f1_drift_from_recomputing_it": round(drift, 8),
        "why_this_must_hold": (
            "our shipped positive call is the per-chain top-q at this same q, "
            "so a recomputation that disagreed would be ordering tied scores "
            "differently from the harness"
        ),
        "tie_break": "stable sort of the negated score over residue numbers "
                     "ascending, applied identically to both methods",
    }
    # A handful of tied scores ordered differently would be harmless; a shift
    # in our own F1 would not be, because the matched delta would then mix a
    # change of threshold with a change of our own arm.
    if drift > 1e-6:
        raise SystemExit(
            f"recomputing our own top-q call moves our F1 by {drift:.3g} "
            f"(identical on {same} of {len(units)} units). The matched delta "
            "would confound the threshold with our own arm")

    rows = {
        "A_common_q": {
            "q_ours": q_ours, "q_p2rank": q_ours,
            "table_field_f1": round(_mean(ours_q), 6),
            "p2rank_f1": round(_mean(arm(1, "topq", q_ours)), 6),
            "primary": paired(ours_q, arm(1, "topq", q_ours), _mean),
            "secondary_trimmed_mean": paired(ours_q, arm(1, "topq", q_ours),
                                             _trimmed),
        },
        "B_each_tuned_on_train": {
            "q_ours": q_ours, "q_p2rank": q_p2,
            "table_field_f1": round(_mean(ours_q), 6),
            "p2rank_f1": round(_mean(arm(1, "topq", q_p2)), 6),
            "primary": paired(ours_q, arm(1, "topq", q_p2), _mean),
            "secondary_trimmed_mean": paired(ours_q, arm(1, "topq", q_p2),
                                             _trimmed),
        },
    }

    # Rule C: the whole curve, both arms, so the oracle threshold is legible.
    curve = []
    for q in Q_GRID:
        curve.append({"q": q,
                      "table_field_f1": round(_mean(arm(0, "topq", q)), 6),
                      "p2rank_f1": round(_mean(arm(1, "topq", q)), 6)})
    p2_oracle = max(curve, key=lambda r: r["p2rank_f1"])
    ours_oracle = max(curve, key=lambda r: r["table_field_f1"])
    oracle_gap = paired(arm(0, "topq", q_ours),
                        arm(1, "topq", p2_oracle["q"]), _mean)

    published = json.loads(BOOT.read_text())["metrics"]["residue_f1"][
        "paired_vs_baseline"]["table_field"]
    survives = {k: not v["primary"]["crosses_zero"] for k, v in rows.items()}
    governing = rows["B_each_tuned_on_train"]

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
            "re-binarising a frozen score at a new threshold is a new "
            "statement about the held-out fold. Counting only re-runs would "
            "make an unlimited number of such statements free"
        ),
        "selection_provenance": str(PREREG.relative_to(ROOT)),
        "ordering": order,
        "n_units": len(units),
        "reproduces_the_published_numbers": repro,
        "our_rule_recovers_our_own_call": our_rule_recovers_our_call,
        "native_call_reference": {
            "table_field_f1": round(_mean(ours_native), 6),
            "p2rank_f1": round(_mean(p2_native), 6),
            "published_delta": published["delta_point"],
            "published_ci": [published["delta_ci_low"],
                             published["delta_ci_high"]],
        },
        "matched": rows,
        "f1_against_q": curve,
        "oracle": {
            "p2rank_best_q_on_the_held_out_fold": p2_oracle["q"],
            "p2rank_f1_at_its_oracle_q": p2_oracle["p2rank_f1"],
            "table_field_best_q_on_the_held_out_fold": ours_oracle["q"],
            "table_field_f1_at_its_oracle_q": ours_oracle["table_field_f1"],
            "our_shipped_q_vs_p2rank_oracle_q": oracle_gap,
            "what_this_is": (
                "our method at the q it shipped, against P2Rank at the best q "
                "the held-out fold admits. P2Rank could not have known that q; "
                "the comparison is an upper bound on what any per-chain "
                "threshold could give it, not a score P2Rank achieved"
            ),
            "our_shipped_q_is_within_of_our_oracle": round(
                ours_oracle["table_field_f1"] - _mean(ours_q), 6),
        },
        "forecast_vs_outcome": {
            "forecast": plan["forecast"][
                "expected_matched_delta_if_the_gain_transfers"],
            "outcome_under_the_governing_rule":
                governing["primary"]["delta_point"],
            "forecast_error": round(
                governing["primary"]["delta_point"]
                - plan["forecast"]["expected_matched_delta_if_the_gain_transfers"],
                6),
        },
        "survives_under": survives,
        "governing_rule": "B_each_tuned_on_train",
        "conclusion": (
            plan["what_will_be_written_under_each_outcome"]["if_it_survives"]
            if survives["B_each_tuned_on_train"]
            else plan["what_will_be_written_under_each_outcome"][
                "if_it_does_not_survive"]
        ),
        "rules_disagree": survives["A_common_q"] != survives[
            "B_each_tuned_on_train"],
        "statistic": plan["statistic"],
    }


def _report(d: dict) -> None:
    n = d["native_call_reference"]
    print(f"\nmatched operating point, {d['n_units']} held-out units "
          f"(read {d['test_fold_read_index']})")
    print(f"  native calls, as published   ours {n['table_field_f1']:.4f}  "
          f"P2Rank {n['p2rank_f1']:.4f}  delta {n['published_delta']:+.4f}")
    for k, v in d["matched"].items():
        p = v["primary"]
        star = "" if p["crosses_zero"] else "  *"
        print(f"  {k:<22} q {v['q_ours']:.2f}/{v['q_p2rank']:.2f}  "
              f"ours {v['table_field_f1']:.4f}  P2Rank {v['p2rank_f1']:.4f}  "
              f"delta {p['delta_point']:+.4f} "
              f"[{p['delta_ci_low']:+.4f}, {p['delta_ci_high']:+.4f}] "
              f"p={p['p_two_sided_bootstrap']:.3f}{star}")
    o = d["oracle"]
    g = o["our_shipped_q_vs_p2rank_oracle_q"]
    print(f"  P2Rank oracle q={o['p2rank_best_q_on_the_held_out_fold']:.2f}  "
          f"F1 {o['p2rank_f1_at_its_oracle_q']:.4f}   "
          f"ours at shipped q vs that: {g['delta_point']:+.4f} "
          f"[{g['delta_ci_low']:+.4f}, {g['delta_ci_high']:+.4f}]")
    f = d["forecast_vs_outcome"]
    print(f"  forecast {f['forecast']:+.4f}  outcome "
          f"{f['outcome_under_the_governing_rule']:+.4f}  "
          f"error {f['forecast_error']:+.4f}")
    print(f"  survives: {d['survives_under']}   "
          f"rules disagree: {d['rules_disagree']}")


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
    if not d["ordering"]["preregistration_is_an_ancestor_of_the_read"]:
        print("FAILED: the artifact does not record the preregistration as "
              "preceding the read")
        return 1
    if not all(r["agrees"] for r in
               d["reproduces_the_published_numbers"].values()):
        print("FAILED: the read did not reproduce the published per-method F1")
        return 1
    rec = d["our_rule_recovers_our_own_call"]
    if rec["f1_drift_from_recomputing_it"] > 1e-6:
        print("FAILED: recomputing our own call moved our own F1, so the "
              "matched delta is not only about the threshold")
        return 1
    plan = json.loads(PREREG.read_text())
    want = {"if_it_survives", "if_it_does_not_survive"}
    said = plan["what_will_be_written_under_each_outcome"]
    survived = d["survives_under"][d["governing_rule"]]
    key = "if_it_survives" if survived else "if_it_does_not_survive"
    if want and d["conclusion"] != said[key]:
        print("FAILED: the conclusion is not the sentence preregistered for "
              "this outcome")
        return 1
    for k, v in d["matched"].items():
        p = v["primary"]
        if p["crosses_zero"] != (p["delta_ci_low"] <= 0.0 <= p["delta_ci_high"]):
            print(f"FAILED: {k} mislabels whether its interval crosses zero")
            return 1
    curve = d["f1_against_q"]
    best = max(curve, key=lambda r: r["p2rank_f1"])
    if abs(best["q"] - d["oracle"]["p2rank_best_q_on_the_held_out_fold"]) > 1e-9:
        print("FAILED: the recorded oracle q is not the curve's argmax")
        return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="audit the committed read artifact")
    ap.add_argument("--calibrate", action="store_true",
                    help="recompute only the already-published native-call F1, "
                         "to test this file's plumbing without making a new "
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
