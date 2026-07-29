#!/usr/bin/env python3
"""What the methods cost, measured so that the comparison is about the methods.

The runtime figure this repository has been quoting is a by-product of the
evaluation runs: each method was timed however the harness happened to invoke it,
on whatever threads it happened to take, around whatever boundary the wrapper
happened to draw. It gave a factor of about fourteen, and roughly none of that
factor was established. Three things were wrong with it.

Every P2Rank call paid for a fresh JVM. A JVM start is around four tenths of a
second before any structure is read, and no deployment serving more than one
receptor pays it per receptor, so charging it per chain measures our script and
not their method.

Neither side had a pinned thread count. Both can use several cores, and a
comparison in which one side happened to get more of them is a comparison of
scheduling.

The boundaries differed. Ours was scoring a matrix that a cache had already been
built for; theirs was a subprocess that parsed a PDB, wrote CSVs to a temporary
directory, and was then parsed back. Those are not the same unit of work.

So this measures both methods around one boundary --- a receptor file on disk to a
score for every residue in it --- on one thread, on the same machine, over the same
chains, in two regimes that are each meaningful and are reported separately.

The cold regime is one process per chain: what a shell loop costs. The warm regime
is one process for all of them, which is what a served deployment costs, and it is
the honest headline because it charges neither side for its startup. Reporting only
the warm number for us and the cold number for them is exactly the error being
corrected here, so both are reported for both.

Our own cost is also split at the point where our claim lives. The claim is that
no floating-point model is evaluated at inference, and that is true of the scoring
step: \\NTabTables{} integer look-ups and a dot product. It says nothing about
getting the \\NTabWires{} wires out of a PDB in the first place, which is
floating-point geometry and is most of the time. A cost section that quoted only
the total would be hiding which half the architecture is responsible for.

The chains are the 770 training receptors, not the 192 evaluation units. Cost does
not depend on a label, so there is no reason to spend a look at the held-out fold
to measure it, and this file cannot be used to choose anything even in principle.

Usage:
  PYTHONPATH=src python3.12 tools/runtime_cost.py
  PYTHONPATH=src python3.12 tools/runtime_cost.py --check
  PYTHONPATH=src python3.12 tools/runtime_cost.py --cold-sample 96 --warm-repeats 2
"""
from __future__ import annotations

# Both libraries read these when they load, so they are set before numpy is
# imported anywhere. A comparison in which one side silently took eight cores is
# not a comparison of the methods.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
RECEPTORS = ROOT / "data/cryptobench_apo/train_receptors"
OUT = ROOT / "results/architecture_sweep/RUNTIME_COST.json"

SCHEMA = "geoaudit.runtime_cost.v1"
THREADS = 1
COLD_SAMPLE = 96
WARM_REPEATS = 3
SEED = 20260728


def _rss_bytes(children: bool = False) -> int:
    """Peak resident set size. Bytes on macOS, kibibytes on Linux."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    peak = resource.getrusage(who).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _cpu_s(children: bool = False) -> float:
    """User plus system CPU seconds consumed so far."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    r = resource.getrusage(who)
    return r.ru_utime + r.ru_stime


def _parallelism(cpu_s: float, wall_s: float) -> float:
    """CPU seconds burned per wall second: 1.0 is one thread, 4.0 is four.

    Asking a tool for one thread and checking that it took one are different
    things, and a batch entry point that quietly fans out across cores would
    make the comparison a measurement of core count.
    """
    return cpu_s / wall_s if wall_s > 0 else 0.0


def _power_state() -> dict:
    """Where the machine's electricity is coming from, and how much is left.

    These laptops clock down on battery and clock down further as the charge
    falls, so a run that starts plugged in and ends on a dying battery measures
    the power manager as much as the code. Both methods are timed in the same
    session, which cancels a uniform slowdown from the ratio, but they are timed
    one after the other, so a drift over the session does not cancel.
    """
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {"source": "unknown"}
    pct = re.search(r"(\d+)%", out)
    return {"source": "AC" if "AC Power" in out else "battery",
            "charge_percent": int(pct.group(1)) if pct else None}


def _hold_awake() -> subprocess.Popen | None:
    """Keep the machine from sleeping for as long as this process lives.

    A measurement that runs while the lid closes records the nap. It happened:
    a run showed two repeats at 263 and 309 seconds and a third that had been
    going for 72 minutes on 36 seconds of CPU, because the machine slept
    underneath it. Wall clock is the quantity being measured, so there is no
    correcting that after the fact -- it has to be prevented.
    """
    try:
        return subprocess.Popen(
            ["caffeinate", "-dimsu", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _quantiles(v: list[float]) -> dict:
    if not v:
        return {}
    a = np.asarray(sorted(v), dtype=float)
    q1, med, q3 = (float(np.quantile(a, q)) for q in (0.25, 0.5, 0.75))
    return {"n": len(v), "median_s": round(med, 4),
            "q1_s": round(q1, 4), "q3_s": round(q3, 4),
            "iqr_s": round(q3 - q1, 4),
            "min_s": round(float(a[0]), 4), "max_s": round(float(a[-1]), 4),
            "total_s": round(float(a.sum()), 2)}


def _units() -> list[str]:
    stems = sorted(p.name[:-len("_receptor.pdb")]
                   for p in RECEPTORS.glob("*_receptor.pdb"))
    if not stems:
        raise SystemExit(
            f"no receptors under {RECEPTORS.relative_to(ROOT)}. They are not "
            f"committed; fetch them before measuring. Checking the committed "
            f"artifact with --check needs none of them")
    return stems


def _path(unit: str) -> Path:
    return RECEPTORS / f"{unit}_receptor.pdb"


class Ours:
    """The counting field, timed around the boundary it is deployed at."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        from pocket_bench.methods.table_field import TableField

        self.field = TableField.load(FIELD)
        self.load_s = time.perf_counter() - t0

    def score(self, unit: str) -> tuple[float, float, int]:
        """``(feature_s, score_s, n_residues)`` for one receptor.

        Split where the claim is: everything up to the wires is floating-point
        geometry, everything after is integer look-ups.
        """
        from pocket_bench.methods.algebraic_descriptors import (
            FEATURE_NAMES, algebraic_residue_features)
        from pocket_bench.methods.wide_descriptors import build_wide

        path, chain = _path(unit), unit.split("_")[1] if "_" in unit else None
        t0 = time.perf_counter()
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        X, _ = build_wide(F, codes, ctr, n_res_per, tuple(FEATURE_NAMES),
                          self.field.prop)
        t1 = time.perf_counter()
        s = self.field.score_matrix(X, ctr, n_res_per)
        self.field.positive_call(s)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, len(resseq)


def _java_env() -> dict:
    from pocket_bench.methods.p2rank_wrap import _java_env as je

    env = dict(je())
    # P2Rank's own thread option covers its work pool; the JVM still respects
    # these for anything native it reaches.
    env["OMP_NUM_THREADS"] = str(THREADS)
    return env


def _prank() -> str:
    from pocket_bench.methods.p2rank_wrap import _find_p2rank

    exe = _find_p2rank()
    if not exe:
        raise SystemExit(
            "P2Rank (prank) not found; set P2RANK_HOME. Checking the committed "
            "artifact with --check does not need it")
    return str(exe)


def p2rank_cold(units: list[str]) -> tuple[list[float], int]:
    """One JVM per chain: what a shell loop over receptors costs."""
    exe, env = _prank(), _java_env()
    times: list[float] = []
    for i, unit in enumerate(units, 1):
        with tempfile.TemporaryDirectory(prefix="rtcold_") as tmp:
            work = Path(tmp)
            local = work / "rec.pdb"
            local.write_bytes(_path(unit).read_bytes())
            out = work / "out"
            out.mkdir()
            t0 = time.perf_counter()
            p = subprocess.run(
                [exe, "predict", "-f", str(local), "-o", str(out),
                 "-threads", str(THREADS)],
                capture_output=True, text=True, timeout=900, env=env)
            dt = time.perf_counter() - t0
            if p.returncode != 0 or not list(out.rglob("*_residues.csv")):
                raise SystemExit(
                    f"P2Rank failed on {unit}: "
                    f"{(p.stderr or p.stdout or '')[-300:]}")
            times.append(dt)
        if i % 16 == 0:
            print(f"    p2rank cold {i}/{len(units)} "
                  f"(median so far {np.median(times):.3f}s)", flush=True)
    return times, _rss_bytes(children=True)


def p2rank_warm(units: list[str]) -> tuple[float, int, int, float]:
    """One JVM for every chain: what a served deployment costs.

    P2Rank's dataset mode is its own batch entry point, so this is the tool used
    the way it is meant to be used rather than a loop we wrapped around it.
    """
    exe, env = _prank(), _java_env()
    with tempfile.TemporaryDirectory(prefix="rtwarm_") as tmp:
        work = Path(tmp)
        pdbs = work / "pdbs"
        pdbs.mkdir()
        for unit in units:
            shutil.copyfile(_path(unit), pdbs / f"{unit}.pdb")
        ds = work / "all.ds"
        ds.write_text("\n".join(f"pdbs/{u}.pdb" for u in units) + "\n")
        out = work / "out"
        out.mkdir()
        c0, t0 = _cpu_s(children=True), time.perf_counter()
        p = subprocess.run(
            [exe, "predict", str(ds), "-o", str(out),
             "-threads", str(THREADS)],
            capture_output=True, text=True, timeout=36000, env=env)
        dt = time.perf_counter() - t0
        cpu = _cpu_s(children=True) - c0
        done = len(list(out.rglob("*_residues.csv")))
        if p.returncode != 0:
            raise SystemExit(
                f"P2Rank batch failed: {(p.stderr or p.stdout or '')[-400:]}")
        if done != len(units):
            raise SystemExit(
                f"P2Rank batch scored {done} of {len(units)} chains; an "
                f"amortised per-chain cost over a partial run would be wrong")
    return dt, done, _rss_bytes(children=True), cpu


def jvm_floor() -> float:
    """What a JVM start costs before any structure is read.

    Measured as the tool's own no-dataset exit, which loads the JVM and P2Rank's
    config and then stops. It is the quantity the cold regime charges per chain
    and the warm regime charges once.
    """
    exe, env = _prank(), _java_env()
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run([exe, "predict"], capture_output=True, text=True,
                       timeout=300, env=env)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs))


def build(cold_sample: int, warm_repeats: int) -> dict:
    awake = _hold_awake()
    power_before = _power_state()
    units = _units()
    rng = np.random.default_rng(SEED)
    cold_units = sorted(rng.choice(units, size=min(cold_sample, len(units)),
                                   replace=False).tolist())

    print(f"{len(units)} training receptors, {THREADS} thread, "
          f"warm over all of them x{warm_repeats}, cold over "
          f"{len(cold_units)}", flush=True)

    print("  ours, warm", flush=True)
    ours = Ours()
    feat: dict[str, list[float]] = {u: [] for u in units}
    scor: dict[str, list[float]] = {u: [] for u in units}
    n_res = {}
    warm_totals, warm_cpus, ours_cpu = [], [], 0.0
    for rep in range(warm_repeats):
        c0, t0 = _cpu_s(), time.perf_counter()
        for unit in units:
            f, s, n = ours.score(unit)
            feat[unit].append(f)
            scor[unit].append(s)
            n_res[unit] = n
        warm_totals.append(time.perf_counter() - t0)
        warm_cpus.append(_cpu_s() - c0)
        ours_cpu += warm_cpus[-1]
        print(f"    repeat {rep + 1}/{warm_repeats}: {warm_totals[-1]:.1f}s "
              f"wall, {warm_cpus[-1]:.1f}s CPU", flush=True)
    ours_rss = _rss_bytes()
    ours_par = _parallelism(ours_cpu, sum(warm_totals))

    # Identical work repeated on an undisturbed machine burns the same CPU per
    # wall second every time. When it does not, the machine did something else:
    # it slept, it throttled, or another job took the cores. None of those are
    # properties of the code being timed, and all of them land in the wall clock
    # that is the headline number here, so the run is refused rather than
    # averaged. This is not hypothetical -- a run was lost to a closed lid.
    ratios = [_parallelism(c, w) for c, w in zip(warm_cpus, warm_totals)]
    if warm_repeats > 1 and max(ratios) > 1.25 * min(ratios):
        raise SystemExit(
            "the repeats of the warm pass did not run under the same "
            "conditions: CPU seconds per wall second came out as "
            + ", ".join(f"{r:.2f}" for r in ratios)
            + ". The most likely cause is the machine sleeping, throttling, or "
              "sharing the cores with something else. Re-run it undisturbed; "
              "wall clock is the quantity being measured and cannot be "
              "corrected after the fact")

    # The median over repeats first, so a single scheduling hiccup on one chain
    # does not enter the distribution over chains.
    f_med = {u: float(np.median(v)) for u, v in feat.items()}
    s_med = {u: float(np.median(v)) for u, v in scor.items()}
    tot = {u: f_med[u] + s_med[u] for u in units}

    print("  ours, cold (one process per chain)", flush=True)
    cold_ours = []
    for i, unit in enumerate(cold_units, 1):
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-c",
             "from pocket_bench.methods.table_field import TableField\n"
             "import sys\n"
             "tf = TableField.load(sys.argv[1])\n"
             "r, s, c = tf.score_receptor(sys.argv[2], chain=sys.argv[3])\n"
             "assert len(r) == len(s)\n",
             str(FIELD), str(_path(unit)), unit.split("_")[1]],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                 "PYTHONDONTWRITEBYTECODE": "1"})
        if p.returncode != 0:
            raise SystemExit(f"our cold call failed on {unit}: "
                             f"{(p.stderr or '')[-400:]}")
        cold_ours.append(time.perf_counter() - t0)
        if i % 32 == 0:
            print(f"    ours cold {i}/{len(cold_units)}", flush=True)

    print("  p2rank, JVM floor", flush=True)
    floor = jvm_floor()
    print(f"    {floor:.3f}s per start", flush=True)

    print("  p2rank, warm (one JVM, dataset mode)", flush=True)
    p2_warm_total, p2_done, _, p2_cpu = p2rank_warm(units)
    p2_par = _parallelism(p2_cpu, p2_warm_total)
    print(f"    {p2_warm_total:.1f}s for {p2_done} chains "
          f"({p2_warm_total / p2_done:.3f}s each), "
          f"{p2_par:.2f} CPU-s per wall-s", flush=True)

    print("  p2rank, cold (one JVM per chain)", flush=True)
    p2_cold, p2_rss = p2rank_cold(cold_units)

    ours_warm_per_chain = float(np.median(list(tot.values())))
    p2_warm_per_chain = p2_warm_total / p2_done
    scoring_only = float(np.median(list(s_med.values())))

    # Wall clock is what a user waits and is the number worth leading with, but
    # it is only a fair comparison if both sides got the same number of cores.
    # Asking for one thread does not guarantee getting one -- Accelerate ignores
    # the environment variable on this platform -- so CPU seconds per chain is
    # recorded too. It is thread-independent, and it is the number that settles
    # the comparison when the wall-clock one was taken under uneven parallelism.
    ours_cpu_per_chain = ours_cpu / (len(units) * warm_repeats)
    p2_cpu_per_chain = p2_cpu / p2_done

    def _verdict(ours: float, theirs: float, unit: str) -> str:
        if theirs > ours:
            return (f"the counting field is the cheaper of the two per chain in "
                    f"{unit}, by {theirs / ours:.2f} times")
        return (f"P2Rank is the cheaper of the two per chain in {unit}, by "
                f"{ours / theirs:.2f} times")

    # The verdict is read off the measurement rather than written in advance.
    # This file exists because the number it replaces was asserted, and an
    # asserted replacement would be the same mistake with better plumbing.
    warm_verdict = _verdict(ours_warm_per_chain, p2_warm_per_chain, "wall clock")
    cpu_verdict = _verdict(ours_cpu_per_chain, p2_cpu_per_chain, "CPU seconds")
    if p2_warm_per_chain <= ours_warm_per_chain:
        warm_verdict += (
            ". The speed advantage this repository previously claimed does not "
            "survive giving both methods one process for the whole dataset: it "
            "was an artifact of starting a JVM for every chain while our own "
            "field was loaded once")
    if scoring_only < p2_warm_per_chain:
        scoring_verdict = (
            f"the part of our cost the architecture is responsible for is "
            f"genuinely small: the table look-ups and the integer sum take "
            f"{scoring_only:.4f}s, which is less than "
            f"P2Rank's whole per-chain cost. What is not small is extracting "
            f"the wires, and that is floating-point geometry shared with any "
            f"method that reads the same descriptors")
    else:
        scoring_verdict = (
            f"even our scoring step alone ({scoring_only:.4f}s) costs more "
            f"than P2Rank's whole per-chain figure, so no part of the cost "
            f"claim survives")
    field_bytes = FIELD.stat().st_size
    gz = len(gzip.compress(FIELD.read_bytes(), 9))
    model_dir = Path(_prank()).parent / "models" / "default"
    p2_model_bytes = (sum(p.stat().st_size for p in model_dir.rglob("*")
                          if p.is_file()) if model_dir.is_dir() else None)

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": (
            "what each method costs per chain when both are given the same "
            "boundary, the same thread count, the same machine and the same "
            "structures"),
        "why_the_previous_number_was_not_a_measurement": [
            "every P2Rank call paid for a fresh JVM, which no deployment "
            "serving more than one receptor pays per receptor",
            "neither side had a pinned thread count, so the ratio partly "
            "measured which of them the scheduler favoured",
            "the boundaries differed: ours scored a matrix a cache had already "
            "built, theirs parsed a PDB and round-tripped CSVs through a "
            "temporary directory",
        ],
        "boundary": (
            "a receptor file on disk to a score for every residue in it. For us "
            "that is parsing, the 43 local quantities, the 645 wires, the "
            "quantisation, the table look-ups and the gate. For P2Rank it is its "
            "own predict entry point and the residue table it writes"),
        "population": {
            "chains": len(units),
            "which": "the 770 training receptors",
            "why_not_the_evaluation_fold": (
                "cost does not depend on a label, so measuring it is no reason "
                "to spend a look at the held-out fold"),
            "residues": int(sum(n_res.values())),
            "median_residues_per_chain": int(np.median(list(n_res.values()))),
        },
        "controls": {
            "threads": THREADS,
            "thread_environment": {v: os.environ.get(v) for v in
                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS")},
            "p2rank_threads_flag": f"-threads {THREADS}",
            "machine": platform.platform(),
            "processor": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True).stdout.strip() or None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warm_repeats": warm_repeats,
            "per_chain_statistic": "median over repeats, then the distribution "
                                   "over chains",
            "power_at_the_start": power_before,
            "power_at_the_end": _power_state(),
            "sleep_was_held_off": awake is not None,
            "warm_repeat_wall_s": [round(t, 1) for t in warm_totals],
            "warm_repeat_cpu_s": [round(c, 1) for c in warm_cpus],
            "why_the_repeats_are_listed": (
                "so a reader can see that they ran under the same conditions. "
                "The build refuses to write this file if their CPU-per-wall "
                "ratios differ by more than a quarter, which is how a machine "
                "that slept or throttled mid-run announces itself"),
            "seed_for_the_cold_subsample": SEED,
        },
        "warm": {
            "what_it_is": "one process for every chain: the steady state of a "
                          "served deployment, charging neither side for startup",
            "table_field": {
                **_quantiles(list(tot.values())),
                "features_median_s": round(float(np.median(list(f_med.values()))), 4),
                "scoring_median_s": round(float(np.median(list(s_med.values()))), 4),
                "fraction_of_the_median_spent_on_features": round(
                    float(np.median(list(f_med.values()))) / ours_warm_per_chain, 4),
                "peak_rss_mb": round(ours_rss / 2 ** 20, 1),
                "one_time_load_s": round(ours.load_s, 4),
                "cpu_seconds_per_wall_second": round(ours_par, 2),
                "cpu_seconds_per_chain": round(ours_cpu_per_chain, 4),
            },
            "p2rank": {
                "n": p2_done,
                "total_s": round(p2_warm_total, 2),
                "amortised_per_chain_s": round(p2_warm_per_chain, 4),
                "mode": "its own dataset entry point, one JVM",
                "cpu_seconds_per_wall_second": round(p2_par, 2),
                "cpu_seconds_per_chain": round(p2_cpu_per_chain, 4),
                "note": "P2Rank reports no per-chain time in this mode, so only "
                        "an amortised mean is available and no IQR is claimed",
            },
            "ratio_p2rank_over_table_field": round(
                p2_warm_per_chain / ours_warm_per_chain, 2),
            "ratio_p2rank_over_table_field_cpu": round(
                p2_cpu_per_chain / ours_cpu_per_chain, 2),
            "verdict": warm_verdict,
            "verdict_on_cpu_seconds": cpu_verdict,
        },
        "did_either_side_get_more_than_one_thread": {
            "how": "CPU seconds charged to the process divided by wall seconds "
                   "elapsed. One thread fully busy is 1.0; anything much above "
                   "it means the tool ignored the thread count it was given",
            "table_field": round(ours_par, 2),
            "p2rank_batch": round(p2_par, 2),
            "both_within_tolerance": bool(ours_par < 1.5 and p2_par < 1.5),
            "who_got_more": ("table_field" if ours_par > p2_par else "p2rank"),
            "why_the_conclusion_survives_it": (
                "the side that got more cores is the side the wall-clock "
                "comparison favours, so an uneven thread count can only have "
                "flattered it. Reading the comparison in CPU seconds instead, "
                "which no thread count can tilt, moves the result further in "
                "the same direction rather than reversing it"),
            "note": "the thread environment variables are set before numpy is "
                    "imported and P2Rank is given -threads 1, but Apple's "
                    "Accelerate framework does not honour "
                    "VECLIB_MAXIMUM_THREADS for every kernel, so the request is "
                    "recorded as a request and the outcome is measured",
        },
        "cold": {
            "what_it_is": "one process per chain: what a shell loop costs, and "
                          "what the previously quoted figure was measuring on "
                          "one side only",
            "n_chains": len(cold_units),
            "table_field": _quantiles(cold_ours),
            "p2rank": {**_quantiles(p2_cold),
                       "peak_rss_mb": round(p2_rss / 2 ** 20, 1)},
            "ratio_of_medians": round(
                float(np.median(p2_cold)) / float(np.median(cold_ours)), 2),
            "jvm_start_median_s": round(floor, 4),
            "jvm_start_as_fraction_of_p2ranks_cold_median": round(
                floor / float(np.median(p2_cold)), 4),
        },
        "model_size": {
            "table_field_json_bytes": field_bytes,
            "table_field_json_mb": round(field_bytes / 2 ** 20, 2),
            "table_field_gzip_mb": round(gz / 2 ** 20, 2),
            "p2rank_default_model_bytes": p2_model_bytes,
            "p2rank_default_model_mb": (round(p2_model_bytes / 2 ** 20, 2)
                                        if p2_model_bytes else None),
            "note": "ours is the whole detector: 5152 tables of cell rates plus "
                    "the integer multiplicities and the propensity table. There "
                    "is no separate feature extractor to ship, because the "
                    "quantities are computed from coordinates",
        },
        "what_this_does_and_does_not_support": {
            "in_the_steady_state": warm_verdict,
            "in_cpu_seconds": cpu_verdict + (
                ", which is the reading no thread count can tilt and therefore "
                "the one that settles the comparison here"),
            "one_process_per_chain": (
                f"the counting field is "
                f"{float(np.median(p2_cold)) / float(np.median(cold_ours)):.2f} "
                f"times cheaper, and {100 * floor / float(np.median(p2_cold)):.0f} "
                f"per cent of what it is cheaper than is a JVM start"),
            "where_our_own_cost_is": scoring_verdict,
            "what_survives_on_cost": (
                "the compiled detector is a small artifact and the scoring step "
                "is nearly free"
                + ("" if p2_warm_per_chain > ours_warm_per_chain else
                   ". What does not survive is a per-chain advantage over "
                   "P2Rank in a served deployment")),
        },
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "it runs on training receptors, reads no label, and produces no "
            "quantity that could rank two methods on accuracy"),
        "per_chain": {u: {"features_s": round(f_med[u], 5),
                          "scoring_s": round(s_med[u], 5),
                          "n_residues": n_res[u]} for u in units},
    }


def _report(d: dict) -> None:
    w, c = d["warm"], d["cold"]
    tf, p2 = w["table_field"], w["p2rank"]
    print(f"\nfair cost, {d['population']['chains']} chains, "
          f"{d['controls']['threads']} thread, {d['controls']['processor']}")
    print(f"  warm (one process for all chains, the deployable steady state)")
    print(f"    table field   median {tf['median_s']:.3f}s  "
          f"IQR {tf['iqr_s']:.3f}s  peak RSS {tf['peak_rss_mb']:.0f} MB")
    print(f"      of which features {tf['features_median_s']:.3f}s "
          f"({100 * tf['fraction_of_the_median_spent_on_features']:.1f}%), "
          f"table look-ups {tf['scoring_median_s']:.4f}s")
    print(f"    p2rank        {p2['amortised_per_chain_s']:.3f}s per chain "
          f"amortised over {p2['n']}")
    print(f"    ratio         {w['ratio_p2rank_over_table_field']:.2f}x on wall "
          f"clock, {w['ratio_p2rank_over_table_field_cpu']:.2f}x on CPU seconds "
          f"({tf['cpu_seconds_per_chain']:.3f} vs "
          f"{p2['cpu_seconds_per_chain']:.3f} CPU-s per chain)")
    print(f"    threads       ours {tf['cpu_seconds_per_wall_second']:.2f}, "
          f"p2rank {p2['cpu_seconds_per_wall_second']:.2f} CPU-s per wall-s "
          f"(both asked for 1)")
    print(f"    -> {w['verdict']}")
    print(f"    -> {w['verdict_on_cpu_seconds']}")
    print(f"  cold (one process per chain, {c['n_chains']} chains)")
    print(f"    table field   median {c['table_field']['median_s']:.3f}s")
    print(f"    p2rank        median {c['p2rank']['median_s']:.3f}s, of which "
          f"{c['jvm_start_median_s']:.3f}s is the JVM "
          f"({100 * c['jvm_start_as_fraction_of_p2ranks_cold_median']:.0f}%)")
    print(f"    ratio         {c['ratio_of_medians']:.2f}x")
    m = d["model_size"]
    print(f"  model  ours {m['table_field_json_mb']:.2f} MB JSON "
          f"({m['table_field_gzip_mb']:.2f} MB gzipped), "
          f"p2rank {m['p2rank_default_model_mb']} MB")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    w, c = d["warm"], d["cold"]
    if d["controls"]["threads"] != THREADS:
        print(f"FAILED: measured on {d['controls']['threads']} threads, this "
              f"file pins {THREADS}")
        return 1
    for var, value in d["controls"]["thread_environment"].items():
        if value != str(THREADS):
            print(f"FAILED: {var} was {value} during the measurement, so the "
                  f"thread count was not actually pinned")
            return 1
    # The per-chain rows have to be the population the summary claims.
    if len(d["per_chain"]) != d["population"]["chains"]:
        print(f"FAILED: {len(d['per_chain'])} per-chain rows for a population "
              f"of {d['population']['chains']}")
        return 1
    tf = w["table_field"]
    if abs(tf["features_median_s"] + tf["scoring_median_s"]
           - tf["median_s"]) > 0.05 * tf["median_s"]:
        print("FAILED: the feature and scoring medians do not add up to the "
              "total median, so the split is not a split of this measurement")
        return 1
    # The warm ratio is the number the chapter leads with, and it must not be
    # quietly replaced by the cold one, which is the error being corrected.
    if w["ratio_p2rank_over_table_field"] >= c["ratio_of_medians"]:
        print(f"FAILED: the warm ratio ({w['ratio_p2rank_over_table_field']}) is "
              f"no smaller than the cold one ({c['ratio_of_medians']}); the "
              f"chapter says removing the per-chain JVM start shrinks the "
              f"margin and would be wrong")
        return 1
    # The conditions the run happened under travel with it, because a wall clock
    # measured on a machine that slept or throttled is not a measurement.
    ctl = d["controls"]
    walls, cpus = (ctl.get("warm_repeat_wall_s") or [],
                   ctl.get("warm_repeat_cpu_s") or [])
    if len(walls) != ctl["warm_repeats"] or len(cpus) != len(walls):
        print(f"FAILED: {ctl['warm_repeats']} warm repeats were run but the "
              f"artifact lists {len(walls)} wall and {len(cpus)} CPU totals; "
              f"regenerate it with a build that records them")
        return 1
    ratios = [c / w for c, w in zip(cpus, walls) if w > 0]
    if len(ratios) > 1 and max(ratios) > 1.25 * min(ratios):
        print(f"FAILED: the warm repeats ran at "
              f"{', '.join(f'{r:.2f}' for r in ratios)} CPU seconds per wall "
              f"second; the machine was not in the same state throughout and "
              f"the wall clock is not a measurement of the code")
        return 1
    if not ctl.get("sleep_was_held_off"):
        print("FAILED: the run did not hold a power assertion, so nothing "
              "stopped the machine sleeping underneath it")
        return 1
    # The two methods are timed one after the other, so a machine that changes
    # its clock policy partway through charges the difference to whichever went
    # second rather than to either method.
    before = (ctl.get("power_at_the_start") or {}).get("source")
    after = (ctl.get("power_at_the_end") or {}).get("source")
    if before != after:
        print(f"FAILED: the machine was on {before} power when the run started "
              f"and {after} when it finished; the two methods were timed under "
              f"different clock policies")
        return 1
    # A thread count that was requested and not granted does not by itself
    # invalidate the comparison, but it does if it favours the side the
    # conclusion favours. That is the condition worth failing on.
    par = d["did_either_side_get_more_than_one_thread"]
    wall_favours = ("table_field" if w["ratio_p2rank_over_table_field"] > 1.0
                    else "p2rank")
    if not par["both_within_tolerance"] and par["who_got_more"] == wall_favours:
        print(f"FAILED: {par['who_got_more']} used more CPU per wall second "
              f"(ours {par['table_field']}, p2rank {par['p2rank_batch']}) and "
              f"is also the side the wall-clock conclusion favours; the "
              f"comparison would be reporting its own thread advantage")
        return 1
    # Both verdicts are derived, and both derivations have to still hold. A
    # retraction that silently became a boast would be worse than the original
    # error.
    for ratio, verdict, name in (
            (w["ratio_p2rank_over_table_field"], w["verdict"], "wall clock"),
            (w["ratio_p2rank_over_table_field_cpu"], w["verdict_on_cpu_seconds"],
             "CPU seconds")):
        if (ratio > 1.0) != ("the counting field is the cheaper" in verdict):
            print(f"FAILED: the {name} ratio is {ratio} but the recorded "
                  f"verdict reads '{verdict[:70]}...'")
            return 1
    # The two regimes must not be allowed to disagree without the artifact
    # saying which one settles the question.
    tfw = w["table_field"]
    if abs(tfw["cpu_seconds_per_chain"]
           - tfw["median_s"] * tfw["cpu_seconds_per_wall_second"]) > \
            0.35 * tfw["cpu_seconds_per_chain"]:
        print(f"FAILED: our CPU seconds per chain "
              f"({tfw['cpu_seconds_per_chain']}) is not consistent with the "
              f"wall median times the measured parallelism; one of the two was "
              f"recorded from a different run")
        return 1
    if tf["fraction_of_the_median_spent_on_features"] <= 0.5:
        print(f"FAILED: features are now "
              f"{tf['fraction_of_the_median_spent_on_features']:.2f} of our "
              f"cost; the chapter says most of our time is the floating-point "
              f"geometry and not the look-ups, and must be rewritten")
        return 1
    if d.get("test_fold_read_index") is not None:
        print("FAILED: a cost measurement has started declaring a fold read")
        return 1
    if "train" not in d["population"]["which"]:
        print(f"FAILED: measured on {d['population']['which']}, not the "
              f"training receptors")
        return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cold-sample", type=int, default=COLD_SAMPLE)
    ap.add_argument("--warm-repeats", type=int, default=WARM_REPEATS)
    args = ap.parse_args(argv)
    if args.check:
        return _check()
    d = build(args.cold_sample, args.warm_repeats)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
