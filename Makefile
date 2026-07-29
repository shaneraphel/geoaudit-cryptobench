PYTHON ?= python3.12
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: verify test consistency readme strict-json macros environment archive icloud ledger artifacts figures recompute residues published crossval cases freeze
verify: consistency strict-json readme wires compileapp macros environment archive icloud ledger artifacts recompute residues published crossval cases quotient gap prereg read5 banks sens p2op match read6 trainop match2 read7 audit cost interp endpoint cov subplan read8 read9 plmw plmseq plmscore plmplan read10 pmplan read11 curveplan read12 rule
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/verify_claims.py --root .

# The four case studies are chosen by a stated rule from the labels and the raw
# per-residue output, both of which are committed, so which structures they are
# and what they scored re-derive anywhere. The spatial blocks need the receptor
# PDBs, which are not committed; where they are absent this checks what it can
# and says so rather than blocking.
cases:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/select_case_studies.py --check

# The architecture was chosen on one half-split of the training fold. This holds
# the cross-validation that says the choice survives 28 other splits to its own
# tables: CI has neither the descriptor cache nor the OSF folds, so it cannot
# rerun the selections, but it can check that the headline follows from the
# per-split rankings recorded beneath it and still names the frozen winner.
crossval:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/crossvalidate_architecture.py --check

# The quotient counterattack: that the capacity arithmetic recomputes, that the
# summary follows from the fourteen per-split rankings, and that the selection
# artifact still says it never read the test fold.
quotient:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/counterattack_quotient_tables.py --check

# The gap diagnosis: capacity is not what binds, the fan-out is, and the fitted
# linear readout is handed wires the counting field never sees. Refuses an
# artifact that claims a test-fold read or whose arithmetic no longer adds up.
gap:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/gap_decomposition.py --audit

# The functional the paired comparison is summarised with was fixed on the
# training partition, and this refuses an artifact whose named statistic is no
# longer the one its own selection rule returns from its own recorded numbers,
# or that has quietly lost the forecast it committed to.
prereg:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_statistic.py --check

# The fifth reading of the held-out fold, taken under that statistic. Refuses an
# artifact whose named statistic has drifted from the committed choice, or that
# has stopped reporting the mean beside it.
read5:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregistered_read.py --check

# The generated invariant banks. Refuses an artifact whose recorded descriptor
# counts no longer match the modules, or whose lift does not follow from its own
# per-split ceilings.
banks:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/expand_invariant_bank.py --check

# The quantisation and fan-out sweep. Refuses a checkpoint of an unfinished run,
# which would let the paper quote a range over settings that were never all
# measured, and refuses one whose published row disagrees with the shipped
# configuration.
sens:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/sensitivity_sweep.py --check

# P2Rank's operating point, chosen on the training receptors by the same grid
# search that chose ours. Refuses an artifact whose recorded q is not the argmax
# of its own committed curve, or whose re-run disagreed with the training-fold
# summary the paper already quotes. Needs no JVM.
p2op:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/p2rank_train_operating_point.py --check

# The matched-threshold plan, including the sentence the paper must carry if the
# F1 margin does not survive. Refuses an artifact whose rules or committed
# outcomes have changed, or whose forecast is no longer the subtraction it says.
match:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_matched_operating_point.py --check

# The sixth reading. Refuses a read that did not first reproduce the published
# per-method F1, whose plan is not an ancestor of it in git, or whose stated
# conclusion is not the sentence preregistered for the outcome it got.
read6:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/matched_operating_point_read.py --check

# Every threshold either method is held to, chosen on the training fold under
# both objectives and again on a half the scored field never counted. Refuses an
# artifact whose selected q is not the argmax of its own curve, that no longer
# reproduces the shipped q, or that claims to have touched the held-out fold.
trainop:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/train_operating_points.py --check

# The plan for the seventh read. Rebuilt in memory and compared field by field
# against what is committed, so a plan that has quietly drifted to match its
# inputs fails instead of passing.
match2:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_matched_full.py --check

# The seventh reading: four conventions, four metrics, three resampling units.
# Refuses a read that does not reproduce the frozen bootstrap under the
# deployment rules, that moves the matched F1 delta read six published, whose
# precision and recall deltas disagree in sign at a matched budget, or whose
# verdict does not follow from its own interval.
read7:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/matched_full_read.py --check

# The auditability claim, taken apart on residues the detector got right and
# wrong. Regenerating it needs the uncommitted receptors, but checking it does
# not: the gate re-adds the family terms and refuses an artifact whose
# decomposition no longer sums to the score, whose cases have drifted from the
# committed selection, or that has started declaring a test-fold read.
audit:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/audit_decomposition.py --check

# The controlled cost measurement, which withdrew a speedup this repository had
# published. Re-measuring needs the receptors and twenty minutes; the gate only
# reads the artifact, and it refuses one whose recorded verdict disagrees with
# its own ratios, or whose wall-clock conclusion favours the side that was
# granted more cores than it asked for.
cost:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/runtime_cost.py --check

# What the readout is worth over a linear model on the same wires, and what each
# step of the architecture buys. The gate refuses an artifact whose published arm
# does not score what the sensitivity sweep says the published configuration
# scores on the same half, whose reported means are not the means of the
# per-chain vectors stored beside them, or whose list of unresolved arms
# disagrees with its own intervals.
interp:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/interpretable_baselines.py --check

# Which endpoint the paper is entitled to lead with. The gate recomputes, from
# the commit graph as it stands now, how many indexed fold reads precede the
# preregistration, and fails if that count no longer justifies the demotion, if
# the primary endpoint stops being the mean, or if the mean starts to resolve --
# each of which is a different paper from the one written around it.
endpoint:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/endpoint_status.py --check

# Where on the fold the two detectors differ. Three gates, in order: the
# covariates must still follow from the deposit and the labels without any score
# being opened, the plan must still describe them and still call itself
# exploratory, and the eighth read must still reproduce read five and keep every
# surviving band next to the trend test that qualifies it.
cov:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/subgroup_covariates.py --check

subplan:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_subgroups.py --check

read8:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/subgroup_read.py --check

# Whether the field is any use as a pocket finder rather than a residue scorer.
# The counting field emits no pockets of its own, so the read builds a stage from
# its residue scores; the gate exists because the construction is the part a
# reader should distrust, and it has to keep reproducing from the plan that fixed
# it before the fold was opened.
read9:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/pocket_read.py --check

# The benchmark's own cryptic-site baseline. Four gates, in order: the network
# must still be the one published on OSF, read out of that checkpoint at the
# offsets its own index states; the baseline's scores must still be the ones the
# plan pinned, computed at the layer the authors' worked example identifies; the
# plan must still call itself exploratory and still carry a sentence for losing;
# and the tenth read must still reproduce, including which of those sentences it
# is entitled to print. The weight gate needs the OSF checkpoint, and the score
# gate needs neither it nor the encoder -- it checks the artifact, not the model.
plmw:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/export_plmnn_weights.py --check

plmseq:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/plmnn_sequences.py --check

plmscore:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/plmnn_embed.py --check

plmplan:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_plmnn.py --check

read10:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/plmnn_read.py --check

# The threshold axis. Three operating points cannot answer whether they were the
# flattering three, so the twelfth read binarises the same frozen scores at every
# calling fraction from 2% to 40%. The gate checks the two things that make a
# curve over 39 cut points admissible rather than a search: that it reproduces
# the seventh read where the two overlap, and that nothing downstream reads a
# value off it.
# The cryptic-specific baseline the benchmark names. The plan's gate is the
# one that matters: the rebuild has to keep reproducing PocketMiner's own
# published residue counts, because a comparison against a wrongly assembled
# baseline is worth nothing however wide its margin.
pmplan:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_pocketminer.py --check

read11:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/pocketminer_read.py --check

# The labelling rule, recovered by fitting candidates against the benchmark's own
# training records rather than assumed from its prose. An external validation set
# whose labels mean something slightly different is not validation, so this gate
# runs before anything is built on the rule. It reads the training fold only.
rule:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/recover_cryptobench_rule.py --check

curveplan:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/preregister_threshold_curve.py --check

read12:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/threshold_curve.py --check

# The headline, rederived from the committed labels and raw per-residue scores by
# code that imports nothing from the harness. The other gates check that the
# artifacts agree with each other; this one checks that they agree with the data.
# It is what caught the residue-numbering collision, which every self-consistent
# artifact in the repository had faithfully propagated.
recompute:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/recompute_from_raw.py

# What counts as one residue, and what that rule costs. Needs the receptors, so
# on a machine that has not fetched them this reports and passes rather than
# blocking; where they are present it pins the recall denominator.
residues:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_residue_identity.py

# Our P2Rank must land on the published CryptoBench row once the aggregation
# convention is matched. Saying "the harness was fixed" is not evidence; this is.
published:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/reproduce_published_p2rank.py

# The declared environment and the real one drifted once, in the worst possible
# direction: pyproject demanded a numpy that segfaults here. The lock is
# measured, and this fails if the machine no longer matches it.
environment:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_environment.py --audit

# Appendix A states the input contract: the 43 local quantities, the
# neighbourhood each is read over, its value at the boundary, and the rule that
# expands them into the wires. It is generated from the modules, and the
# appendix says so in its own header, so it has to be checked -- an unguarded
# generated file is a file that describes a version of the code that no longer
# exists.
wires:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/emit_wire_appendix.py --check

# Everything between a wire and a score: the bank and its seed, the cell rule,
# the solve, the gate, the boundary cases and the sample flow. The gate redraws
# the table bank from the seed the artifact records and refuses to pass if it
# does not reproduce the shipped bank pair for pair, so the appendix cannot
# document a draw nobody can repeat.
compileapp:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools $(PYTHON) tools/emit_compile_appendix.py --check

# The manuscript cites macros only, so a stale macro file is a stale paper.
macros:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_frozen_numbers.py --check

# Re-derive both paired-bootstrap artifacts from the telemetry. Every artifact a
# reader is asked to trust has a generator in the repository; this is it.
freeze:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_environment.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/build_test_fold_ledger.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/classify_artifacts.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/freeze_bootstrap.py --all --quiet
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/crossvalidate_architecture.py --quiet
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/select_case_studies.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_frozen_numbers.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/render_results_section.py --write

# The strongest baseline must be recomputable from its archived raw output
# alone, without a JVM. This checks coverage, provenance and checksums.
archive:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/check_p2rank_archive.py

# The bulk raw PDBs are not committed; the manifest says they are cached
# in iCloud, and this checks that they are and that they are unaltered.
icloud:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/verify_icloud_cache.py

# How often the held-out fold has been scored, counted from the artifacts.
# A hand-written version of this disclosure was wrong by a factor of three.
ledger:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/build_test_fold_ledger.py --check

# Every frozen artifact declares what it is for. Two dozen architecture
# sweeps beside a headline number must be a stated fact, not an inference.
# This also holds the committed images to their sources: `artifacts` fails if
# a figure's input changed underneath it, which needs no plotting library.
artifacts:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/classify_artifacts.py --check

# Redraw the README's images from the frozen artifacts. Needs matplotlib, which
# is why it is a separate target: CI checks the images, it does not redraw them.
figures:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/make_official_figures.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/classify_artifacts.py

# Two frozen reports may never disagree about the same quantity.
consistency:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/check_report_consistency.py

# The README's results block is generated from the artifacts, never typed.
readme:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/render_results_section.py --check

# No bare NaN/Infinity token may reach a published JSON.
strict-json:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/check_strict_json.py

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v
