PYTHON ?= python3.12
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: verify test consistency readme strict-json macros environment archive icloud ledger artifacts figures recompute residues published crossval cases freeze
verify: consistency strict-json readme macros environment archive icloud ledger artifacts recompute residues published crossval cases quotient gap prereg read5 banks
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
