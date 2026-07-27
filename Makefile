PYTHON ?= python3.12
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: verify test consistency readme strict-json macros environment archive icloud ledger freeze
verify: consistency strict-json readme macros environment archive icloud ledger
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/verify_claims.py --root .

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
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/freeze_bootstrap.py --all --quiet
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
