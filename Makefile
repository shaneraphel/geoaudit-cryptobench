PYTHON ?= python3.12
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: verify test consistency readme strict-json macros freeze
verify: consistency strict-json readme macros
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/verify_claims.py --root .

# The manuscript cites macros only, so a stale macro file is a stale paper.
macros:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_frozen_numbers.py --check

# Re-derive both paired-bootstrap artifacts from the telemetry. Every artifact a
# reader is asked to trust has a generator in the repository; this is it.
freeze:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/freeze_bootstrap.py --all --quiet
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/emit_frozen_numbers.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/render_results_section.py --write

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
