PYTHON ?= python3.12
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: verify test
verify:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/verify_claims.py --root .

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v
