.PHONY: setup test demo clean
PY := .venv/bin/python

# uv is the fast path (and can fetch CPython 3.11 itself); plain venv is the fallback.
setup:
	uv venv --python 3.11 .venv || python3.11 -m venv .venv
	uv pip install --python $(PY) -r requirements.txt -e . \
	  || ($(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt -e .)
	@echo "ok: $$($(PY) --version), linkrag installed editable"

test:
	$(PY) -m pytest -q

demo:
	$(PY) scripts/demo.py

clean:
	rm -rf .pytest_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
