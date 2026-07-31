# CairnIQ test entrypoints.
#
# The suite is ~2500 tests. Before this file the only way to run it was a bare
# `pytest`, which meant every change — however small — paid for the whole suite,
# serially, with ~500 live Yahoo Finance calls mixed in. These targets make the
# common case (checking the change you just made) fast, and keep the full run
# available for when it matters.
#
#   make test           full suite, parallel — run this before you push
#   make test-changed   only the tests your uncommitted changes can affect
#   make test-since B=x  only the tests changed since git ref x (default: main)
#   make test-map        rebuild the change->test map (after adding test files)
#   make test-serial     full suite, one process (for debugging cross-test state)
#
# Parallelism uses pytest-xdist (requirements-dev.txt). JOBS is 4 rather than
# `auto`: `auto` matches physical cores and over-subscribes this box, where each
# extra worker re-pays the interpreter+import cost for less work (measured: 57.0s
# at -n 4 vs 61.0s at -n auto). Override per-run with `make test JOBS=auto`.
# The offline network guard (tests/net_guard.py) is always on; a test that
# truly needs a socket marks itself @pytest.mark.allow_network.

# Prefer the project virtualenv's interpreter if present, so `make test` works
# whether or not the venv is activated; fall back to python3 on PATH.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PYTEST ?= $(PYTHON) -m pytest
JOBS   ?= 4
B      ?= main

.PHONY: test test-changed test-since test-map test-serial help

help:
	@sed -n 's/^#\{1,\} \{0,1\}//p' Makefile | sed -n '1,20p'

test:
	$(PYTEST) tests/ -q -n $(JOBS)

test-serial:
	$(PYTEST) tests/ -q

# Run only what the working-tree diff can reach. The selector prints selected test
# paths on stdout (diagnostics go to stderr); the literal token "tests" is its
# signal to fall back to the whole suite (a new/unmapped file, a changed conftest).
#
# A selected subset runs SERIALLY on purpose: each xdist worker re-pays the ~5s
# interpreter+import cost, which for the common small selection costs more than it
# saves (measured: a 127-test selection is 8s serial vs 15s at -n auto). The
# whole-suite fallback DOES parallelise, because there the import cost is amortised
# over the whole suite and -n wins (51s -> 30s).
test-changed:
	@sel="$$($(PYTHON) scripts/impacted_tests.py)"; \
	if [ -z "$$sel" ]; then \
		echo "nothing to test"; \
	elif [ "$$sel" = "tests" ]; then \
		$(PYTEST) tests/ -q -n $(JOBS); \
	else \
		echo "$$sel" | xargs $(PYTEST) -q; \
	fi

# Same idea, but for a branch's worth of change (everything since ref B).
test-since:
	@sel="$$($(PYTHON) scripts/impacted_tests.py --base $(B))"; \
	if [ -z "$$sel" ]; then \
		echo "nothing to test"; \
	elif [ "$$sel" = "tests" ]; then \
		$(PYTEST) tests/ -q -n $(JOBS); \
	else \
		echo "$$sel" | xargs $(PYTEST) -q; \
	fi

test-map:
	$(PYTHON) scripts/impacted_tests.py --build-map
