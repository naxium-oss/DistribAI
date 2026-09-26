# DistribAI contributor shortcuts.
#
# The NATIVE C++ PORT (tools/cpp_port) is the primary codebase: the default
# target and every non-legacy target below drive it. The Python-era stack
# (services_python, worker, client, tests, ...) is FROZEN under legacy/ with
# its own packaging (legacy/pyproject.toml), drive it through the legacy-*
# targets, which simply run the original commands with cwd=legacy.

# Prefer the shared .venv; fall back to python3 (CI installs into system python).
VENV_PY := $(wildcard .venv/bin/python)
PY := $(if $(VENV_PY),$(VENV_PY),$(shell command -v python3 2>/dev/null))

.PHONY: help all clean test test-golden test-features dualrun gpu asan tsan \
        port-check suite suite-quick mutation-check lint typecheck ci \
        torch torch-test translate translate-test \
        grid grid-test grid-edge-test grid-regression-test grid-security-test \
        grid-soak-test usage-test trainer-edge-test grid-torch-test grid-all \
        screenshots suite-json suite-store \
        grid-expose grid-join grid-providers gridlink-test \
        legacy-install legacy-lint legacy-format-check legacy-typecheck \
        legacy-test-unit legacy-test-security legacy-test-integration \
        legacy-test legacy-coverage legacy-proto legacy-ci

help:
	@echo "Primary (native C++ port under tools/cpp_port):"
	@echo "  all (default) build the port binaries"
	@echo "  test            port parity + golden + feature gates"
	@echo "  port-check      one-shot build + gates + bench report"
	@echo "  suite / suite-quick / mutation-check / asan / tsan / gpu / dualrun"
	@echo "  lint / typecheck  ruff + mypy scoped to the root surface"
	@echo "LibTorch live train path (ONLY train path):"
	@echo "  torch           build build/cpp_port/distribai_torch_train (C++20 + .venv torch)"
	@echo "  torch-test      python-torch vs C++-LibTorch parity gate"
	@echo "  translate       translate a Python trainer: make translate ARGS=\"trainer.py --out jobs/x\""
	@echo "  translate-test  fail-closed translator tests (pytest)"
	@echo "Native grid (C++ coordinator + workers):"
	@echo "  grid            build build/cpp_port/distribai_orch and distribai_worker"
	@echo "  grid-test       end-to-end grid gate: coordinator + two workers + one job"
	@echo "  grid-torch-test translate a PyTorch script and run it on real LibTorch workers"
	@echo "  grid-edge-test  refuse bad bundles, bad sessions, bad assets, failing workers"
	@echo "  grid-regression-test  the grid bugs that must not come back"
	@echo "  grid-security-test    admin token, invite, session binding, task ownership"
	@echo "  grid-soak-test  many workers and many jobs in one burst"
	@echo "  usage-test      run, cancel, restart, resume, then read the surfaces"
	@echo "  trainer-edge-test     trainer edge cases from real translated fixtures"
	@echo "  grid-all        every grid and trainer gate"
	@echo "  suite-json / suite-store  the suite's JSON and SQLite store categories"
	@echo "  screenshots     refresh docs/assets/dashboard-*.png from a live grid"
	@echo "Grid access (free tunnels + join links):"
	@echo "  grid-expose     publish the grid: make grid-expose ARGS=\"--provider cloudflare --port 50051\""
	@echo "  grid-join       join from here: make grid-join LINK='distribai://join?...' ARGS=\"--exec\""
	@echo "  grid-providers  show which tunnel providers are ready"
	@echo "  gridlink-test   tunnel/link/join tests (pytest)"
	@echo "Legacy (frozen Python stack under legacy/):"
	@echo "  legacy-install  pip install -e legacy (into .venv)"
	@echo "  legacy-lint / legacy-format-check / legacy-typecheck"
	@echo "  legacy-test-unit / -security / -integration / legacy-test"
	@echo "  legacy-coverage / legacy-proto / legacy-ci"

# ---------------------------------------------------------------- port ----
all:
	$(MAKE) -C tools/cpp_port all

clean:
	$(MAKE) -C tools/cpp_port clean

test:
	$(MAKE) -C tools/cpp_port test test-golden test-features

test-golden:
	$(MAKE) -C tools/cpp_port test-golden

test-features:
	$(MAKE) -C tools/cpp_port test-features

dualrun:
	$(MAKE) -C tools/cpp_port dualrun

gpu:
	$(MAKE) -C tools/cpp_port gpu

# ------------------------------------------------------------- libtorch ----
# LibTorch is the ONLY live train path; the frozen Python stack never runs jobs.
torch:
	$(MAKE) -C tools/cpp_port torch

torch-test:
	$(MAKE) -C tools/cpp_port torch-test

translate-test:
ifdef PY
	$(PY) -m pytest tools/trainer_translate/tests -q -p no:cacheprovider
else
	@echo "translate-test: no python found, skipped"
endif

translate:
ifdef PY
	PYTHONPATH=tools $(PY) -m trainer_translate.translate $(ARGS)
else
	@echo "translate: no python found"; exit 1
endif

# ------------------------------------------------------------- gridlink ----
# Free Cloudflare quick tunnel / optional ngrok; local bind stays the default.
grid-expose:
ifdef PY
	PYTHONPATH=tools $(PY) -m gridlink expose $(ARGS)
else
	@echo "grid-expose: no python found"; exit 1
endif

# The link carries `&` query separators, so pass it in LINK (quoted for the
# shell here) and keep ARGS for flags:
#   make grid-join LINK='distribai://join?...&...' ARGS="--ephemeral --exec"
grid-join:
ifdef PY
	@set -- $(if $(LINK),'$(LINK)',) $(ARGS); PYTHONPATH=tools $(PY) -m gridlink join "$$@"
else
	@echo "grid-join: no python found"; exit 1
endif

grid-providers:
ifdef PY
	PYTHONPATH=tools $(PY) -m gridlink providers
else
	@echo "grid-providers: no python found"; exit 1
endif

gridlink-test:
ifdef PY
	$(PY) -m pytest tools/gridlink/tests -q -p no:cacheprovider
else
	@echo "gridlink-test: no python found, skipped"
endif

# ---------------------------------------------------------- native grid ----
# The coordinator and workers are C++ binaries; state lives in SQLite and job
# bundles live under runtime/grid/. Needs libsqlite3-dev to build the coordinator.
grid:
	$(MAKE) -C tools/cpp_port grid

grid-test:
	$(MAKE) -C tools/cpp_port grid-test

# The whole product slice: translate a PyTorch script, then run it on the real
# LibTorch trainer across two native workers. Skips itself without torch.
grid-torch-test:
	$(MAKE) -C tools/cpp_port grid-torch-test

# Edge cases and regressions for the grid, plus the trainer's own edge cases.
grid-edge-test:
	$(MAKE) -C tools/cpp_port grid-edge-test

grid-regression-test:
	$(MAKE) -C tools/cpp_port grid-regression-test

grid-security-test:
	$(MAKE) -C tools/cpp_port grid-security-test

grid-soak-test:
	$(MAKE) -C tools/cpp_port grid-soak-test

usage-test:
	$(MAKE) -C tools/cpp_port usage-test

trainer-edge-test:
	$(MAKE) -C tools/cpp_port trainer-edge-test

suite-json:
	$(MAKE) -C tools/cpp_port suite-json

suite-store:
	$(MAKE) -C tools/cpp_port suite-store

grid-all:
	$(MAKE) -C tools/cpp_port grid-all

# Regenerate docs/assets/dashboard-*.png from a real grid (needs Chrome).
screenshots:
	bash tools/cpp_port/grid/screenshots.sh

asan:
	$(MAKE) -C tools/cpp_port asan

tsan:
	$(MAKE) -C tools/cpp_port tsan

# O8: one-shot build + gates + bench, regenerates runtime/baselines/cpp_port_check_report.md
port-check:
	bash tools/cpp_port/port_check.sh $(ARGS)

suite:
	bash tools/cpp_port/tests/suite/run_suite.sh

suite-quick:
	bash tools/cpp_port/tests/suite/run_suite.sh --quick

mutation-check:
	bash tools/cpp_port/tests/suite/selfcheck_mutations.sh

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy tools/cpp_port --python-version=3.12 --ignore-missing-imports

ci: lint typecheck test suite-json port-check suite-quick mutation-check translate-test \
    grid-test grid-edge-test grid-regression-test grid-security-test grid-soak-test \
    usage-test grid-torch-test trainer-edge-test gridlink-test

# -------------------------------------------------------------- legacy ----
# The Python-era stack keeps its original config (legacy/pyproject.toml);
# every target below is the original root command with cwd=legacy.
# Manual only: no default or CI target runs these. Invoke them when you are
# fixing the frozen stack, not as a gate on ordinary changes.

legacy-install:
ifdef VENV_PY
	$(VENV_PY) -m pip install -e legacy --no-build-isolation || $(VENV_PY) -m pip install -e legacy
else
	@echo "legacy-install: no .venv/bin/python found, skipped (create a venv first)"
endif

legacy-lint:
	cd legacy && ruff check .

legacy-format-check:
	cd legacy && ruff format --check .

legacy-typecheck:
	cd legacy && mypy services_python/ worker/src/ --ignore-missing-imports --no-strict-optional --follow-imports=silent --python-version=3.12

legacy-test-unit:
	cd legacy && pytest tests/unit -v

legacy-test-security:
	cd legacy && pytest tests/security -v

legacy-test-integration:
	cd legacy && pytest tests/integration -v --timeout=900

legacy-test: legacy-test-unit legacy-test-security

legacy-coverage:
	cd legacy && pytest tests/unit tests/security --cov=services_python --cov=worker --cov-report=term-missing --cov-report=xml

legacy-proto:
	cd legacy && python -m grpc_tools.protoc --python_out=worker/src/distribai_proto --grpc_python_out=worker/src/distribai_proto -I proto proto/distribai.proto
	cd legacy && python -c "from pathlib import Path; p=Path('worker/src/distribai_proto/distribai_pb2_grpc.py'); p.write_text(p.read_text().replace('import distribai_pb2 as distribai__pb2', 'from . import distribai_pb2 as distribai__pb2'))"

legacy-ci: legacy-lint legacy-typecheck legacy-test
