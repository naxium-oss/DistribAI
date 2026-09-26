# DistribAI C++ port. Build with zig c++ (portable, no root) or any g++/clang++.
# `make gpu=1` additionally compiles the CUDA kernels when nvcc is available.

# Prefer system g++ when present (CI images); else the pinned zig toolchain.
CXX ?= $(shell command -v g++ 2>/dev/null || echo ../../.toolchain/zig-x86_64-linux-0.16.0/zig\ c++)
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra -I../..
NVCC ?= nvcc
BUILD := ../../build/cpp_port

CORE := core/tensor.hpp
WSPC_CPP := core/workspace.cpp
SANDBOX := sandbox/sandbox.hpp
GPU_H := gpu/gpu_shim.hpp

PARITY := core/parity.hpp core/mt19937.hpp
GOLDEN := tests/golden_seed42.hpp

TARGETS := $(BUILD)/distribai_train $(BUILD)/distribai_bench $(BUILD)/test_parity \
           $(BUILD)/test_golden $(BUILD)/distribai_dualrun

# --- LibTorch live train path (tools/cpp_port/torch) ----------------------
# The port's own engine stays dependency-free; the LibTorch path is the ONLY
# live trainer the grid runs and is built against the shared .venv torch
# install (headers + libs). PyTorch 2.14 requires C++20 and the libstdc++
# ABI, so this target uses $(CXX) with -std=c++20.
# Prefer the shared .venv; fall back to python3 (CI installs torch there).
TORCH_PY ?= $(if $(wildcard ../../.venv/bin/python),../../.venv/bin/python,$(shell command -v python3 2>/dev/null))
TORCH_DIR := $(shell $(TORCH_PY) -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null)
TORCH_INC := $(TORCH_DIR)/include $(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LIB := $(TORCH_DIR)/lib
TORCH_TARGET := $(BUILD)/distribai_torch_train

.PHONY: all clean test test-features test-golden dualrun gpu asan tsan port-check torch torch-test

all: $(TARGETS)
$(BUILD)/distribai_train: apps/train.cpp $(CORE) $(SANDBOX) $(GPU_H) gpu/gpu_shim.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) apps/train.cpp gpu/gpu_shim.cpp -o $@

$(BUILD)/distribai_bench: apps/bench.cpp $(CORE) $(SANDBOX) $(GPU_H) gpu/gpu_shim.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) apps/bench.cpp gpu/gpu_shim.cpp -o $@

$(BUILD)/test_parity: tests/test_parity.cpp $(CORE)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_parity.cpp -o $@

$(BUILD)/test_golden: tests/test_golden.cpp $(PARITY) $(CORE) $(SANDBOX) $(GOLDEN) $(WSPC_CPP)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_golden.cpp $(WSPC_CPP) -o $@

$(BUILD)/distribai_dualrun: apps/dualrun.cpp $(PARITY) $(CORE) $(SANDBOX) $(WSPC_CPP)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) apps/dualrun.cpp $(WSPC_CPP) -o $@

# GPU acceleration layer: only when nvcc exists (no CUDA toolkit on this box).
gpu:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "== building fused CUDA kernels =="; \
		$(NVCC) -O3 -arch=native -c gpu/fused_kernels.cu -o $(BUILD)/fused_kernels.o; \
	else \
		echo "== nvcc not found: GPU layer stays as CPU fallback (fused_kernels.cu kept in tree) =="; \
	fi

test: $(BUILD)/test_parity
	$(BUILD)/test_parity

$(BUILD)/test_features: tests/test_features.cpp $(CORE)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_features.cpp core/workspace.cpp core/envelope.cpp -o $@

test-features: $(BUILD)/test_features
	$(BUILD)/test_features

# LibTorch live train path. Builds only when the shared .venv torch tree is
# present; without it the target prints a skip note (the parity/golden gates
# that do not need torch still run).
torch: $(TORCH_TARGET)

$(TORCH_TARGET): torch/torch_train.cpp torch/json_lite.hpp core/envelope.hpp core/envelope.cpp core/multi_model.hpp sandbox/sandbox.hpp
	@mkdir -p $(BUILD)
	@if [ -z "$(TORCH_DIR)" ]; then \
		echo "torch: SKIP, no torch in $(TORCH_PY) (pip install -e legacy / pip install torch)"; \
	else \
		echo "== building LibTorch live trainer (C++20, ABI=$(shell $(TORCH_PY) -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null)) =="; \
		$(CXX) -std=c++20 -O2 -march=native -Wall -Wextra -I../.. \
			$(addprefix -isystem ,$(TORCH_INC)) \
			torch/torch_train.cpp core/envelope.cpp -o $@ \
			-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB); \
	fi

# LibTorch parity gate: Python-torch vs C++ LibTorch on the same TorchScript
# model + deterministic data (must-match tolerances).
torch-test: $(TORCH_TARGET)
	@if [ -x "$(TORCH_TARGET)" ] && [ -x "$(TORCH_PY)" ]; then \
		../../tools/bench/run_limited.sh $(TORCH_PY) tests/torch_parity.py; \
	else echo "torch-test: SKIP (need torch + $(TORCH_TARGET))"; fi

# --- Native grid (tools/cpp_port/grid) ------------------------------------
# The coordinator persists state in SQLite (runtime/db/schema.sql) and serves the
# worker API, the operator API and the dashboard from one HTTP port. The worker
# runs the LibTorch trainer under the rlimit contract and reports envelopes.
# Both need only libc plus SQLite; the worker adds OpenSSL when it is present so
# an https:// tunnel URL works.
GRID_TARGETS := $(BUILD)/distribai_orch $(BUILD)/distribai_worker
SQLITE_CFLAGS := $(shell pkg-config --cflags sqlite3 2>/dev/null)
SQLITE_LIBS := $(shell pkg-config --libs sqlite3 2>/dev/null || echo -lsqlite3)
HAVE_SQLITE := $(shell test -f /usr/include/sqlite3.h && echo 1)
HAVE_OPENSSL := $(shell test -f /usr/include/openssl/ssl.h && echo 1)
ifeq ($(HAVE_OPENSSL),1)
OPENSSL_FLAGS := -DDISTRIBAI_HAVE_OPENSSL
OPENSSL_LIBS := -lssl -lcrypto
endif

.PHONY: grid grid-test grid-torch-test grid-edge-test grid-regression-test grid-security-test \
        grid-soak-test usage-test grid-all trainer-edge-test suite-store suite-json

grid: $(GRID_TARGETS)

$(BUILD)/distribai_orch: grid/orchestrator.cpp grid/store.cpp grid/store.hpp grid/http.hpp \
                         grid/protocol.hpp grid/base64.hpp torch/json_lite.hpp \
                         core/envelope.cpp core/envelope.hpp \
                         core/multi_model.hpp sandbox/sandbox.hpp
	@mkdir -p $(BUILD)
	@if [ "$(HAVE_SQLITE)" != "1" ]; then \
		echo "grid: SKIP - install libsqlite3-dev to build the coordinator"; \
	else \
		echo "== building the native grid coordinator (SQLite) =="; \
		$(CXX) -std=c++17 -O2 -Wall -Wextra -I../.. $(SQLITE_CFLAGS) \
			grid/orchestrator.cpp grid/store.cpp core/envelope.cpp \
			-o $@ $(SQLITE_LIBS) -lpthread; \
	fi

$(BUILD)/distribai_worker: grid/worker.cpp grid/http_client.hpp grid/http.hpp grid/protocol.hpp \
                           grid/base64.hpp torch/json_lite.hpp sandbox/sandbox.hpp
	@mkdir -p $(BUILD)
	@echo "== building the native grid worker (rlimits$(if $(HAVE_OPENSSL),, no TLS)) =="
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I../.. $(OPENSSL_FLAGS) \
		grid/worker.cpp -o $@ $(OPENSSL_LIBS) -lpthread

# Grid gate: builds a fake trainer, runs a real orchestrator plus two real
# workers on loopback, and checks the job, the aggregate and the ledger.
grid-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/grid_e2e.sh

# Full product slice: translate a plain PyTorch script, submit the bundle, run the
# real LibTorch trainer on two workers, check the aggregate. Skips without torch.
grid-torch-test: $(GRID_TARGETS) $(TORCH_TARGET)
	../../tools/bench/run_limited.sh bash tests/grid_e2e_torch.sh

# Edge cases against the real binaries: bad worker messages, invite/token gates,
# static asset abuse, refused bundles, failing workers, cancelled jobs, restarts.
grid-edge-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/grid_edge.sh

# Regressions for bugs that actually happened (fork limits, shutdown hang, cold
# start, shared node ids, cancelled work, credit counting, abandoned tasks).
grid-regression-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/grid_regression.sh

# Authorization: admin token, invite, session binding, task ownership, leakage.
grid-security-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/grid_security.sh

# Load: many workers, many jobs in a burst, ledger and counters reconciled.
grid-soak-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/grid_soak.sh

# One operator's whole day: run, cancel, restart, resume, read the surfaces.
usage-test: $(GRID_TARGETS) $(BUILD)/grid_fake_trainer
	../../tools/bench/run_limited.sh bash tests/real_usage.sh

# Trainer edge cases driven by real translated fixtures. Skips without torch.
trainer-edge-test: $(TORCH_TARGET)
	../../tools/bench/run_limited.sh bash tests/trainer_edge.sh

# The whole grid and trainer surface, in the order it is worth watching.
grid-all: grid-test grid-edge-test grid-regression-test grid-security-test grid-soak-test \
          usage-test grid-torch-test trainer-edge-test

# Store category: SQLite persistence, claimed and finished work, migrations.
suite-store:
	@mkdir -p $(BUILD)
	@if [ "$(HAVE_SQLITE)" != "1" ]; then \
		echo "suite-store: SKIP - install libsqlite3-dev to build the coordinator store"; \
	else \
		$(CXX) -std=c++17 -O2 -Wall -Wextra -I. -I../.. \
			tests/suite/categories/suite_store.cpp grid/store.cpp core/envelope.cpp \
			-o $(BUILD)/suite_store $(SQLITE_LIBS) -lpthread && \
		../../tools/bench/run_limited.sh $(BUILD)/suite_store; \
	fi

# JSON reader/writer category: header only, no link step.
suite-json:
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I. -I../.. \
		tests/suite/categories/suite_json.cpp -o $(BUILD)/suite_json
	../../tools/bench/run_limited.sh $(BUILD)/suite_json

$(BUILD)/grid_fake_trainer: tests/grid_fake_trainer.cpp core/envelope.cpp core/envelope.hpp
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I../.. tests/grid_fake_trainer.cpp core/envelope.cpp -o $@

# Phase B gate: golden must-match contracts (mission fails on break).
test-golden: $(BUILD)/test_golden
	$(BUILD)/test_golden

# Phase C4 gate: dual-run Python-vs-C++ envelope diff (needs .venv).
dualrun:
	../../tools/bench/run_limited.sh ../../.venv/bin/python tests/compare.py --seeds 42,43,44,0,123,7,999,1,2,3

# Phase D: sanitizers on the C++ core (parity + golden suites).
asan: 
	$(CXX) $(CXXFLAGS) -fsanitize=address -fno-omit-frame-pointer tests/test_golden.cpp $(WSPC_CPP) -o $(BUILD)/test_golden_asan
	$(CXX) $(CXXFLAGS) -fsanitize=address -fno-omit-frame-pointer tests/test_parity.cpp -o $(BUILD)/test_parity_asan
	ASAN_OPTIONS=detect_leaks=1 $(BUILD)/test_golden_asan
	ASAN_OPTIONS=detect_leaks=1 $(BUILD)/test_parity_asan

tsan:
	$(CXX) $(CXXFLAGS) -fsanitize=thread tests/test_parity.cpp -o $(BUILD)/test_parity_tsan
	$(BUILD)/test_parity_tsan

# O8: one-shot build + gates + bench, regenerates runtime/baselines/cpp_port_check_report.md
port-check:
	bash port_check.sh $(ARGS)

clean:
	rm -rf $(BUILD)
