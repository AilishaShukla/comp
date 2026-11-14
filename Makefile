# Local (laptop) targets. FPGA builds live in kernel/Makefile and run on the Vitis VM.
#
#   make golden    build the C++ golden model
#   make t1        gate T1: golden C++ == Python integer reference
#   make profile   per-layer CPU timing on a 540x540 frame (B5)
#   make vectors   generate the T2 kernel test vectors (seeded)
#   make t2        gate T2: kernel C++ == golden on every case (KERNEL=kernel/src/pl_vN.cpp)
#   make t2-selftest   the harness must reject a kernel that writes zeros
#   make t2-mutation   inject 11 known bugs into V0; the harness must catch all of them
#   make test      t1 + t2 + self-test + mutation
#   make clean

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror
PYTHON   ?= python3
KERNEL   ?= kernel/src/pl_v0.cpp
BUILD    := build

GOLDEN_SRCS := golden/espcn_golden.cpp golden/run_golden.cpp
GOLDEN_HDRS := golden/espcn_golden.h golden/quant.h

.PHONY: all golden t1 profile vectors t2 t2-selftest t2-mutation test clean

all: golden

golden: $(BUILD)/run_golden $(BUILD)/profile_cpu

$(BUILD)/run_golden: $(GOLDEN_SRCS) $(GOLDEN_HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Igolden $(GOLDEN_SRCS) -o $@

$(BUILD)/profile_cpu: golden/profile_cpu.cpp golden/espcn_golden.cpp $(GOLDEN_HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Igolden golden/profile_cpu.cpp golden/espcn_golden.cpp -o $@

profile: $(BUILD)/profile_cpu
	$(BUILD)/profile_cpu 540 540 3

t1: golden
	$(PYTHON) test/t1_crosscheck.py

vectors: test/vectors/manifest.csv

test/vectors/manifest.csv: model/gen_vectors.py model/quant_ref.py
	$(PYTHON) -m model.gen_vectors

t2: vectors
	test/check_conv.sh $(KERNEL)

t2-selftest: vectors
	@if test/check_conv.sh test/kernels/null_kernel.cpp > $(BUILD)/selftest.log 2>&1; then \
		echo "SELF-TEST FAILED: harness accepted a kernel that writes zeros"; exit 1; \
	else echo "self-test ok: harness rejects the null kernel"; fi

t2-mutation: vectors
	$(PYTHON) test/mutation_test.py

test: t1 t2 t2-selftest t2-mutation

clean:
	rm -rf $(BUILD)
