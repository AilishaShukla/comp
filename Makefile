# Local (laptop) targets. FPGA builds live in kernel/Makefile and run on the Vitis VM.
#
#   make golden    build the C++ golden model
#   make t1        gate T1: golden C++ == Python integer reference
#   make profile   per-layer CPU timing on a 540x540 frame (B5)
#   make clean

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror
PYTHON   ?= python3
BUILD    := build

GOLDEN_SRCS := golden/espcn_golden.cpp golden/run_golden.cpp
GOLDEN_HDRS := golden/espcn_golden.h golden/quant.h

.PHONY: all golden t1 profile clean

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

clean:
	rm -rf $(BUILD)
