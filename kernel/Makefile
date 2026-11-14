# Vitis build for the conv3x3_l2 kernel on the VCK5000. Runs on the lab VM.
#
# Adapted from the Xilinx Vitis example Makefile (Copyright 2021 Xilinx Inc.,
# Apache License 2.0, http://www.apache.org/licenses/LICENSE-2.0).
#
#   make hls     VERSION=v1               HLS only (T3): check the II/depth log
#   make all     VERSION=v1               xclbin + host.exe (30+ min, run in tmux)
#   make all     VERSION=v0 TARGET=hw_emu hardware emulation (T4); run `make emconfig` once
#   make save    VERSION=v1               copy xclbin, routed reports and logs to ../results/v1/
#   make host                             host.exe only
#
# Each version builds from src/pl_$(VERSION).cpp into its own _x / build_dir, so
# one version's reports are never overwritten by another's.
# Clock settings are fixed by the course (300 MHz HLS target, 200 MHz link).

ECHO     := @echo
VERSION  ?= v0
TARGET   ?= hw
PLATFORM := xilinx_vck5000_gen4x8_qdma_2_202220_1
KERNEL   := conv3x3_l2

SRC        := src/pl_$(VERSION).cpp
TEMP_DIR   := ./_x.$(VERSION).$(TARGET)
BUILD_DIR  := ./build_dir.$(VERSION).$(TARGET)
KERNEL_XO  := $(TEMP_DIR)/$(KERNEL).xo
XSA_OBJ    := $(KERNEL)_$(VERSION)_$(TARGET).xsa
XCLBIN     := $(KERNEL)_$(VERSION)_$(TARGET).xclbin
HLS_LOG    := $(TEMP_DIR)/$(KERNEL).log
EXECUTABLE := host.exe
HOST_SRCS  := src/host.cpp ../golden/espcn_golden.cpp
KERNEL_DEPS := $(SRC) src/conv3x3_l2.h ../golden/quant.h

CXXFLAGS := -std=c++17 -O2 -Wall -Wno-deprecated-declarations
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -I../golden -Isrc
LDFLAGS  := -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

XOCCFLAGS    := --platform $(PLATFORM) -t $(TARGET) -s -g
KERNEL_FLAGS := --kernel $(KERNEL) -Isrc -I../golden --hls.clock 300000000:$(KERNEL)
XOCCLFLAGS   := --kernel_frequency 200 --platform $(PLATFORM) -t $(TARGET) -s -g

REPORTS := $(BUILD_DIR)/reports/link/imp
RESULTS := ../results/$(VERSION)

.PHONY: help all build hls host emconfig save clean cleanall

help:
	$(ECHO) "make hls|all|save VERSION=vN [TARGET=hw|hw_emu]; make host; make emconfig; make clean"

all: build host

build: $(XCLBIN)

hls: $(KERNEL_XO)
	@grep -E "Pipelining result|Final II|loop constraints" $(HLS_LOG) || true

$(KERNEL_XO): $(KERNEL_DEPS)
	@test -f $(SRC) || (echo "missing $(SRC)" && exit 1)
	mkdir -p $(TEMP_DIR)
	v++ $(XOCCFLAGS) $(KERNEL_FLAGS) --temp_dir $(TEMP_DIR) -c -o $@ $(SRC) | tee $(HLS_LOG)

$(XSA_OBJ): $(KERNEL_XO)
	mkdir -p $(BUILD_DIR)
	v++ -l $(XOCCLFLAGS) --temp_dir $(BUILD_DIR) -o $@ $^ | tee $(BUILD_DIR)/link.log

$(XCLBIN): $(XSA_OBJ)
	v++ -p -t $(TARGET) -f $(PLATFORM) $^ -o $@ --package.boot_mode=ospi
	$(ECHO) "### $(XCLBIN) done"

host: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_SRCS) src/conv3x3_l2.h ../golden/espcn_golden.h ../golden/quant.h
	$(CXX) -o $@ $(HOST_SRCS) $(CXXFLAGS) $(LDFLAGS)

# Hardware emulation config (once per VM). Then: export XCL_EMULATION_MODE=hw_emu
emconfig:
	emconfigutil --platform $(PLATFORM)

# Copy everything a version's report needs into results/vN/ (docs/08-runbook.md section 6).
save:
	mkdir -p $(RESULTS)
	cp $(SRC) $(RESULTS)/
	cp $(XCLBIN) $(RESULTS)/
	cp $(HLS_LOG) $(RESULTS)/hls.log
	cp $(REPORTS)/impl_1_full_util_routed.rpt $(REPORTS)/impl_1_slr_util_routed.rpt $(RESULTS)/
	cp $(REPORTS)/impl_1_$(PLATFORM)_bb_locked_timing_summary_routed.rpt $(RESULTS)/
	test -f $(RESULTS)/NOTES.md || cp ../results/TEMPLATE_NOTES.md $(RESULTS)/NOTES.md
	$(ECHO) "saved to $(RESULTS). Add run.log from the VCK and fill in NOTES.md + metrics.json"

clean:
	rm -rf $(EXECUTABLE) *.xclbin *.xsa *.ltx *.info *.link_summary *.compile_summary
	rm -rf .Xil .ipcache *.log *.jou *.wdb *.wcfg xrt.run_summary fpga_l2_out.bin

cleanall: clean
	rm -rf _x.* build_dir.* emconfig.json .run
