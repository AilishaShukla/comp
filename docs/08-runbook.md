# 08 · Runbook (project-specific commands)

This page lists the exact commands for this project. The build runs on a lab VM with Vitis 2022.2 and the run on a host with a VCK5000 card. Check the shell prompt before every command: the VM and the VCK host are different machines.

## 1. Build files (`kernel/`)

`kernel/Makefile` starts from the course lab Makefile, with these changes (decision D15):

| Item | Lab 3 | This project | Why |
|---|---|---|---|
| Kernel name | `pl` | `conv3x3_l2` (`--kernel`, `--hls.clock`, host string) | Kernel name from [01-architecture](01-architecture.md) §3 |
| Source | `src/pl.cpp`, overwritten per version | `src/pl_$(VERSION).cpp`, chosen with `VERSION=vN` | No copy step, so `pl.cpp` can't drift from `pl_vN.cpp` |
| Build dirs | `_x.hw.<platform>`, `build_dir.hw.<platform>` | `_x.$(VERSION).$(TARGET)`, `build_dir.$(VERSION).$(TARGET)` | One version's reports never overwrite another's |
| Outputs | `pl_hw.xclbin` | `conv3x3_l2_$(VERSION)_$(TARGET).xclbin` | Same reason |
| Includes | `-I.` | `-Isrc -I../golden` (kernel and host) | Kernel and host share `quant.h`; host links the golden model |
| `save` target | – | copies source, xclbin, HLS log, routed util + timing reports to `results/$(VERSION)/` | Replaces the manual copy list |
| `--kernel_frequency 200`, `--hls.clock 300000000` | – | **unchanged** | Guideline Q&A 5 |

`host.exe` uses `group_id(0)` IFM, `(1)` wts, `(2)` bias, `(3)` OFM.

**Optional connectivity file** (`kernel/link.cfg`, add `--config link.cfg` to `XOCCLFLAGS`). Use it only if the default bank mapping fails. **`MC_NOC0` is a placeholder I haven't checked.** Look up the real memory names in the platform guide [R23], or with `platforminfo -p xilinx_vck5000_gen4x8_qdma_2_202220_1` on the VM:

```ini
[connectivity]
sp=conv3x3_l2_1.ifm:MC_NOC0
sp=conv3x3_l2_1.wts:MC_NOC0
sp=conv3x3_l2_1.bias:MC_NOC0
sp=conv3x3_l2_1.ofm:MC_NOC0
```

## 2. Laptop: generate vectors and run local checks

```bash
cd <repo-root>
make test                                        # T1 + T2 on V0 + harness self-test + mutation test
make t2 KERNEL=kernel/src/pl_v1.cpp              # T2 for one version: must print ALL LOCAL CHECKS PASSED
test/check_conv.sh kernel/src/pl_v1.cpp impulse_center   # a single case while debugging
```

## 3. Upload to the VM

Vectors are large (the 540² IFM is 18.7 MB). Upload only what the step needs.

```bash
# from the Mac
rsync -av --exclude 'vectors/' kernel golden results vm1:~/comp/
rsync -av test/vectors/{random_small_1,random_wide,real_frame} vm1:~/comp/vectors/
```

## 4. VM: HLS check, then build (inside tmux)

```bash
tmux new -s comp            # or: tmux attach -t comp
cd ~/comp/kernel && source env.sh
make hls VERSION=v1         # T3, a few minutes; prints the Pipelining / II lines
make all VERSION=v1         # 30+ min. Don't Ctrl+C after "bitstream generation"
```

**Hardware emulation (T4)**, on a small frame only:

```bash
make emconfig                                  # once
make all VERSION=v1 TARGET=hw_emu
export XCL_EMULATION_MODE=hw_emu
./host.exe conv3x3_l2_v1_hw_emu.xclbin ../vectors/random_small_1 1
unset XCL_EMULATION_MODE                       # before any real board run
```

## 5. VCK: run (T5)

```bash
# on the VM
scp host.exe conv3x3_l2_v1_hw.xclbin vck5000:~/comp/
scp -r ../vectors/real_frame vck5000:~/comp/vectors/
# on the VCK
ssh vck5000
source /opt/xilinx/xrt/setup.sh
cd ~/comp && ./host.exe conv3x3_l2_v1_hw.xclbin vectors/real_frame | tee run.log
```

`host.exe <xclbin> <vector_dir> [n_runs=10]` must print `TEST PASSED!` and the metrics block from [03-metrics.md](03-metrics.md) §2. V0 is slow on the full frame: use `n_runs=1`, or `random_wide`.

## 6. Save results right after a passing run

```bash
# on the VM, in ~/comp/kernel
make save VERSION=v1                           # source, xclbin, hls.log, routed util + timing rpts
scp vck5000:~/comp/run.log ../results/v1/run.log
```

Then copy `results/v1/` back to the laptop (`rsync -av vm1:~/comp/results/ results/`), fill in `NOTES.md` and `metrics.json`.

## 7. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `libxrt_coreutil.so.2: cannot open` | XRT not sourced on the VCK | `source /opt/xilinx/xrt/setup.sh` |
| `xrt::kernel` throws "No such compute unit" | Kernel name mismatch | The HLS top name, `--kernel` flag and host string must all be `conv3x3_l2` |
| Host hangs at `wait()` | A dataflow stream is unbalanced | Should have been caught by the T2 stream-empty check. Rerun `check_conv.sh` |
| HLS stuck on "Creating a HLS clock" for more than 10 min | Partitioning is too aggressive | Shrink To×Ti; partition the weights by Cout only |
| WNS < 0 | Long adder tree / wide unroll | Smaller Ti, or add `#pragma HLS bind_op` latency / pipeline registers |
| Board run takes very long on V0 | Expected: V0 is scalar | Use a smaller frame for V0 and note the scaling in the report |
