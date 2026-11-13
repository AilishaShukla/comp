# 08 · Runbook (project-specific commands)

This page lists the exact commands for this project. The build runs on a lab VM with Vitis 2022.2 and the run on a host with a VCK5000 card. Check the shell prompt before every command: the VM and the VCK host are different machines.

## 1. One-time setup: changes from the Lab 3 build files

`kernel/Makefile` and `kernel/env.sh` start from the course lab build files, with these changes.

| File | Lab 3 | This project | Why |
|---|---|---|---|
| `Makefile` `KERNEL_FLAGS` | `--kernel pl ... --hls.clock 300000000:pl` | `--kernel conv3x3_l2 -I. -I../golden --hls.clock 300000000:conv3x3_l2` | Kernel name from [01-architecture](01-architecture.md) §3; include `quant.h` |
| `Makefile` `$(KERNEL_XO)` deps | `./src/pl.cpp` | `./src/pl.cpp ../golden/quant.h` | Rebuild when the rounding contract changes |
| `Makefile` `HOST_SRCS` | `./src/host.cpp` | `./src/host.cpp` + `-I../golden` in `CXXFLAGS` | Host shares `compare.h` / `quant.h` |
| `host.cpp` | `xrt::kernel(device, uuid, "pl")` | `"conv3x3_l2"` | Must match the HLS top name |
| `host.cpp` buffers | all `krnl.group_id(1)` | `group_id(0)` IFM, `(1)` wts, `(2)` bias, `(3)` OFM | Arg index → its own `gmem` bundle |
| `--kernel_frequency 200` | – | **unchanged** | Guideline Q&A 5 |

Keep the Makefile's 300 MHz HLS clock and 200 MHz link clock exactly as given.

**Optional connectivity file** (`kernel/link.cfg`, add `--config link.cfg` to `XOCCLFLAGS`). Use it only if the default bank mapping fails. **`MC_NOC0` is a placeholder I haven't checked.** Look up the real memory names in the platform guide [R23], or with `platforminfo -p xilinx_vck5000_gen4x8_qdma_2_202220_1` on the VM:

```ini
[connectivity]
sp=conv3x3_l2_1.ifm:MC_NOC0
sp=conv3x3_l2_1.wts:MC_NOC0
sp=conv3x3_l2_1.bias:MC_NOC0
sp=conv3x3_l2_1.ofm:MC_NOC0
```

## 2. Mac: generate vectors and run local checks

```bash
cd <repo-root>
python3 model/gen_vectors.py --out test/vectors          # seeded, rebuilds every T2 case
./test/check_conv.sh kernel/src/pl_v1.cpp all            # T2: must print ALL LOCAL CHECKS PASSED
./test/check_conv.sh kernel/src/pl_v1.cpp impulse_center # a single case while debugging
```

## 3. Upload to the VM

Vectors are large (the 540² IFM is 18.7 MB). Upload only what the step needs.

```bash
# from the Mac
rsync -av --exclude 'vectors/' comp/{kernel,golden} vm1:~/comp/
rsync -av comp/test/vectors/{random_small,real_frame} vm1:~/comp/vectors/
```

## 4. VM: HLS check, then build (inside tmux)

```bash
tmux new -s comp            # or: tmux attach -t comp
cd ~/comp/kernel && source env.sh
cp src/pl_v1.cpp src/pl.cpp
make hls                    # T3, a few minutes
grep -E "Pipelining result|II|constraints" _x.hw.xilinx_vck5000_gen4x8_qdma_2_202220_1/pl.log
make all                    # 30+ min. Don't Ctrl+C after "bitstream generation"
```

**Hardware emulation (T4)**, on a small frame only:

```bash
make hw_emu && make all TARGET=hw_emu
export XCL_EMULATION_MODE=hw_emu
./host.exe pl_hw_emu.xclbin ../vectors/random_small
unset XCL_EMULATION_MODE    # before any real board run
```

## 5. VCK: run (T5)

```bash
# on the VM
scp host.exe pl_hw.xclbin vck5000:~/comp/
scp -r ../vectors/real_frame vck5000:~/comp/vectors/
# on the VCK
ssh vck5000
source /opt/xilinx/xrt/setup.sh
cd ~/comp && ./host.exe pl_hw.xclbin vectors/real_frame | tee run.log
```

`host.exe <xclbin> <vector_dir> [n_runs=10]` must print `TEST PASSED!` and the metrics block from [03-metrics.md](03-metrics.md) §2.

## 6. Save results right after a passing run

```bash
# on the VM, in ~/comp/kernel
V=v1
R=build_dir.hw.xilinx_vck5000_gen4x8_qdma_2_202220_1/reports/link/imp
mkdir -p ../results/$V
cp src/pl.cpp ../results/$V/pl_$V.cpp
cp pl_hw.xclbin host.exe ../results/$V/
cp $R/impl_1_full_util_routed.rpt $R/impl_1_slr_util_routed.rpt \
   $R/impl_1_xilinx_vck5000_gen4x8_qdma_2_202220_1_bb_locked_timing_summary_routed.rpt ../results/$V/
cp _x.hw.xilinx_vck5000_gen4x8_qdma_2_202220_1/pl.log ../results/$V/hls.log
scp vck5000:~/comp/run.log ../results/$V/run.log
cp ../results/TEMPLATE_NOTES.md ../results/$V/NOTES.md      # then fill it in
```

Then copy `results/$V/` back to the Mac (`rsync -av vm1:~/comp/results/ comp/results/`) and fill in `metrics.json`.

## 7. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `libxrt_coreutil.so.2: cannot open` | XRT not sourced on the VCK | `source /opt/xilinx/xrt/setup.sh` |
| `xrt::kernel` throws "No such compute unit" | Kernel name mismatch | The HLS top name, `--kernel` flag and host string must all be `conv3x3_l2` |
| Host hangs at `wait()` | A dataflow stream is unbalanced | Should have been caught by the T2 stream-empty check. Rerun `check_conv.sh` |
| HLS stuck on "Creating a HLS clock" for more than 10 min | Partitioning is too aggressive | Shrink To×Ti; partition the weights by Cout only |
| WNS < 0 | Long adder tree / wide unroll | Smaller Ti, or add `#pragma HLS bind_op` latency / pipeline registers |
| Board run takes very long on V0 | Expected: V0 is scalar | Use a smaller frame for V0 and note the scaling in the report |
