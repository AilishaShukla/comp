# 03 · Metrics: definitions and how to measure them

Every number in the reports must trace back to a saved artifact in `results/vN/`. Define the number here first, then measure it.

## 1. Constants

| Symbol | Value | Note |
|---|---|---|
| H, W | 540, 540 | Full-run frame (L2 operates at low resolution) |
| MAC_L2 | 9 · 64 · 32 · H · W = **5,374,771,200** | L2 multiply-accumulates per frame |
| OPS_L2 | 2 · MAC_L2 = **10.75 G** | 1 MAC = 2 ops (the course GOPS convention) |
| MAC_app | 21,184 · H · W = 6.18 G | All three conv layers |
| f | 200 MHz | Kernel clock; must equal the routed frequency |

## 2. Performance metrics

| Metric | Definition | How to measure |
|---|---|---|
| **T_kernel** (s) | Median wall time of `run()` + `wait()`, over N = 10 runs after 1 warm-up | `std::chrono::high_resolution_clock` in `host.cpp` around the run/wait pair only |
| T_h2d, T_d2h (s) | Buffer sync time to and from the device | Timed separately; reported but **not** included in GOPS |
| **GOPS** | OPS_L2 / T_kernel / 1e9 | Computed from T_kernel |
| Speedup | T_kernel(V0) / T_kernel(Vn), plus step-to-step Vn−1 / Vn | From the table |
| Cycles (measured) | T_kernel · f | Used to compare against the model in §4 |
| PE efficiency | GOPS / (2 · MACs_per_cycle · f / 1e9) | Shows how close the design gets to its own peak |
| Run-to-run spread | (max − min) / median over the N runs | Flag it if above 5 % |

All of this comes from the host's stdout. Save it verbatim as `results/vN/run.log`.

## 3. Hardware metrics

| Metric | Source file (in `build_dir.hw.xilinx_vck5000_gen4x8_qdma_2_202220_1/reports/link/imp/`) | Field |
|---|---|---|
| LUT, LUTAsMem, REG, BRAM, URAM, DSP | `impl_1_full_util_routed.rpt`, `impl_1_slr_util_routed.rpt` | "Used Resources" row for the kernel |
| % utilization | same | Used / Available |
| WNS (ns) | `impl_1_xilinx_vck5000_gen4x8_qdma_2_202220_1_bb_locked_timing_summary_routed.rpt` | Design Timing Summary, first column. Must be ≥ 0 |
| Freq (MHz) | same | Clock `clkwiz_aclk_kernel_00_clk_out1`: 1000 / period |
| Layout | Vivado, implemented design | Screenshot per version (M3) |

HLS estimates (`pl_csynth.rpt`) are logged for the model in §4, but **never** reported as final resources.

## 4. Analytical performance model

For each pipelined loop nest, read `Final II` and `Depth` from the HLS log:

```text
cycles(loop) = trip_count · II + (Depth − 1)
```

| Version | Predicted kernel cycles | Main term |
|---|---|---|
| V0 | Sum over the scalar loop nests (expect ≫ 10⁹) | DDR access latency per MAC |
| V1 | load(IFM tile) + compute + store, run **in sequence** per tile | Sum of the stages |
| V2 | H·W·(9·64/Ti) · II + fill | Compute: about 10.5 M cycles at Ti = 16 |
| V3 | max(read, compute, write) + pipeline fill | Compute-bound if all stages run at II = 1 |

```text
model error = (T_kernel · f − predicted_cycles) / predicted_cycles
```

Target |error| ≤ 20 %. If it is larger, explain why in the report (AXI burst stalls, DDR refresh, a stream that is too shallow, a loop that missed its II, host overhead in T_kernel).

## 5. Roofline (M1)

| Quantity | Formula |
|---|---|
| Compute roof (GOPS) | 2 · DSP_count · f (one INT8 MAC per DSP per cycle; double it only if packing is implemented) |
| Memory roof | BW_DDR · arithmetic intensity |
| Arithmetic intensity, L2 alone | OPS_L2 / (IFM + OFM + weights bytes) = 10.75 G / 28.0 MB ≈ **384 OP/B** |
| Arithmetic intensity, app layer-by-layer | 12.35 G / ~57 MB ≈ 215 OP/B |

Plot every version's measured GOPS as a point. Points should move up toward the compute roof as the versions improve. DSP_count and BW_DDR come from the VCK5000 data sheet; cite it.

## 6. Application and quality metrics

| Metric | Definition | How |
|---|---|---|
| T_app_cpu | End-to-end Y-plane pipeline (L1 + L2 + L3 + shuffle), CPU-only golden model | `golden/profile_cpu.cpp`, median of 10 runs, per layer |
| T_app_fpga | Same pipeline with L2 on the FPGA, **including** T_h2d and T_d2h | `host.exe` |
| App speedup | T_app_cpu / T_app_fpga | vs. Amdahl bound 1 / (0.13 + 0.87/s) |
| fps | 1 / T_app_fpga | |
| PSNR (dB) | 10·log10(255² / MSE) on Y, border-cropped by 2 px | `model/eval_psnr.py`, Set5 / Set14 |
| SSIM | Standard SSIM on Y | Same |

Note that the CPU timing machine is the VCK5000 host's CPU. Record its model (`lscpu`) in the report.

## 7. `results/vN/metrics.json` schema

```json
{
  "version": "v2",
  "frame": [540, 540],
  "t_kernel_s": {"median": 0.0, "min": 0.0, "max": 0.0, "runs": 10},
  "t_h2d_s": 0.0, "t_d2h_s": 0.0,
  "gops": 0.0,
  "predicted_cycles": 0, "measured_cycles": 0, "model_error_pct": 0.0,
  "resources": {"LUT": 0, "LUTAsMem": 0, "REG": 0, "BRAM": 0, "URAM": 0, "DSP": 0},
  "freq_mhz": 200.0, "wns_ns": 0.0,
  "hls_loops": [{"name": "", "trip": 0, "ii": 0, "depth": 0}],
  "test": "PASSED", "mismatches": 0
}
```

A small `model/make_table.py` turns all `metrics.json` files into the report table and the charts in [10-visualization.md](10-visualization.md) View 5, so numbers never get re-typed by hand.
