# 09 · Report outline and grading checklist

Each row maps a guideline requirement (the course guideline PDF, kept locally and not redistributed) to the report section where it appears and the artifact it comes from. Tick each one before you submit.

## Proposal (PDF + presentation)

| ✓ | Required content | Section | Source |
|---|---|---|---|
| [ ] | Project scope and target application | 1 | [00-goals.md](00-goals.md), brief §1–2 |
| [ ] | Profiling results (timer-based or literature) | 2 | B5 CPU timing + analytical table (brief §4) |
| [ ] | Why this kernel | 3 | L2 = 87 % of MACs; Amdahl's law |
| [ ] | Expected delivery + how to evaluate | 4 | [00-goals.md](00-goals.md) goals table, [03-metrics.md](03-metrics.md) |
| [ ] | Timeline for the three milestones | 5 | [04-plan.md](04-plan.md) |
| [ ] | Background, related work, tools | 6 | [07-references.md](07-references.md) suggested paragraph |
| [ ] | Supplement (links, papers) | App. | 07-references §G |

## Milestone 1 report

| ✓ | Required content | Source |
|---|---|---|
| [ ] | Target application stated (ESPCN ×2, Y, INT8) | [06-model-and-quantization.md](06-model-and-quantization.md) §1 |
| [ ] | Model architecture, layer by layer (conv and non-conv: ReLU, pixel shuffle) | 06 §1 |
| [ ] | Per-layer kernel information: shapes, MACs, bytes, measured time | Brief §4 + B5 |
| [ ] | Roofline on VCK5000, with L2 and whole-app points | [03-metrics.md](03-metrics.md) §5, R17/R18/R21/R22 |
| [ ] | Quantization quality (T0) | `eval_psnr.py` output |

## Milestone 2 report

| ✓ | Required content | Source |
|---|---|---|
| [ ] | Final code design (architecture + version plan) | [01-architecture.md](01-architecture.md) |
| [ ] | **Hardware emulation passes** (log excerpt) | `results/vN/NOTES.md` T4 |
| [ ] | Optimizations applied so far, with HLS II/depth | `hls.log` excerpts |
| [ ] | Test strategy and results (T1, T2) | [02-test-plan.md](02-test-plan.md) |

## Milestone 3 report.pdf

| ✓ | Section | Required content | Source |
|---|---|---|---|
| [ ] | 1 Introduction | Scope, application, chosen kernel | Proposal, condensed |
| [ ] | 2 Design | Architecture diagrams; pragmas and techniques per version | 01, `results/vN/NOTES.md` |
| [ ] | 3 Results table | Time, GOPS, LUT, LUTAsMem, REG, BRAM, URAM, DSP, Freq, WNS for **each** version | `make_table.py` from `metrics.json` |
| [ ] | 4 Analytical model | Estimated compute + communication cycles per version; vs. on-board; reasons for gaps | 03 §4, NOTES "Model vs. measured" |
| [ ] | 5 Layouts | Vivado implemented-design screenshot per version | `results/vN/layout.png` |
| [ ] | 6 Expected vs. onboard | Where the results differ from expectations, and why | NOTES |
| [ ] | 7 Application level | Full ESPCN: CPU-only vs. CPU+FPGA time, fps, Amdahl bound, PSNR (T6) | 03 §6 |
| [ ] | 8 Related work + conclusion | Position vs. R6, R7, O4, O5 | 07 |

## Submission zip: `project_<BrownID>.zip`

```text
project_<BrownID>.zip
├── report.pdf
├── src/
│   ├── host.cpp
│   ├── quant.h              (+ any shared headers the kernel includes)
│   ├── pl_v0.cpp
│   ├── pl_v1.cpp
│   └── ... pl_vN.cpp
├── v0/
│   ├── pl_hw.xclbin         (guideline example names it cnn.xclbin; keep ours, or rename consistently)
│   ├── impl_1_xilinx_vck5000_gen4x8_qdma_2_202220_1_bb_locked_timing_summary_routed.rpt
│   └── impl_1_full_util_routed.rpt   (+ impl_1_slr_util_routed.rpt)
├── v1/ ...
└── vN/ ...
```

| ✓ | Check |
|---|---|
| [ ] | One folder per version; each xclbin is the one that produced that version's numbers |
| [ ] | The guideline names `impl_1_kernel_util_routed.rpt` / `impl_1_hw_bb_locked_timing_summary_routed.rpt`; our platform's files are named differently. Include ours and note the mapping in the report |
| [ ] | No vectors, build directories or `_x*` folders in the zip |
| [ ] | Every number in report.pdf can be traced to a file in the zip or in `results/` |
