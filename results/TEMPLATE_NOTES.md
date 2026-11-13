# vN · <one-line summary of what changed>

Copy this file to `results/vN/NOTES.md`. Fill in sections 1–2 **before** coding (design first), and sections 3–6 as results arrive.

## 1. Change from the previous version

- What changed (pragmas, loop structure, buffers, ports):
- Why, i.e. which bottleneck in v(N−1) it targets:
- Architecture doc row updated? [ ] [01-architecture.md §6](../../docs/01-architecture.md)

## 2. Prediction (before building)

| Loop / stage | Trip count | Expected II | Expected depth | Cycles |
|---|---:|---:|---:|---:|
| | | | | |
| **Total predicted** | | | | |

Predicted T_kernel = cycles × 5 ns = ____ s → ____ GOPS
Expected resources (rough): DSP ____, BRAM ____

## 3. Gate checklist ([02-test-plan.md §4](../../docs/02-test-plan.md))

```text
[ ] T2 all cases pass on Mac          date:
[ ] T3 make hls; II / depth recorded  date:
[ ] T4 hw_emu TEST PASSED             date:   (required for v0 + final)
[ ] T5 board TEST PASSED on 540×540   date:
[ ] metrics.json filled
[ ] xclbin + util + timing rpts + hls.log + run.log saved here
[ ] pl_vN.cpp saved
```

## 4. HLS log (T3)

Paste the relevant `Pipelining result` lines and the loop-constraint status:

```text

```

| Loop | Final II | Depth | Matches prediction? |
|---|---:|---:|---|
| | | | |

## 5. Measured (T5)

| Metric | Value |
|---|---|
| T_kernel median (s) | |
| min / max (s) | |
| GOPS | |
| T_h2d / T_d2h (s) | |
| LUT / LUTAsMem / REG | |
| BRAM / URAM / DSP | |
| Freq (MHz) / WNS (ns) | |

## 6. Model vs. measured

- Measured cycles = T_kernel × 200 MHz = ______
- Model error = ______ %
- Explanation of the gap (burst stalls, missed II, fill/drain, host overhead, ...):

## 7. Lessons / surprises

- 
