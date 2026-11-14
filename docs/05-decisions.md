# 05 · Decision log

Add a row whenever a decision changes the architecture, tests or metrics. Never delete rows. If a decision is reversed, mark it superseded and add a new row.

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| D1 | Model is **ESPCN ×2** | 3 plain conv layers + pixel shuffle; fits HLS; all compute at low resolution | FSRCNN-s (needs a transposed-conv engine), QuickSRNet | Accepted |
| D2 | **Y channel only**; chroma upscaled by bicubic | Standard practice in SR papers; 1 input channel | Full RGB (3× L1 work, little gain) | Accepted |
| D3 | **ReLU** instead of tanh | Folds into the INT8 clamp; no lookup table | tanh LUT | Accepted (record it as a model change) |
| D4 | **INT8** weights and activations, INT32 accumulators | DSP-friendly; matches the brief | INT4 (more quality risk), FP (too costly) | Accepted |
| D5 | Accelerate **L2** first | 87 % of MACs | All layers at once (more risk) | Accepted; S1/S2 are stretch |
| D6 | **HWC** layout | Cin = 64 × 8 bit = exactly one 512-bit word per pixel | CHW (strided reads across channels) | Accepted |
| D7 | Shared `quant.h` for golden model and kernel | Removes rounding drift as a source of mismatches | Re-implementing it in each | Accepted |
| D8 | Bit-exact comparison is the correctness gate; PSNR is only a quality metric | PSNR can hide index bugs | Tolerance-based comparison | Accepted |
| D9 | Report T_kernel without host↔device transfers; report transfers separately | Matches the course convention; transfers are counted in the app metric | A single combined time | Accepted |
| D10 | Starting PE array To=32 × Ti=16 (512 MAC/cycle), grown from 128 | ~52 ms per frame predicted; well under the DSP budget | 1,024 MAC/cycle (timing risk) | Tentative; revisit after E1 |
| D11 | Train on **DIV2K** train (800 imgs) directly; skip the T91 first run | Official ETH mirror is reachable and scripted in `model/data.py`; T91 has no reliable official download | T91 first, then DIV2K (original plan in 06 §1) | Accepted 2026-11-13 |
| D12 | Training: LR patch **32×32** (HR 64), batch 64, **L1 loss**, Adam 1e-3 → 1e-4 at 80 %, 100k iterations, flip/rot90 augmentation | 17×17 patches underuse the GPU; L1 is the usual SR loss. Neither changes the network or the kernel contract | 17×17 patches, MSE loss | Accepted 2026-11-13 |
| D13 | CPU baseline for speedups = **faster of two loop orders per layer** (`golden/profile_cpu.cpp`); the golden model stays naive | The naive golden L1 (Cin = 1 innermost) doesn't vectorize and would inflate L1's share to 61 %. A fair baseline makes the FPGA speedup credible | Golden loops as the baseline; hand-written SIMD | Accepted 2026-11-13 |
| D14 | `quantize_multiplier` also accepts scales M ≥ 1 (shift < 31), with the same mult ∈ [2³⁰, 2³¹), shift ∈ [1, 62] contract | Random-weight T1 cases can produce M ≥ 1; the kernel formula is unchanged | Assert M < 1 | Accepted 2026-11-13 |
| D15 | `kernel/Makefile` builds `src/pl_$(VERSION).cpp` into per-version `_x.<ver>.<target>/` and `build_dir.<ver>.<target>/`, with a `make save` target | Removes the "copy pl_vN.cpp over pl.cpp" step and the risk of one build overwriting another version's reports | Lab 3 single `pl.cpp` flow | Accepted 2026-11-13 |
| D16 | Each vector dir carries `shape.txt` (`H W`) | `host.exe` takes only the directory, so H/W can't be passed wrong | H, W on the command line | Accepted 2026-11-13 |
