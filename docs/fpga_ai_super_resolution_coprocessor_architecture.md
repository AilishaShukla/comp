# INT8 Super-Resolution CNN Accelerator on the VCK5000

**Course:** ENGN 2911X Final Project (self-designed track) · **Board:** AMD VCK5000 · **Tools:** Vitis / Vitis HLS 2022.2, XRT

---

## 1. Project statement

Design, build and measure an HLS accelerator for the **3×3 convolution layer that dominates a ×2 ESPCN super-resolution network**, running on the VCK5000 programmable logic. The kernel takes INT8 feature maps and weights and returns INT8 outputs that **match a quantized software reference bit for bit**. A host program runs the full network (CPU layers + FPGA kernel) so that performance can be reported for both the kernel and the whole application.

**Motivation, in one line:** mobile apps could download low-resolution images and upscale them on the device, which saves bandwidth. A dedicated low-power accelerator is the natural place for that work. This project builds and characterizes the core of such an accelerator. Section 10 keeps the wider product vision as context only.

---

## 2. Scope

| In scope (graded deliverable) | Out of scope (future-work context only) |
|---|---|
| ESPCN ×2, luminance (Y) channel, INT8 weights/activations | Training a new model or quantization-aware-training research |
| HLS kernel for the conv layer(s) on VCK5000 PL | AI Engine (AIE) array implementation |
| `host.cpp` that runs the full network and checks results | MIPI-CSI/DSI camera or display paths |
| V0 → V*n* optimization versions with measured results | Linux DMA drivers, Android LiteRT/NNAPI delegates |
| Roofline, analytical cycle model, application-level analysis | Other boards (Lattice CrossLink-NX, Kria, Agilex) |
| Hardware emulation (M2) and on-board bitstream (M3) | Content-adaptive bilinear/CNN routing |

---

## 3. Target application and model choice

**Recommendation: ESPCN ×2 on the Y channel.**

| Model | Pros | Cons for this project |
|---|---|---|
| **ESPCN** | Only 3 conv layers; all work happens at low resolution; output is a cheap pixel shuffle (just a reindex). Weights fit easily on chip. | Lower quality than larger models. That is fine, because the project is about the accelerator. |
| FSRCNN-s | Small; well known on FPGAs | Its last layer is a 9×9 **transposed** conv, which is awkward and inefficient in HLS (many zero-inserted MACs, or a separate deconv engine) |
| QuickSRNet | Designed to quantize well for mobile | Fewer public reference implementations; more layer types to support |

ESPCN layers (from the original paper, r = 2, 1 input channel):

| Layer | Op | Shape (k×k, Cin→Cout) | Activation |
|---|---|---|---|
| L1 | conv | 5×5, 1 → 64 | tanh / ReLU* |
| L2 | conv | 3×3, 64 → 32 | tanh / ReLU* |
| L3 | conv | 3×3, 32 → r²=4 | none |
| PS | pixel shuffle | 4 ch → 1 ch at 2× resolution | none |

\*Use ReLU (or clipped ReLU) instead of tanh so INT8 quantization stays simple. Record this as a model change.

**Reference model:** train or download a PyTorch ESPCN, apply post-training INT8 quantization, then export the weights and a **bit-accurate C++ golden model**. The golden model is the correctness oracle for the local testbench (`test/`) and for `host.cpp`.

---

## 4. Workload profile (analytical; confirm with CPU timers for the proposal)

Input 540×540 Y frame (291,600 pixels) → 1080×1080 output.

| Layer | MAC / pixel | MACs / frame | Share |
|---|---:|---:|---:|
| L1 5×5, 1→64 | 1,600 | 0.47 G | 7.6 % |
| **L2 3×3, 64→32** | **18,432** | **5.37 G** | **87.0 %** |
| L3 3×3, 32→4 | 1,152 | 0.34 G | 5.4 % |
| Pixel shuffle | 0 (reindex) | – | ~0 % |
| **Total** | 21,184 | **6.18 GMAC = 12.35 GOP** | |

For a 960×540 → 1920×1080 frame, the total is **21.96 GOP**.

**Kernel selection:** L2 is 87 % of the compute. Accelerating only L2 caps the application speedup at about 7.7× (Amdahl's law). Accelerating all three conv layers removes that cap. Plan: **L2 first** (the required kernel), then generalize the engine to L1 and L3 if time allows (Section 7, V4).

The proposal still needs **measured** profiling. Time each layer of the C++ golden model with `std::chrono` on the host CPU, as the guideline asks, and put that table next to this one.

---

## 5. Platform and roofline inputs

| Parameter | Value | Source / note |
|---|---|---|
| Kernel clock | 200 MHz | Fixed by course Makefile |
| DSP58 slices (VC1902) | ≈1,968 | **Verify** against the VCK5000 data sheet |
| Off-chip memory | DDR4 | **Look up** peak bandwidth for the roofline |
| Total weights (all layers, INT8) | ≈21 KB | Fits entirely in on-chip BRAM/registers |

**Compute roof:** 1 INT8 MAC per DSP per cycle gives ≈1,968 × 200 MHz ≈ 394 GMAC/s ≈ **0.79 TOPS**. That is an upper bound; a realistic design will use a fraction of it. Packing 2 INT8 MACs per DSP could roughly double it (report this only if it is actually implemented).

**Arithmetic intensity:**
- *Layer-by-layer through DDR:* about 57 MB of traffic per 540² frame (dominated by the 64-channel and 32-channel intermediates) → ≈215 OP/byte.
- *Fully fused, streaming:* about 1.5 MB (input + output only) → ≈8,400 OP/byte.

Both points sit far to the right of any realistic DDR ridge point. **The design is compute-bound**, so the optimization effort belongs in the MAC array and II, not in memory bandwidth. Wide ports still matter so that loads don't serialize against compute.

---

## 6. Accelerator architecture (HLS)

This is the same dataflow shape as Lab 3 V4:

```text
            DDR (gmem0)                     DDR (gmem1)
                │ 512-bit                         ▲ 512-bit
                ▼                                 │
           ┌─────────┐   stream   ┌───────────────────────┐  stream  ┌─────────┐
           │ readIFM │──────────▶ │ line buffer (3 rows)  │────────▶ │ writeOFM│
           └─────────┘            │ + 3×3 window regs     │          └─────────┘
                                  └──────────┬────────────┘               ▲
           weights (gmem2) ──▶ on-chip       ▼                            │
             loaded once     W[32][64][3][3] ┌─────────────────────┐      │
                                             │ PE array            │──────┘
                                             │ To × Ti MACs/cycle  │  + bias, ReLU,
                                             │ (unroll Cout × Cin) │    requantize
                                             └─────────────────────┘
```

Key design decisions:

- **Line buffers:** keep 2 previous rows × W × Cin in BRAM plus a 3×3×Cin window in registers. One new pixel vector enters per cycle, and each input pixel is read from DDR only once.
- **Parallelism:** unroll over output channels (To) and input channels (Ti). Partition the weight array and window to match. Start with To = 32, Ti = 16, which is 512 MACs/cycle.
- **Reduction II:** accumulating across Cin and the 3×3 taps is a loop-carried dependency, which is exactly the guideline's "Final II = 7" trap. Use an unrolled adder tree within a cycle and keep partial sums in INT32 registers so the pipelined loop achieves II = 1.
- **Data types:** `ap_int<8>` for activations and weights, `ap_int<32>` for accumulators. Requantize with a fixed-point multiply + shift and clamp to [−128, 127] (or [0, 127] after ReLU).
- **Packing:** 64 INT8 values per 512-bit word. Derive every wide index from the scalar index, then divide by elements-per-word (the Lab 2 lesson).
- **Pixel shuffle** is a pure address remap. Do it in `writeOFM` or on the host.

**Expected L2 performance (To × Ti = 512):** 5.37 GMAC / 512 ≈ 10.5 M cycles → **≈52 ms per 540² frame** at 200 MHz (≈102 GOPS). This is the baseline for the analytical model. Doubling the MAC array to 1,024 roughly halves it, if DSPs and timing allow.

---

## 7. Version plan

| Version | Change | Purpose |
|---|---|---|
| V0 | Straightforward nested-loop 3×3 conv, scalar ports, no pragmas | Baseline, and a correctness reference on hardware |
| V1 | On-chip weight/IFM buffers, `pipeline II=1`, partition + unroll over Cin/Cout, reduction fix | Main compute speedup |
| V2 | 512-bit ports, line-buffer streaming (no IFM re-reads) | Remove load stalls, cut BRAM use |
| V3 | `dataflow` with `hls::stream` stages (read → window → PE → write) | Overlap memory and compute |
| V4 *(stretch)* | Same engine runs L1 and L3 (configurable K/Cin/Cout), or a fused L1→L2→L3 pipeline | Raise the Amdahl cap at application level |

After each version, record: on-board time, GOPS, the full resource row, Freq, WNS, the HLS II/depth log lines, and the predicted cycles from the model. Save the `.xclbin` and the reports before the next build (see [08-runbook.md](08-runbook.md) §6).

---

## 8. Evaluation

| Criterion | How it is measured | Pass condition |
|---|---|---|
| Correctness | Kernel output vs. C++ INT8 golden model | Bit-exact (0 mismatches) |
| Image quality | PSNR/SSIM of the full pipeline vs. FP32 model and bicubic upscaling on Set5/Set14 | INT8 within ~0.3 dB of FP32 and above bicubic |
| Kernel performance | Host timer around the kernel call; GOPS = 2·MACs / time | Each version beats the previous one; analytical model within ~10–20 % |
| Application performance | End-to-end time per frame: CPU-only golden model vs. CPU + FPGA | Report speedup and frames/s, and explain the gap from the Amdahl bound |
| Implementation | Routed util + timing reports | 200 MHz, WNS ≥ 0 |

---

## 9. Milestone mapping

| Milestone | Deliverables from this plan |
|---|---|
| **Proposal** | Sections 1–4, measured CPU profiling table, reason for choosing the kernel, timeline, related work (Section 10) |
| **M1** | Final model + quantized weights + golden model; layer-by-layer profile; roofline using Section 5 with verified board numbers |
| **M2** | V0–V2 (or V3) code, local conv testbench (`test/`) passing, **hardware emulation passing**, HLS II/depth analysis |
| **M3** | Bitstreams for all versions, layout screenshots, results table, analytical vs. measured comparison, application-level analysis, submission zip |

**Tooling to build early:** a local testbench for conv (`test/tb_conv.cpp`). It should load the golden weights and inputs, call the kernel on the CPU, and compare against the golden output. This keeps the ~1-second check loop from the labs.

---

## 10. Background, related work, and long-term vision

These sources motivate the project. **None of them is part of the deliverable**, and the claims below should be cited properly or softened in the proposal.

- **ESPCN** (Shi et al., CVPR 2016): sub-pixel convolution; all compute happens at low resolution. *Primary reference.*
- **FSRCNN** (Dong et al., ECCV 2016): the shrink/expand design; used for comparison in Section 3.
- **QuickSRNet** (Qualcomm, 2023): an SR network designed for mobile quantization.
- **FPGA SR accelerators** in the literature use line-buffered streaming conv with INT8/INT4 quantization. Cite specific papers and their reported fps/resource numbers instead of general claims.
- **Commercial context:** discrete display/visual processors in phones (e.g., Pixelworks), low-power MIPI FPGAs (e.g., Lattice CrossLink-NX), and client-side upscaling (e.g., NVIDIA RTX Video Super Resolution). These show that dedicated upscaling hardware is a real product category.

**Long-term product vision (not in scope):** a low-power coprocessor next to a mobile ARM SoC with two modes. (A) An app-callable upscaler reached through a DMA driver and a LiteRT delegate. (B) An inline upscaler on the MIPI camera/display path. This project's kernel is the compute core both modes would share.

---

## 11. Risks and corrections to the earlier draft

| Earlier claim | Assessment | Action |
|---|---|---|
| "< 5 ms per 540→1080 image", "< 8 ms per 1080p frame" | Would need about 2.5–2.7 TOPS sustained, over 3× the VCK5000 PL DSP peak at 200 MHz | Drop it. Set targets from the model in Section 6 (~50 ms per 540² frame for L2 at 512 MAC/cycle) |
| "50–75 % data savings" | 2× upscaling means 4× fewer pixels, but compressed-byte savings depend on the codec and content | Rephrase as "4× fewer pixels transmitted", or cite a measurement |
| "Content-adaptive dispatch cuts dynamic power up to 40 %" | No source given | Remove, or cite a paper |
| "1080p/4K at 60+ fps on modest FPGA fabric" | Overstated without a specific citation | Cite a specific design and its device |
| Platform: Lattice / Kria / Agilex | The course board is the VCK5000 | Keep only as future work |
| Target: "FSRCNN-s **or** ESPCN, 2× **and** 3×" | Too broad for one semester | Fix on ESPCN ×2 |

**Main technical risks:**
1. The reduction loop doesn't reach II = 1. Mitigation: an adder tree plus partial sums, checked in the HLS log after `make hls`.
2. Partitioning that is too large stalls HLS or fails timing. Mitigation: grow To×Ti step by step (128 → 256 → 512).
3. Golden-model mismatch from rounding during requantization. Mitigation: define the rounding rule once in a shared header used by both the golden model and the kernel.
4. Hardware emulation is slow on full frames. Mitigation: use a reduced frame size for emulation, and the full 540² frame on the board.
