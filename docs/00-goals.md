# 00 · Purpose and goals

## Purpose

Phones and browsers spend bandwidth and battery downloading full-resolution images. If an image is sent at half resolution (4× fewer pixels) and upscaled on the device by a small neural network, the transfer shrinks. But the upscaling compute then has to run somewhere cheap. A dedicated accelerator is that "somewhere".

This project builds and characterizes the **core compute engine** of such an accelerator: an INT8 convolution kernel for the ESPCN ×2 super-resolution network, on the AMD VCK5000 FPGA, using Vitis HLS.

## Problem statement

> Accelerate the dominant 3×3 convolution layer (L2: 64→32 channels, 87 % of the MACs) of an INT8 ESPCN ×2 network on the VCK5000 PL. Produce **bit-exact** results compared with a software golden model, show measured speedup across optimization versions, and analyze the end-to-end performance of the full upscaling application.

## Goals

| # | Goal | Measured by (see [03-metrics.md](03-metrics.md)) |
|---|---|---|
| G1 | A correct L2 conv kernel on hardware | 0 mismatches vs. golden model on every test vector and on the full 540×540 frame |
| G2 | Large speedup from V0 to the final version | Kernel time, GOPS, speedup per version |
| G3 | Performance we can explain | Analytical cycle model within ~20 % of on-board time for each version |
| G4 | Hardware quality | Closes at 200 MHz, WNS ≥ 0; resources reported per version |
| G5 | Application-level result | End-to-end time per frame and fps, CPU-only vs. CPU + FPGA, compared with the Amdahl bound |
| G6 | Output images are usable | INT8 pipeline PSNR within 0.5 dB of FP32 and at least 1.0 dB above bicubic on Set5 (D19) |

## Stretch goals

- S1: Run L1 and L3 on the same (parameterized) engine, which removes the 7.7× Amdahl cap.
- S2: Fuse L1→L2→L3 into one streaming dataflow kernel (no intermediates in DDR).
- S3: Pack 2 INT8 MACs per DSP.

## Non-goals

Training research, AI Engine programming, MIPI camera/display paths, Linux drivers, Android delegates, and other FPGA boards. These are recorded as future work only.

## Success means

The M3 report can show, for every version: a correct on-board run, a full results table, a prediction from the analytical model that roughly matches the measurement, and an application-level analysis. All of that is backed by artifacts saved under `results/`.
