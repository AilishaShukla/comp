# 07 · References

Every claim in the proposal and reports should cite an ID from here. Status: ✅ = details checked against the source when this list was made (Nov 2026); ⚠️ = number or detail still to confirm from the full text before quoting it.

**How to use:** "Used for" says which doc or report section each reference supports. When citing a performance number from a paper, read the original PDF first; do not quote from this table.

---

## A. Super-resolution models

| ID | Reference | Used for | Status |
|---|---|---|---|
| R1 | W. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," CVPR 2016. [arXiv:1609.05158](https://arxiv.org/abs/1609.05158) | **Primary model** (ESPCN, pixel shuffle). Layer shapes in 01/06 | ✅ |
| R2 | C. Dong, C. C. Loy, X. Tang, "Accelerating the Super-Resolution Convolutional Neural Network," ECCV 2016. [arXiv:1608.00367](https://arxiv.org/abs/1608.00367) | FSRCNN / FSRCNN-s comparison; why not deconv (D1). FSRCNN-s runs at 24 fps on a generic CPU | ✅ |
| R3 | G. Berger et al. (Qualcomm AI Research), "QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms," 2023. [arXiv:2303.04336](https://arxiv.org/abs/2303.04336) | Mobile, quantization-robust SR; 1080p ×2 in 2.2 ms on a phone NPU (motivation and comparison point) | ✅ |
| R4 | P. Navarrete Michelini, Y. Lu, X. Jiang, "edge-SR: Super-Resolution For The Masses," WACV 2022. [arXiv:2108.10335](https://arxiv.org/abs/2108.10335) · code [github.com/pnavarre/eSR](https://github.com/pnavarre/eSR) | One-layer SR at the extreme low-cost end; quality-vs-speed discussion | ✅ |
| R5 | A. Ignatov et al., "Real-Time Quantized Image Super-Resolution on Mobile NPUs, Mobile AI 2021 Challenge: Report," CVPRW 2021. [arXiv:2105.07825](https://arxiv.org/abs/2105.07825) · 2022 follow-up [arXiv:2211.05910](https://arxiv.org/abs/2211.05910) | Evidence that INT8 SR is practical on mobile accelerators; Full-HD runtimes | ⚠️ runtimes |

## B. FPGA / hardware SR accelerators (related work)

| ID | Reference | Relevance | Status |
|---|---|---|---|
| R6 | Y. Kim, J.-S. Choi, M. Kim, "A Real-Time Convolutional Neural Network for Super-Resolution on FPGA With Applications to 4K UHD 60 fps Video Services," IEEE TCSVT 29(8), 2019. DOI [10.1109/TCSVT.2018.2864321](https://doi.org/10.1109/TCSVT.2018.2864321) | **Closest prior work.** Line-by-line processing, no DRAM frame buffer, depthwise-separable conv, quantized; FHD→4K at 60 fps | ✅ (⚠️ resource / PSNR numbers) |
| R7 | J.-W. Chang, K.-W. Kang, S.-J. Kang, "An Energy-Efficient FPGA-based Deconvolutional Neural Networks Accelerator for Single Image Super-Resolution," IEEE TCSVT 2018. [arXiv:1801.05997](https://arxiv.org/abs/1801.05997) | FSRCNN on FPGA; line-buffer sizing analysis; deconv handling; 92.7 GOP/W | ✅ |
| R8 | Y. Kim et al., "A CNN-Based Multi-Scale Super-Resolution Architecture on FPGA for 4K/8K UHD Applications," MMM 2020. [record](https://aminer.org/pub/5e5e193593d709897ce5bd2b/a-cnn-based-multi-scale-super-resolution-architecture-on-fpga-for-k) | Shares hardware across ×2/×3/×4; motivates a parameterized engine (stretch S1) | ⚠️ |
| R9 | G. Li, Z. Liu, F. Li, J. Cheng, "Block Convolution: Towards Memory-Efficient Inference of Large-Scale CNNs on FPGA," IEEE TCAD 2021. [arXiv:2105.08937](https://arxiv.org/abs/2105.08937) | Keeping intermediates on chip; relevant to layer fusion (stretch S2) | ✅ |
| R9b | Y. Cao et al., "Real-Time Super-Resolution System of 4K-Video Based on Deep Learning," 2021. [arXiv:2107.05307](https://arxiv.org/abs/2107.05307) | Video SR (EGVSR) at 4K / 29.6 fps; context only | ✅ |

## C. Quantization

| ID | Reference | Used for | Status |
|---|---|---|---|
| R10 | B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) | **INT8 weights/activations, INT32 bias, fixed-point requantization**: basis of 06 §2–3 | ✅ |
| R11 | Google, gemmlowp: low-precision GEMM library (fixed-point multiplier and rounding shift). [github.com/google/gemmlowp](https://github.com/google/gemmlowp) | Reference implementation of `quantize_multiplier` / rounding right shift | ✅ |
| R12 | Xilinx, Brevitas: PyTorch quantization-aware training library. [github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas) | Fallback if post-training quantization fails T0 (QAT) | ✅ |

## D. Datasets

| ID | Reference | Used for |
|---|---|---|
| R13 | E. Agustsson, R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," CVPRW 2017. [paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.html) · [DIV2K site](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | DIV2K training set (800 train / 100 val) |
| R14 | J. Yang, J. Wright, T. Huang, Y. Ma, "Image Super-Resolution via Sparse Representation," IEEE TIP 2010 | T91 training set |
| R15 | M. Bevilacqua et al., "Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding," BMVC 2012 | Set5 evaluation |
| R16 | R. Zeyde, M. Elad, M. Protter, "On Single Image Scale-Up Using Sparse-Representations," Curves and Surfaces 2010 | Set14 evaluation |

## E. Performance modeling and FPGA CNN design

| ID | Reference | Used for | Status |
|---|---|---|---|
| R17 | S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," CACM 52(4), 2009. DOI [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | Roofline method (M1, 03 §5) | ✅ |
| R18 | C. Zhang et al., "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks," FPGA 2015. DOI [10.1145/2684746.2689060](https://doi.org/10.1145/2684746.2689060) | **Roofline applied to FPGA CNN loop tiling/unrolling**: template for the To×Ti design-space exploration | ✅ |
| R19 | Y. Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference," FPGA 2017 · [github.com/Xilinx/finn](https://github.com/Xilinx/finn) | Per-layer streaming dataflow architecture (compare with V3) | ✅ repo |
| R20 | T. Aarrestad et al., "Fast convolutional neural networks on FPGAs with hls4ml," MLST 2021. [arXiv:2101.05108](https://arxiv.org/abs/2101.05108) · [github.com/fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml) | HLS line-buffer conv implementation patterns; reuse-factor trade-off (≈ our Ti) | ✅ |

## F. Platform and tools

| ID | Reference | Used for | Status |
|---|---|---|---|
| R21 | AMD, *Versal AI Core Series Data Sheet: Overview* (DS950). VC1902: **1,968 DSP engines** | Roofline compute roof | ✅ count; cite DS950 directly |
| R22 | AMD, *VCK5000 Versal Development Card* product brief / UG1428. 16 GB off-chip memory; **102.4 GB/s** listed as off-chip bandwidth ([brief](https://www.mouser.com/datasheet/2/903/xilinx_vck5000_product_brief-2634597.pdf)) | Roofline memory roof | ⚠️ sources disagree (one table says 70 GB/s "DDR" vs. 102.4 "total"): confirm in UG1428 |
| R23 | AMD, *VCK5000 Gen4x8 QDMA Vitis Platform* user guide. [link](https://xilinx.github.io/Alveo-Versal-Platforms/alveoversalplatforms/build/html/docs/user-guide/ug_vck5000_gen4x8_qdma.html) | Platform memory banks and connectivity for the `.cfg` | ✅ |
| R24 | AMD, *Vitis High-Level Synthesis User Guide* (UG1399), 2022.2 edition | Pragmas: pipeline, unroll, array_partition, dataflow, stream | ✅ |
| R25 | [github.com/Xilinx/Vitis-HLS-Introductory-Examples](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples) | Minimal HLS patterns (line buffer, dataflow, streams) | ✅ |
| R26 | [github.com/Xilinx/Vitis_Accel_Examples](https://github.com/Xilinx/Vitis_Accel_Examples) | XRT host code, multi-bundle `m_axi` examples | ✅ |
| R27 | [github.com/Xilinx/Vitis_Libraries](https://github.com/Xilinx/Vitis_Libraries) (vision L1: `xf::cv` window/line-buffer utilities, resize) | Line-buffer reference; possible bicubic baseline on FPGA | ✅ |

## G. Open-source implementations to study

| ID | Project | What to take from it | Caveat |
|---|---|---|---|
| O1 | [Lornatang/ESPCN-PyTorch](https://github.com/Lornatang/ESPCN-PyTorch) | Training/eval scripts, pretrained weights | Check license; uses tanh by default |
| O2 | [clibdev/ESPCN-PyTorch](https://github.com/clibdev/ESPCN-PyTorch) | Fork updated for recent PyTorch, ONNX export | Same |
| O3 | [yjn870/ESPCN-pytorch](https://github.com/yjn870/ESPCN-pytorch) | Small, readable reference; h5 data prep for T91/Set5 | Old (PyTorch 1.0) |
| O4 | [amiq-consulting/image-upscaling-CNN](https://github.com/amiq-consulting/image-upscaling-CNN) · [write-up](https://www.consulting.amiq.com/2021/08/23/how-to-accelerate-an-image-upscaling-cnn-on-fpga-using-hls/) | **FSRCNN in Vivado HLS** on Zynq-7000 (PYNQ-Z1). Compares FPGA, bicubic and CPU; reports up to 41× vs. Python CPU on 64×64 inputs | Older Vivado HLS 2019.1; small board |
| O5 | [Deploying SR algorithms on VCK5000 (Hackster)](https://www.hackster.io/freya-wang/deploying-super-resolution-algorithms-on-vck5000-be9c49) | **Same board.** WDSR via Vitis AI (DPU), with pixel shuffle on the CPU. A useful "off-the-shelf DPU" comparison point | Vitis AI, not HLS; no numbers in text |
| O6 | [W. Buchta et al., Hardware Accelerated Super Resolution (WPI MQP)](https://buchtawill.github.io/documents/Hardware_Accelerated_Super_Resolution_MQP.pdf) | Student project: FSRCNN + bilinear in Vitis HLS on ZCU102. Comparable scope; good report structure example | ⚠️ check code availability |
| O7 | [Lornatang/FSRCNN-PyTorch](https://github.com/Lornatang/FSRCNN-PyTorch) | FSRCNN baseline if a comparison model is needed | – |
| O8 | [ychfan/wdsr](https://github.com/ychfan/wdsr) | WDSR model used in O5 | – |
| O9 | [Qualcomm QuickSRNet models (Hugging Face)](https://huggingface.co/qualcomm/QuickSRNetMedium) | Pre-quantized mobile SR models; parameter counts (Medium ≈55 K) | – |

## H. Commercial context (motivation only; cite product pages, not as technical evidence)

- Pixelworks visual processors (X-series), discrete display co-processors in phones.
- Lattice CrossLink-NX / sensAI, low-power FPGAs with MIPI D-PHY.
- NVIDIA RTX Video Super Resolution, client-side upscaling of streamed video.

---

## Suggested related-work paragraph (proposal)

> Efficient SR networks such as FSRCNN [R2] and ESPCN [R1] perform most computation at low resolution, and ESPCN's sub-pixel (pixel-shuffle) output avoids the transposed convolutions that complicate FPGA mapping [R7]. INT8 integer-only inference [R10] has been shown to preserve SR quality on mobile accelerators [R3, R5]. Prior FPGA SR accelerators use line-buffered streaming so that no full frame is stored in DRAM [R6, R7], reaching FHD→4K at 60 fps with custom lightweight networks [R6]. We follow the roofline-guided loop-unrolling methodology of [R18] to size a 3×3 convolution engine for the dominant ESPCN layer on the VCK5000, and compare against an HLS FSRCNN implementation [O4] and a Vitis AI DPU deployment on the same board [O5].
