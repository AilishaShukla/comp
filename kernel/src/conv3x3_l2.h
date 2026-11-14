// conv3x3_l2.h - kernel interface for ESPCN layer L2 (3x3 conv, 64 -> 32, ReLU).
//
// This is the contract from docs/01-architecture.md section 3. Every kernel
// version (pl_v0.cpp ... pl_vN.cpp), the local testbench and host.cpp include it.
#pragma once

#include <ap_int.h>

typedef ap_int<512> block_t;  // one AXI beat: 64 x int8, or 16 x int32

const int CIN = 64;
const int COUT = 32;
const int KSIZE = 3;
const int TAPS = KSIZE * KSIZE;          // 9
const int WTS_WORDS = COUT * TAPS;       // 288: word (co*9 + ky*3 + kx) = 64 Cin
const int BIAS_WORDS = COUT / 16;        // 2:   word j = bias[16j .. 16j+15]
const int PIX_PER_OFM_WORD = 512 / (8 * COUT);  // 2

const int MAX_W = 540;                   // line-buffer depth (V2+)
const int MAX_H = 540;

extern "C" void conv3x3_l2(
    const block_t *ifm,   // gmem0: H*W words, word p = 64 Cin of pixel p (HWC)
    const block_t *wts,   // gmem1: 288 words
    const block_t *bias,  // gmem1: 2 words
    block_t *ofm,         // gmem2: H*W/2 words, word q = 32 Cout of pixels 2q, 2q+1
    int H, int W,         // frame size, W even
    int requant_mult,
    int requant_shift);
