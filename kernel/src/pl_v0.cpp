// pl_v0.cpp - V0 baseline for ESPCN layer L2 (3x3 conv, 64 -> 32, ReLU).
//
// Straight nested loops with no optimization pragmas: every MAC tap reads its
// IFM and weight words directly from DDR. This is the correctness baseline and
// the "before" point for the speedup table (docs/01-architecture.md section 6).
//
// Only interface pragmas are used: each AXI pointer on its own bundle, as in
// the contract. Weights and bias share gmem1.
#include "conv3x3_l2.h"
#include "quant.h"

extern "C" void conv3x3_l2(const block_t *ifm, const block_t *wts, const block_t *bias, block_t *ofm, int H, int W,
                           int requant_mult, int requant_shift) {
#pragma HLS INTERFACE m_axi port = ifm bundle = gmem0 offset = slave depth = 291600
#pragma HLS INTERFACE m_axi port = wts bundle = gmem1 offset = slave depth = 288
#pragma HLS INTERFACE m_axi port = bias bundle = gmem1 offset = slave depth = 2
#pragma HLS INTERFACE m_axi port = ofm bundle = gmem2 offset = slave depth = 145800
#pragma HLS INTERFACE s_axilite port = H
#pragma HLS INTERFACE s_axilite port = W
#pragma HLS INTERFACE s_axilite port = requant_mult
#pragma HLS INTERFACE s_axilite port = requant_shift
#pragma HLS INTERFACE s_axilite port = return

    const int n_words = H * W / PIX_PER_OFM_WORD;

ofm_word:
    for (int q = 0; q < n_words; q++) {
        block_t out = 0;
    pixel:
        for (int p = 0; p < PIX_PER_OFM_WORD; p++) {
            const int pix = q * PIX_PER_OFM_WORD + p;
            const int y = pix / W;
            const int x = pix % W;
        cout:
            for (int co = 0; co < COUT; co++) {
                const block_t bword = bias[co / 16];
                int32_t acc = (ap_int<32>)bword(32 * (co % 16) + 31, 32 * (co % 16));
            ky:
                for (int ky = 0; ky < KSIZE; ky++) {
                kx:
                    for (int kx = 0; kx < KSIZE; kx++) {
                        const int iy = y + ky - 1;
                        const int ix = x + kx - 1;
                        if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;  // zero padding
                        const block_t xw = ifm[iy * W + ix];
                        const block_t ww = wts[co * TAPS + ky * KSIZE + kx];
                    cin:
                        for (int ci = 0; ci < CIN; ci++) {
                            const ap_int<8> xv = xw(8 * ci + 7, 8 * ci);
                            const ap_int<8> wv = ww(8 * ci + 7, 8 * ci);
                            acc += (int32_t)xv * (int32_t)wv;
                        }
                    }
                }
                const ap_int<8> o = quant::requantize_relu(acc, requant_mult, requant_shift);
                const int bit = 8 * (COUT * p + co);
                out(bit + 7, bit) = o;
            }
        }
        ofm[q] = out;
    }
}
