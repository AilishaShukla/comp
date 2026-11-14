// null_kernel - matches the conv3x3_l2 interface but writes zeros.
// `make t2-selftest` runs the harness on it and expects every case that has a
// nonzero output to FAIL, which proves the harness can catch a wrong kernel.
#include "conv3x3_l2.h"

extern "C" void conv3x3_l2(const block_t *, const block_t *, const block_t *, block_t *ofm, int H, int W, int,
                           int) {
    for (int q = 0; q < H * W / PIX_PER_OFM_WORD; q++) ofm[q] = 0;
}
