// quant.h - the INT8 requantization contract.
//
// The golden model (golden/espcn_golden.cpp), the host (kernel/src/host.cpp)
// and every kernel version (kernel/src/pl_vN.cpp) include this file, so
// rounding, saturation and ReLU are defined in exactly one place.
// The Python integer reference (model/quant_ref.py) implements the same
// formula independently; gate T1 checks that the two agree bit for bit.
//
// Spec: docs/01-architecture.md section 3, docs/06-model-and-quantization.md section 3.
//
//   prod = (int64) acc * mult
//   out  = (prod + 2^(shift-1)) >> shift        round half up, arithmetic shift
//   out  = clamp(out, lo, hi)                   lo = 0 folds in ReLU
//
// Plain C++ only (no STL), so Vitis HLS can synthesize it.
#pragma once

#include <stdint.h>

namespace quant {

// Output ranges.
const int32_t RELU_LO = 0;      // L1, L2: ReLU folded into the clamp
const int32_t RELU_HI = 127;
const int32_t LINEAR_LO = -128; // L3: no activation
const int32_t LINEAR_HI = 127;

// Rescale an int32 accumulator to int8.
// Preconditions (guaranteed by model/quant_ref.py::quantize_multiplier):
//   2^30 <= mult < 2^31, 1 <= shift <= 62.
// acc and mult are int32, so |acc * mult| < 2^62 and the int64 product cannot overflow.
inline int8_t requantize(int32_t acc, int32_t mult, int32_t shift,
                         int32_t lo, int32_t hi) {
    int64_t prod = (int64_t)acc * (int64_t)mult;
    int64_t rounded = (prod + ((int64_t)1 << (shift - 1))) >> shift;
    if (rounded < lo) rounded = lo;
    if (rounded > hi) rounded = hi;
    return (int8_t)rounded;
}

inline int8_t requantize_relu(int32_t acc, int32_t mult, int32_t shift) {
    return requantize(acc, mult, shift, RELU_LO, RELU_HI);
}

}  // namespace quant
