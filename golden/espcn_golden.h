// espcn_golden.h - bit-exact C++ reference for the INT8 ESPCN x2 network.
//
// This is the correctness oracle for the kernel testbench (T2), hardware
// emulation (T4) and the on-board run (T5). It is deliberately simple:
// plain nested loops, no tiling, no tricks.
//
// Layouts follow docs/01-architecture.md section 3 and docs/06 section 4:
//   activations  HWC, int8
//   weights      [Cout][K][K][Cin], int8
//   bias         [Cout], int32
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace golden {

struct ConvLayer {
    ConvLayer(int k_, int cin_, int cout_, bool relu_ = true) : k(k_), cin(cin_), cout(cout_), relu(relu_) {}
    int k = 0;              // kernel size (5 or 3)
    int cin = 0;
    int cout = 0;
    std::vector<int8_t> w;  // cout * k * k * cin
    std::vector<int32_t> b; // cout
    int32_t mult = 0;
    int32_t shift = 0;
    bool relu = true;
};

struct Model {
    ConvLayer l1{5, 1, 64};
    ConvLayer l2{3, 64, 32};
    ConvLayer l3{3, 32, 4, /*relu=*/false};
};

// Same-padded KxK convolution + bias + requantization, HWC in and out.
void conv2d(const int8_t *in, int h, int w, const ConvLayer &layer, int8_t *out);

// PixelShuffle(2): in is H x W x 4 (HWC), out is 2H x 2W.
//   out[2y+dy][2x+dx] = in[y][x][dy*2 + dx]
void pixel_shuffle2(const int8_t *in, int h, int w, int8_t *out);

struct Outputs {
    std::vector<int8_t> l1, l2, l3, y;
};

// Run the full network on an H x W int8 Y plane.
Outputs run(const Model &m, const int8_t *y_in, int h, int w);

// Load l{1,2,3}_{w,b}.bin and qparams.bin from dir (docs/06 section 4).
Model load_model(const std::string &dir);

// Raw little-endian binary I/O. read_bin exits with a message on a size mismatch.
template <typename T>
std::vector<T> read_bin(const std::string &path, size_t count);
template <typename T>
void write_bin(const std::string &path, const std::vector<T> &data);

}  // namespace golden
