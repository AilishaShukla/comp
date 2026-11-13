#include "espcn_golden.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "quant.h"

namespace golden {

void conv2d(const int8_t *in, int h, int w, const ConvLayer &L, int8_t *out) {
    const int pad = L.k / 2;
    const int32_t lo = L.relu ? quant::RELU_LO : quant::LINEAR_LO;
    const int32_t hi = L.relu ? quant::RELU_HI : quant::LINEAR_HI;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int co = 0; co < L.cout; co++) {
                int32_t acc = L.b[co];
                for (int ky = 0; ky < L.k; ky++) {
                    const int iy = y + ky - pad;
                    if (iy < 0 || iy >= h) continue;  // zero padding
                    for (int kx = 0; kx < L.k; kx++) {
                        const int ix = x + kx - pad;
                        if (ix < 0 || ix >= w) continue;
                        const int8_t *px = in + ((size_t)iy * w + ix) * L.cin;
                        const int8_t *wt = L.w.data() + ((size_t)(co * L.k + ky) * L.k + kx) * L.cin;
                        for (int ci = 0; ci < L.cin; ci++) acc += (int32_t)wt[ci] * (int32_t)px[ci];
                    }
                }
                out[((size_t)y * w + x) * L.cout + co] = quant::requantize(acc, L.mult, L.shift, lo, hi);
            }
        }
    }
}

void pixel_shuffle2(const int8_t *in, int h, int w, int8_t *out) {
    const int ow = 2 * w;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                    out[(size_t)(2 * y + dy) * ow + (2 * x + dx)] = in[((size_t)y * w + x) * 4 + dy * 2 + dx];
}

Outputs run(const Model &m, const int8_t *y_in, int h, int w) {
    const size_t px = (size_t)h * w;
    Outputs o;
    o.l1.resize(px * m.l1.cout);
    o.l2.resize(px * m.l2.cout);
    o.l3.resize(px * m.l3.cout);
    o.y.resize(px * 4);
    conv2d(y_in, h, w, m.l1, o.l1.data());
    conv2d(o.l1.data(), h, w, m.l2, o.l2.data());
    conv2d(o.l2.data(), h, w, m.l3, o.l3.data());
    pixel_shuffle2(o.l3.data(), h, w, o.y.data());
    return o;
}

template <typename T>
std::vector<T> read_bin(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const size_t bytes = (size_t)f.tellg();
    if (bytes != count * sizeof(T)) {
        std::fprintf(stderr, "error: %s is %zu bytes, expected %zu\n", path.c_str(), bytes, count * sizeof(T));
        std::exit(2);
    }
    std::vector<T> v(count);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)bytes);
    return v;
}

template <typename T>
void write_bin(const std::string &path, const std::vector<T> &data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(T)));
}

template std::vector<int8_t> read_bin<int8_t>(const std::string &, size_t);
template std::vector<int32_t> read_bin<int32_t>(const std::string &, size_t);
template void write_bin<int8_t>(const std::string &, const std::vector<int8_t> &);
template void write_bin<int32_t>(const std::string &, const std::vector<int32_t> &);

static void load_layer(const std::string &dir, const char *name, ConvLayer &L) {
    L.w = read_bin<int8_t>(dir + "/" + name + "_w.bin", (size_t)L.cout * L.k * L.k * L.cin);
    L.b = read_bin<int32_t>(dir + "/" + name + "_b.bin", (size_t)L.cout);
}

Model load_model(const std::string &dir) {
    Model m;
    load_layer(dir, "l1", m.l1);
    load_layer(dir, "l2", m.l2);
    load_layer(dir, "l3", m.l3);
    const std::vector<int32_t> qp = read_bin<int32_t>(dir + "/qparams.bin", 6);
    m.l1.mult = qp[0]; m.l1.shift = qp[1];
    m.l2.mult = qp[2]; m.l2.shift = qp[3];
    m.l3.mult = qp[4]; m.l3.shift = qp[5];
    return m;
}

}  // namespace golden
