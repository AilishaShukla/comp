// profile_cpu - per-layer CPU timing of the golden INT8 ESPCN (work-plan chunk B5).
//
//   profile_cpu [H W] [runs] [model_dir]
//
// Defaults: 540 x 540, 3 runs, seeded random weights (timing does not depend
// on the weight values). Prints the median time and share for each layer, next
// to the analytical MAC share (7.6 / 87.0 / 5.4 %), for the proposal and M1.
//
// Two CPU implementations are timed:
//   golden    the reference loops (Cin innermost). Simple on purpose, but L1
//             (Cin = 1) cannot vectorize, so its share is misleading.
//   cout-inner the same math with Cout innermost and weights stored
//             [K][K][Cin][Cout]. Vectorizes L1 well, but L3 (Cout = 4) badly.
//   best      the faster of the two for each layer. This is the CPU baseline
//             used for application-level speedups.
// Both are checked against each other, bit for bit, on every run.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "espcn_golden.h"
#include "quant.h"

using Clock = std::chrono::steady_clock;

static void randomize(golden::ConvLayer &L, std::mt19937 &rng) {
    std::uniform_int_distribution<int> wd(-127, 127), bd(-1000, 1000);
    L.w.resize((size_t)L.cout * L.k * L.k * L.cin);
    L.b.resize(L.cout);
    for (auto &v : L.w) v = (int8_t)wd(rng);
    for (auto &v : L.b) v = bd(rng);
    L.mult = 1 << 30;
    L.shift = 40;
}

// Cout-innermost conv. Same result as golden::conv2d, bit for bit.
struct FastLayer {
    const golden::ConvLayer *L;
    std::vector<int8_t> wt;  // [K][K][Cin][Cout]
    explicit FastLayer(const golden::ConvLayer &l) : L(&l), wt(l.w.size()) {
        for (int co = 0; co < l.cout; co++)
            for (int t = 0; t < l.k * l.k; t++)
                for (int ci = 0; ci < l.cin; ci++)
                    wt[((size_t)t * l.cin + ci) * l.cout + co] = l.w[((size_t)co * l.k * l.k + t) * l.cin + ci];
    }
};

static void conv2d_fast(const int8_t *in, int h, int w, const FastLayer &F, int8_t *out) {
    const golden::ConvLayer &L = *F.L;
    const int pad = L.k / 2, co_n = L.cout;
    const int32_t lo = L.relu ? quant::RELU_LO : quant::LINEAR_LO;
    const int32_t hi = L.relu ? quant::RELU_HI : quant::LINEAR_HI;
    std::vector<int32_t> acc(co_n);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int co = 0; co < co_n; co++) acc[co] = L.b[co];
            for (int ky = 0; ky < L.k; ky++) {
                const int iy = y + ky - pad;
                if (iy < 0 || iy >= h) continue;
                for (int kx = 0; kx < L.k; kx++) {
                    const int ix = x + kx - pad;
                    if (ix < 0 || ix >= w) continue;
                    const int8_t *px = in + ((size_t)iy * w + ix) * L.cin;
                    const int8_t *wr = F.wt.data() + (size_t)(ky * L.k + kx) * L.cin * co_n;
                    for (int ci = 0; ci < L.cin; ci++) {
                        const int32_t v = px[ci];
                        const int8_t *wc = wr + (size_t)ci * co_n;
                        for (int co = 0; co < co_n; co++) acc[co] += v * (int32_t)wc[co];
                    }
                }
            }
            int8_t *o = out + ((size_t)y * w + x) * co_n;
            for (int co = 0; co < co_n; co++) o[co] = quant::requantize(acc[co], L.mult, L.shift, lo, hi);
        }
    }
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    const int h = argc > 2 ? std::atoi(argv[1]) : 540;
    const int w = argc > 2 ? std::atoi(argv[2]) : 540;
    const int runs = argc > 3 ? std::atoi(argv[3]) : 3;

    golden::Model m;
    if (argc > 4) {
        m = golden::load_model(argv[4]);
    } else {
        std::mt19937 rng(1);
        randomize(m.l1, rng);
        randomize(m.l2, rng);
        randomize(m.l3, rng);
    }

    std::mt19937 rng(2);
    std::uniform_int_distribution<int> px(0, 127);
    std::vector<int8_t> y_in((size_t)h * w);
    for (auto &v : y_in) v = (int8_t)px(rng);

    const size_t n = (size_t)h * w;
    std::vector<int8_t> l1(n * 64), l2(n * 32), l3(n * 4), y(n * 4);
    std::vector<int8_t> f1(n * 64), f2(n * 32), f3(n * 4), fy(n * 4);
    const FastLayer F1(m.l1), F2(m.l2), F3(m.l3);
    std::vector<double> tg[4], tf[4];
    const auto secs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    for (int r = 0; r < runs; r++) {
        Clock::time_point t[5];
        t[0] = Clock::now();
        golden::conv2d(y_in.data(), h, w, m.l1, l1.data()); t[1] = Clock::now();
        golden::conv2d(l1.data(), h, w, m.l2, l2.data());   t[2] = Clock::now();
        golden::conv2d(l2.data(), h, w, m.l3, l3.data());   t[3] = Clock::now();
        golden::pixel_shuffle2(l3.data(), h, w, y.data());  t[4] = Clock::now();
        for (int i = 0; i < 4; i++) tg[i].push_back(secs(t[i], t[i + 1]));

        t[0] = Clock::now();
        conv2d_fast(y_in.data(), h, w, F1, f1.data());      t[1] = Clock::now();
        conv2d_fast(f1.data(), h, w, F2, f2.data());        t[2] = Clock::now();
        conv2d_fast(f2.data(), h, w, F3, f3.data());        t[3] = Clock::now();
        golden::pixel_shuffle2(f3.data(), h, w, fy.data()); t[4] = Clock::now();
        for (int i = 0; i < 4; i++) tf[i].push_back(secs(t[i], t[i + 1]));

        if (f1 != l1 || f2 != l2 || f3 != l3 || fy != y) {
            std::fprintf(stderr, "error: baseline output differs from golden\n");
            return 1;
        }
    }

    const char *names[4] = {"L1 conv5x5 1->64", "L2 conv3x3 64->32", "L3 conv3x3 32->4", "pixel shuffle"};
    const double macs_px[4] = {1600, 18432, 1152, 0};
    std::printf("CPU profile, %d x %d frame, median of %d runs, single thread (outputs bit-exact)\n", h, w, runs);
    const char *titles[3] = {"golden: Cin innermost", "Cout innermost", "best per layer (CPU baseline)"};
    for (int which = 0; which < 3; which++) {
        double med[4], total = 0;
        for (int i = 0; i < 4; i++) {
            const double g = median(tg[i]), f = median(tf[i]);
            med[i] = which == 0 ? g : which == 1 ? f : std::min(g, f);
            total += med[i];
        }
        std::printf("\n[%s]\n%-20s %10s %8s %10s %8s\n", titles[which], "layer", "time (s)", "share", "GOP/s",
                    "MAC %");
        for (int i = 0; i < 4; i++) {
            const double gop = 2.0 * macs_px[i] * n / 1e9;
            std::printf("%-20s %10.4f %7.1f%% %10.2f %7.1f%%\n", names[i], med[i], 100 * med[i] / total,
                        macs_px[i] > 0 ? gop / med[i] : 0.0, 100 * macs_px[i] / 21184.0);
        }
        std::printf("%-20s %10.4f\n", "total", total);
    }
    return 0;
}
