// host.cpp - XRT host for the conv3x3_l2 kernel (ESPCN layer L2).
//
//   host.exe <xclbin> <vector_dir> [n_runs=10]
//
// Loads one test-vector directory (model/gen_vectors.py), runs the kernel
// 1 warm-up + n_runs times, checks the output bit for bit against the golden
// model, and prints the metrics block defined in docs/03-metrics.md section 2.
// The output is also written to fpga_l2_out.bin for the visual diff map.
//
// T_kernel times run() + wait() only. Buffer syncs are timed separately
// (T_h2d, T_d2h) and are not part of GOPS (decision D9).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include "espcn_golden.h"

namespace {

const int CIN = 64, COUT = 32, TAPS = 9;
const double F_KERNEL_HZ = 200e6;

using Clock = std::chrono::high_resolution_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void read_shape(const std::string &dir, int &h, int &w) {
    std::ifstream f(dir + "/shape.txt");
    if (!(f >> h >> w) || h <= 0 || w <= 0 || w % 2) {
        std::fprintf(stderr, "error: bad or missing %s/shape.txt (need 'H W', W even)\n", dir.c_str());
        std::exit(2);
    }
}

// Allocate a device buffer for kernel argument `arg` and fill it with `bytes` from src.
xrt::bo make_input(xrt::device &dev, xrt::kernel &k, int arg, const void *src, size_t bytes) {
    const size_t padded = (bytes + 63) / 64 * 64;  // whole 512-bit words
    xrt::bo bo(dev, padded, k.group_id(arg));
    auto *map = bo.map<uint8_t *>();
    std::memset(map, 0, padded);
    std::memcpy(map, src, bytes);
    return bo;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <xclbin> <vector_dir> [n_runs=10]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string xclbin = argv[1], dir = argv[2];
    const int n_runs = argc > 3 ? std::max(1, std::atoi(argv[3])) : 10;

    int h, w;
    read_shape(dir, h, w);
    const size_t px = (size_t)h * w;
    const auto ifm = golden::read_bin<int8_t>(dir + "/l1_out.bin", px * CIN);
    const auto wts = golden::read_bin<int8_t>(dir + "/l2_w.bin", (size_t)COUT * TAPS * CIN);
    const auto bias = golden::read_bin<int32_t>(dir + "/l2_b.bin", COUT);
    const auto qp = golden::read_bin<int32_t>(dir + "/qparams.bin", 6);
    const int32_t mult = qp[2], shift = qp[3];

    std::printf("frame %dx%d  mult=%d shift=%d  runs=1+%d\n", h, w, mult, shift, n_runs);
    std::printf("load %s\n", xclbin.c_str());
    xrt::device device(0);
    const auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, "conv3x3_l2");

    // Arguments: 0 ifm, 1 wts, 2 bias, 3 ofm, 4 H, 5 W, 6 mult, 7 shift.
    auto bo_ifm = make_input(device, krnl, 0, ifm.data(), ifm.size());
    auto bo_wts = make_input(device, krnl, 1, wts.data(), wts.size());
    auto bo_bias = make_input(device, krnl, 2, bias.data(), bias.size() * sizeof(int32_t));
    const size_t ofm_bytes = px * COUT;
    xrt::bo bo_ofm(device, ofm_bytes, krnl.group_id(3));

    auto t0 = Clock::now();
    bo_ifm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_wts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const double t_h2d = seconds(t0, Clock::now());

    std::vector<double> times;
    for (int r = 0; r <= n_runs; r++) {
        auto a = Clock::now();
        auto run = krnl(bo_ifm, bo_wts, bo_bias, bo_ofm, h, w, mult, shift);
        run.wait();
        auto b = Clock::now();
        if (r > 0) times.push_back(seconds(a, b));  // run 0 is the warm-up
    }

    t0 = Clock::now();
    bo_ofm.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const double t_d2h = seconds(t0, Clock::now());
    const auto *out = bo_ofm.map<int8_t *>();
    std::vector<int8_t> got(out, out + ofm_bytes);
    golden::write_bin("fpga_l2_out.bin", got);

    // Golden reference on the same inputs.
    golden::ConvLayer L(3, CIN, COUT);
    L.w = wts;
    L.b = bias;
    L.mult = mult;
    L.shift = shift;
    std::vector<int8_t> gold(ofm_bytes);
    golden::conv2d(ifm.data(), h, w, L, gold.data());

    size_t bad = 0;
    for (size_t i = 0; i < ofm_bytes; i++) {
        if (got[i] == gold[i]) continue;
        if (bad < 10) {
            const size_t p = i / COUT;
            std::printf("  mismatch (y=%zu, x=%zu, co=%zu): fpga %d golden %d\n", p / w, p % w, i % COUT, got[i],
                        gold[i]);
        }
        bad++;
    }

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2], tmin = sorted.front(), tmax = sorted.back();
    const double ops = 2.0 * TAPS * CIN * COUT * (double)px;

    std::printf("\n=== metrics (docs/03-metrics.md) ===\n");
    std::printf("frame             %d x %d\n", h, w);
    std::printf("t_kernel_median   %.6f s\n", med);
    std::printf("t_kernel_min      %.6f s\n", tmin);
    std::printf("t_kernel_max      %.6f s\n", tmax);
    std::printf("runs              %d (+1 warm-up)\n", n_runs);
    std::printf("spread            %.1f %%%s\n", 100 * (tmax - tmin) / med, (tmax - tmin) / med > 0.05 ? "  (> 5 %, check)" : "");
    std::printf("t_h2d             %.6f s\n", t_h2d);
    std::printf("t_d2h             %.6f s\n", t_d2h);
    std::printf("gops              %.3f\n", ops / med / 1e9);
    std::printf("measured_cycles   %.0f\n", med * F_KERNEL_HZ);
    std::printf("mismatches        %zu / %zu\n", bad, ofm_bytes);

    if (bad) {
        std::printf("TEST FAILED!\n");
        return EXIT_FAILURE;
    }
    std::printf("TEST PASSED!\n");
    return 0;
}
