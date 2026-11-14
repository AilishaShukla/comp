// tb_conv - gate T2: run a conv3x3_l2 kernel version on the CPU and compare it,
// bit for bit, with the golden model.
//
//   tb_conv <vector_root> <case> [<case> ...]
//
// For each case directory (made by model/gen_vectors.py) it:
//   1. packs l1_out.bin / l2_w.bin / l2_b.bin into 512-bit words using the bit
//      layout of docs/01-architecture.md section 3 (the same packing host.cpp does);
//   2. calls the kernel;
//   3. unpacks the OFM and compares it with golden::conv2d (C++) and with the
//      expected l2_out.bin from the numpy integer reference;
//   4. fails if any hls::stream still holds data (that would hang on the FPGA).
// Mismatches are reported as (y, x, co) so an index bug points at its loop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hls_stream.h>

#include "conv3x3_l2.h"
#include "espcn_golden.h"

using golden::read_bin;

// Pack a byte array into 512-bit words, 64 bits at a time.
static std::vector<block_t> pack(const void *data, size_t bytes) {
    std::vector<block_t> words((bytes + 63) / 64);
    std::vector<uint8_t> buf(words.size() * 64, 0);
    std::memcpy(buf.data(), data, bytes);
    for (size_t i = 0; i < words.size(); i++)
        for (int j = 0; j < 8; j++) {
            long long chunk;
            std::memcpy(&chunk, &buf[i * 64 + j * 8], 8);
            words[i](64 * j + 63, 64 * j) = chunk;
        }
    return words;
}

static std::vector<int8_t> unpack(const std::vector<block_t> &words, size_t bytes) {
    std::vector<uint8_t> buf(words.size() * 64);
    for (size_t i = 0; i < words.size(); i++)
        for (int j = 0; j < 8; j++) {
            const unsigned long long chunk = words[i](64 * j + 63, 64 * j);
            std::memcpy(&buf[i * 64 + j * 8], &chunk, 8);
        }
    return std::vector<int8_t>(buf.begin(), buf.begin() + bytes);
}

static size_t count_mismatches(const char *what, const std::vector<int8_t> &got, const std::vector<int8_t> &want,
                               int w) {
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] == want[i]) continue;
        if (bad < 10) {
            const size_t p = i / COUT;
            std::printf("    %s: (y=%zu, x=%zu, co=%zu) got %4d want %4d\n", what, p / w, p % w, i % COUT, got[i],
                        want[i]);
        }
        bad++;
    }
    return bad;
}

static bool run_case(const std::string &dir, const std::string &name, int h, int w) {
    const size_t px = (size_t)h * w;
    const auto ifm = read_bin<int8_t>(dir + "/l1_out.bin", px * CIN);
    const auto wts = read_bin<int8_t>(dir + "/l2_w.bin", (size_t)COUT * TAPS * CIN);
    const auto bias = read_bin<int32_t>(dir + "/l2_b.bin", COUT);
    const auto qp = read_bin<int32_t>(dir + "/qparams.bin", 6);
    const auto expected = read_bin<int8_t>(dir + "/l2_out.bin", px * COUT);

    // Golden model on the same inputs.
    golden::ConvLayer L(3, CIN, COUT);
    L.w = wts;
    L.b = bias;
    L.mult = qp[2];
    L.shift = qp[3];
    std::vector<int8_t> gold(px * COUT);
    golden::conv2d(ifm.data(), h, w, L, gold.data());

    // Kernel.
    const std::vector<block_t> ifm_w = pack(ifm.data(), ifm.size());
    const std::vector<block_t> wts_w = pack(wts.data(), wts.size());
    const std::vector<block_t> bias_w = pack(bias.data(), bias.size() * 4);
    std::vector<block_t> ofm_w(px / PIX_PER_OFM_WORD);
    hls::g_stream_leftover = false;
    conv3x3_l2(ifm_w.data(), wts_w.data(), bias_w.data(), ofm_w.data(), h, w, qp[2], qp[3]);
    const std::vector<int8_t> got = unpack(ofm_w, px * COUT);

    const size_t ref_bad = count_mismatches("golden vs numpy", gold, expected, w);
    const size_t bad = count_mismatches("kernel vs golden", got, gold, w);
    const bool ok = bad == 0 && ref_bad == 0 && !hls::g_stream_leftover;
    std::printf("  %-22s %4dx%-4d %s", name.c_str(), h, w, ok ? "PASS" : "FAIL");
    if (bad) std::printf("  (%zu / %zu mismatches)", bad, got.size());
    if (ref_bad) std::printf("  (golden disagrees with numpy: %zu)", ref_bad);
    if (hls::g_stream_leftover) std::printf("  (stream count mismatch)");
    std::printf("\n");
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <vector_root> <case:H:W> [...]\n", argv[0]);
        return 2;
    }
    const std::string root = argv[1];
    int passed = 0, total = 0;
    for (int i = 2; i < argc; i++) {
        // argument format: name:H:W (check_conv.sh reads these from manifest.csv)
        char name[128];
        int h, w;
        if (std::sscanf(argv[i], "%127[^:]:%d:%d", name, &h, &w) != 3) {
            std::fprintf(stderr, "bad case spec '%s' (want name:H:W)\n", argv[i]);
            return 2;
        }
        total++;
        passed += run_case(root + "/" + name, name, h, w);
    }
    std::printf("\nT2: %d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
