// run_golden - run the golden INT8 ESPCN on a vector directory.
//
//   run_golden <dir> <H> <W> [--write]
//
// Reads <dir>/y_in.bin plus the weights and qparams. With --write, writes
// golden_{l1,l2,l3,y}_out.bin. If the reference outputs l{1,2,3}_out.bin and
// y_out.bin (from model/quant_ref.py) exist, compares each one bit for bit.
// Exits 0 only if every comparison passes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "espcn_golden.h"

static bool exists(const std::string &p) { return std::ifstream(p).good(); }

// Compare HWC tensors; print the first few mismatches as (y, x, c).
static bool compare(const char *name, const std::vector<int8_t> &got, const std::vector<int8_t> &want,
                    int w, int c) {
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] == want[i]) continue;
        if (bad < 10) {
            const size_t p = i / c;
            std::printf("  %s mismatch at (y=%zu, x=%zu, c=%zu): got %d want %d\n", name, p / w, p % w, i % c,
                        got[i], want[i]);
        }
        bad++;
    }
    std::printf("%-6s %s  (%zu / %zu mismatches)\n", name, bad ? "FAIL" : "ok", bad, got.size());
    return bad == 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <dir> <H> <W> [--write]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int h = std::atoi(argv[2]);
    const int w = std::atoi(argv[3]);
    const bool write = argc > 4 && std::strcmp(argv[4], "--write") == 0;

    const golden::Model m = golden::load_model(dir);
    const auto y_in = golden::read_bin<int8_t>(dir + "/y_in.bin", (size_t)h * w);
    const golden::Outputs o = golden::run(m, y_in.data(), h, w);

    if (write) {
        golden::write_bin(dir + "/golden_l1_out.bin", o.l1);
        golden::write_bin(dir + "/golden_l2_out.bin", o.l2);
        golden::write_bin(dir + "/golden_l3_out.bin", o.l3);
        golden::write_bin(dir + "/golden_y_out.bin", o.y);
    }

    struct Check { const char *name; const std::vector<int8_t> *got; int w, c; };
    const Check checks[] = {
        {"l1_out", &o.l1, w, 64}, {"l2_out", &o.l2, w, 32}, {"l3_out", &o.l3, w, 4}, {"y_out", &o.y, 2 * w, 1}};

    bool ok = true;
    int compared = 0;
    for (const Check &c : checks) {
        const std::string ref = dir + "/" + c.name + ".bin";
        if (!exists(ref)) continue;
        ok &= compare(c.name, *c.got, golden::read_bin<int8_t>(ref, c.got->size()), c.w, c.c);
        compared++;
    }
    if (compared == 0) {
        std::printf("no reference outputs in %s; nothing compared\n", dir.c_str());
        return 0;
    }
    std::printf(ok ? "GOLDEN MATCHES REFERENCE\n" : "GOLDEN MISMATCH\n");
    return ok ? 0 : 1;
}
