"""Mutation test for the T2 harness: inject known bugs into V0 and check that
the harness catches every one of them.

    python3 test/mutation_test.py         # or: make t2-mutation

Each mutant is a one-line change to kernel/src/pl_v0.cpp that reproduces a bug
class from docs/02-test-plan.md section 2. The script prints which cases fail
for each mutant. Any mutant that passes every case is a hole in the test plan.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "kernel" / "src" / "pl_v0.cpp"

MUTANTS = [
    ("flipped kernel (kx -> 2-kx)", "const int ix = x + kx - 1;", "const int ix = x + (2 - kx) - 1;"),
    ("transposed taps (ky <-> kx)", "wts[co * TAPS + ky * KSIZE + kx]", "wts[co * TAPS + kx * KSIZE + ky]"),
    ("Cin slice off by one channel", "const ap_int<8> xv = xw(8 * ci + 7, 8 * ci);",
     "const ap_int<8> xv = xw(8 * ((ci + 1) % CIN) + 7, 8 * ((ci + 1) % CIN));"),
    ("wrong Cout weight word", "wts[co * TAPS + ky * KSIZE + kx]", "wts[((co + 1) % COUT) * TAPS + ky * KSIZE + kx]"),
    ("output pixels swapped in word", "const int bit = 8 * (COUT * p + co);", "const int bit = 8 * (COUT * (1 - p) + co);"),
    ("no zero padding at right edge", "ix < 0 || ix >= W", "ix < 0 || ix > W"),
    ("bias ignored", "int32_t acc = (ap_int<32>)bword(32 * (co % 16) + 31, 32 * (co % 16));", "int32_t acc = 0;"),
    ("int16 accumulator", "acc += (int32_t)xv * (int32_t)wv;", "acc = (int16_t)(acc + (int32_t)xv * (int32_t)wv);"),
    ("truncate instead of round", "quant::requantize_relu(acc, requant_mult, requant_shift)",
     "quant::requantize_relu(acc, requant_mult, requant_shift) - ((((int64_t)acc * requant_mult) >> (requant_shift - 1)) & 1)"),
    ("32-bit requant product", "quant::requantize_relu(acc, requant_mult, requant_shift)",
     "(int8_t)(((int32_t)(acc * requant_mult) >> 1) > 127 ? 127 : ((int32_t)(acc * requant_mult) >> 1) < 0 ? 0 : ((int32_t)(acc * requant_mult) >> 1))"),
    ("last row skipped", "const int n_words = H * W / PIX_PER_OFM_WORD;",
     "const int n_words = (H > 1 ? H - 1 : H) * W / PIX_PER_OFM_WORD;"),
]


def main() -> int:
    src = V0.read_text()
    holes = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, old, new in MUTANTS:
            assert src.count(old) >= 1, f"mutant '{name}': pattern not found in pl_v0.cpp"
            path = Path(tmp) / "mutant.cpp"
            path.write_text(src.replace(old, new, 1))
            r = subprocess.run([str(ROOT / "test" / "check_conv.sh"), str(path)], capture_output=True, text=True)
            failed = [ln.split()[0] for ln in r.stdout.splitlines() if ln.strip().endswith("FAIL") or " FAIL " in ln]
            caught = r.returncode != 0
            holes += not caught
            print(f"{'caught' if caught else 'MISSED':>6}  {name:<34} failing: {', '.join(failed) or '-'}")
    print(f"\n{len(MUTANTS) - holes}/{len(MUTANTS)} mutants caught")
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
