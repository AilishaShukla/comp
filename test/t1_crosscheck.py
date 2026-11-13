"""Gate T1: C++ golden model == Python integer reference, bit for bit.

For each case: build a seeded ESPCN, quantize it, run the numpy integer
reference (model/quant_ref.py), export everything as .bin files, then run
build/run_golden on the same files and require zero mismatches on every layer.

Random weights are enough here: T1 tests the arithmetic contract, not image
quality. Once a trained checkpoint exists, pass --ckpt to cross-check it too.

    make t1                      # or: python3 test/t1_crosscheck.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.espcn import ESPCN  # noqa: E402
from model.quant_ref import run_int8  # noqa: E402
from model.quantize import export, quantize_input, quantize_model  # noqa: E402

GOLDEN = ROOT / "build" / "run_golden"

# (name, H, W, seed). The sizes cover tiny, odd and non-square frames.
CASES = [
    ("min_frame", 1, 2, 1),
    ("odd_rows", 7, 8, 2),
    ("small", 16, 16, 3),
    ("non_square", 24, 40, 4),
    ("medium", 64, 64, 5),
]


def smooth_image(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A random image-like Y plane in [0, 1]: low-frequency noise plus fine detail."""
    coarse = torch.from_numpy(rng.random((1, 1, max(2, h // 4), max(2, w // 4)), dtype=np.float32))
    base = torch.nn.functional.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)[0, 0].numpy()
    return np.clip(base + 0.1 * rng.standard_normal((h, w)), 0.0, 1.0)


def run_case(name: str, h: int, w: int, seed: int, model: ESPCN, out_root: Path) -> bool:
    rng = np.random.default_rng(seed)
    calib = [smooth_image(32, 32, rng) for _ in range(8)]
    qm = quantize_model(model, calib)

    y_in = quantize_input(smooth_image(h, w, rng))
    ref = run_int8(qm, y_in)

    d = out_root / name
    export(qm, d)
    y_in.tofile(d / "y_in.bin")
    for key, arr in ref.items():
        arr.astype(np.int8).tofile(d / f"{key}.bin")

    print(f"--- {name} ({h}x{w}, seed {seed})")
    r = subprocess.run([str(GOLDEN), str(d), str(h), str(w)], capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip())
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, help="trained FP32 checkpoint (default: seeded random weights)")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "t1")
    args = ap.parse_args()

    if not GOLDEN.exists():
        print(f"error: {GOLDEN} not found; run `make golden` first", file=sys.stderr)
        return 2

    passed = 0
    for name, h, w, seed in CASES:
        torch.manual_seed(seed)
        model = ESPCN().eval()
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        passed += run_case(name, h, w, seed, model, args.out)

    print(f"\nT1: {passed}/{len(CASES)} cases bit-exact")
    if passed == len(CASES):
        print("T1 PASSED")
        return 0
    print("T1 FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
