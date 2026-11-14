"""Generate the T2 kernel test vectors (docs/02-test-plan.md section 2).

    python3 -m model.gen_vectors                                # all synthetic cases
    python3 -m model.gen_vectors --model build/model --image <png>   # also real_frame

Each case directory holds the L2 inputs and the expected output, in the
formats of docs/06-model-and-quantization.md section 4:
    l1_out.bin   IFM, H x W x 64 int8 (HWC)
    l2_w.bin     32 x 3 x 3 x 64 int8
    l2_b.bin     32 int32
    qparams.bin  6 int32; only the L2 pair (slots 2, 3) is used by T2
    l2_out.bin   expected OFM, H x W x 32 int8, from the numpy integer reference
    shape.txt    "H W", so host.exe can take just the directory
Plus manifest.csv: case,H,W,seed,dir,purpose. Everything is seeded and reproducible.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from model.quant_ref import RELU, QLayer, conv_acc, quantize_multiplier, run_layer

ROOT = Path(__file__).resolve().parents[1]
CIN, COUT = 64, 32


def layer(w: np.ndarray, b: np.ndarray, scale: float) -> QLayer:
    mult, shift = quantize_multiplier(scale)
    return QLayer(w.astype(np.int8), b.astype(np.int32), mult, shift, RELU)


def fit_scale(x: np.ndarray, w: np.ndarray, b: np.ndarray, pct: float = 99.0) -> float:
    """Scale that maps the pct-th percentile of positive accumulators to 127."""
    acc = conv_acc(x, w, b)
    pos = acc[acc > 0]
    return 127.0 / float(np.percentile(pos, pct)) if pos.size else 1.0


def random_ifm(rng, h, w):
    """Post-ReLU-like activations: about half zeros, the rest in [1, 127]."""
    x = rng.integers(1, 128, size=(h, w, CIN))
    return np.where(rng.random((h, w, CIN)) < 0.5, 0, x).astype(np.int8)


def random_case(rng, h, w):
    x = random_ifm(rng, h, w)
    wt = rng.integers(-127, 128, size=(COUT, 3, 3, CIN)).astype(np.int8)
    b = rng.integers(-20000, 20000, size=COUT).astype(np.int32)
    return x, layer(wt, b, fit_scale(x, wt, b))


# ---------------------------------------------------------------- targeted cases

def impulse_center(rng):
    h, w = 5, 6
    x = np.zeros((h, w, CIN), np.int8)
    x[2, 3, 0] = 1
    wt = rng.integers(1, 128, size=(COUT, 3, 3, CIN)).astype(np.int8)
    return x, layer(wt, np.zeros(COUT), 1.0)          # out = weight, so ky/kx order is visible


def impulse_per_channel(rng):
    h, w = 3, 3 * CIN
    x = np.zeros((h, w, CIN), np.int8)
    for c in range(CIN):
        x[1, 3 * c + 1, c] = 1                         # windows 3 px apart never overlap
    wt = rng.integers(1, 128, size=(COUT, 3, 3, CIN)).astype(np.int8)
    return x, layer(wt, np.zeros(COUT), 1.0)


def weight_onehot(rng):
    h, w = 4, 4
    x = random_ifm(rng, h, w)
    wt = np.zeros((COUT, 3, 3, CIN), np.int8)
    for co in range(COUT):                             # a different tap and Cin for every Cout
        wt[co, co % 3, (co // 3) % 3, (co * 7) % CIN] = 1
    return x, layer(wt, np.zeros(COUT), 1.0)


def border_ones(rng):
    h, w = 4, 6
    x = np.ones((h, w, CIN), np.int8)
    wt = np.ones((COUT, 3, 3, CIN), np.int8)
    return x, layer(wt, np.zeros(COUT), 1 / 8)         # interior 72, edge 48, corner 32


def saturate_hi(rng):
    h, w = 4, 4
    x = np.full((h, w, CIN), 127, np.int8)
    wt = np.full((COUT, 3, 3, CIN), 127, np.int8)      # acc = 576 * 127^2 = 9.29 M (> 2^23)
    b = rng.integers(0, 1000, size=COUT)
    # 9 taps -> 186 (clamps to 127); edge 6 taps -> 124 and corner 4 taps -> 83 (don't clamp)
    return x, layer(wt, b, 2e-5)


def saturate_lo(rng):
    h, w = 4, 4
    x = np.full((h, w, CIN), 127, np.int8)
    wt = np.full((COUT, 3, 3, CIN), -127, np.int8)     # large negative acc -> ReLU clamps to 0
    wt[::2] = 127                                      # even Cout stay positive, so the two must not mix
    b = rng.integers(-1000, 1000, size=COUT)
    return x, layer(wt, b, 1e-5)


def rounding(rng):
    h, w = 4, 4
    x = (rng.random((h, w, CIN)) < 0.2).astype(np.int8)
    wt = (rng.random((COUT, 3, 3, CIN)) < 0.2).astype(np.int8)
    b = rng.integers(-10, 60, size=COUT)
    return x, layer(wt, b, 0.5)                        # out = (acc + 1) >> 1: every odd acc is a .5 tie


TARGETED = [
    ("impulse_center", impulse_center, "ky/kx order (correlation, not convolution)"),
    ("impulse_per_channel", impulse_per_channel, "Cin bit-slice index"),
    ("weight_onehot", weight_onehot, "Cout <-> weight-word mapping, output packing"),
    ("border_ones", border_ones, "zero padding at edges and corners"),
    ("saturate_hi", saturate_hi, "int32 accumulator, 64-bit product, clamp at 127"),
    ("saturate_lo", saturate_lo, "ReLU / clamp at 0"),
    ("rounding", rounding, "round-half-up matches quant.h"),
]

RANDOM = [
    ("odd_rows", 7, 8, "line-buffer start/end with odd H"),
    ("min_frame", 1, 2, "degenerate frame, window never full"),
    *[(f"random_small_{s}", 16, 16, "general correctness") for s in range(1, 6)],
    ("random_wide", 32, 540, "full-width line buffer, W = 540"),
]


def write_case(out: Path, name: str, x: np.ndarray, L: QLayer) -> np.ndarray:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    y = run_layer(x, L)
    x.astype("<i1").tofile(d / "l1_out.bin")
    L.w.astype("<i1").tofile(d / "l2_w.bin")
    L.b.astype("<i4").tofile(d / "l2_b.bin")
    np.array([0, 0, L.mult, L.shift, 0, 0], "<i4").tofile(d / "qparams.bin")
    y.astype("<i1").tofile(d / "l2_out.bin")
    (d / "shape.txt").write_text(f"{x.shape[0]} {x.shape[1]}\n")
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "test" / "vectors")
    ap.add_argument("--model", type=Path, help="exported model dir (l*_w.bin, qparams.bin) for real_frame")
    ap.add_argument("--image", type=Path, help="HR image for real_frame (LR is cropped/padded to 540x540)")
    args = ap.parse_args()

    rows = []
    for seed, (name, fn, purpose) in enumerate(TARGETED, start=100):
        x, L = fn(np.random.default_rng(seed))
        y = write_case(args.out, name, x, L)
        rows.append((name, *x.shape[:2], seed, purpose, y))
    for i, (name, h, w, purpose) in enumerate(RANDOM):
        seed = int(name.rsplit("_", 1)[1]) if name.startswith("random_small") else 200 + i
        x, L = random_case(np.random.default_rng(seed), h, w)
        y = write_case(args.out, name, x, L)
        rows.append((name, h, w, seed, purpose, y))

    if args.model and args.image:
        rows.append(real_frame(args))

    with open(args.out / "manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case", "H", "W", "seed", "dir", "purpose"])
        for name, h, w, seed, purpose, _ in rows:
            wr.writerow([name, h, w, seed, name, purpose])

    print(f"{'case':<22} {'HxW':>8} {'zero%':>6} {'127%':>6} {'distinct':>8}")
    for name, h, w, _, _, y in rows:
        print(f"{name:<22} {f'{h}x{w}':>8} {100 * (y == 0).mean():6.1f} {100 * (y == 127).mean():6.1f} "
              f"{len(np.unique(y)):8d}")
    print(f"wrote {len(rows)} cases to {args.out}")


def real_frame(args):
    """L2 input = the real L1 output of a 540x540 LR crop, using the exported model."""
    from model.data import lr_hr_pair
    from model.quantize import quantize_input

    def rd(name, dtype, shape):
        return np.fromfile(args.model / name, dtype=dtype).reshape(shape)

    qp = np.fromfile(args.model / "qparams.bin", dtype="<i4")
    l1 = QLayer(rd("l1_w.bin", "<i1", (64, 5, 5, 1)), rd("l1_b.bin", "<i4", (64,)), int(qp[0]), int(qp[1]), RELU)
    l2 = QLayer(rd("l2_w.bin", "<i1", (32, 3, 3, 64)), rd("l2_b.bin", "<i4", (32,)), int(qp[2]), int(qp[3]), RELU)
    lr, _ = lr_hr_pair(args.image)
    frame = np.zeros((540, 540), np.float32)
    h, w = min(540, lr.shape[0]), min(540, lr.shape[1])
    frame[:h, :w] = lr[:h, :w]
    x = run_layer(quantize_input(frame)[:, :, None], l1)
    y = write_case(args.out, "real_frame", x, l2)
    return ("real_frame", 540, 540, 0, f"real L1 output of {args.image.name}", y)


if __name__ == "__main__":
    main()
