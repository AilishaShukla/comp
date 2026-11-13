"""Integer-only reference for the INT8 ESPCN (numpy).

This is written independently of golden/quant.h and golden/espcn_golden.cpp.
Gate T1 (test/t1_crosscheck.py) checks that the two produce identical bytes,
which is the evidence that the rounding contract is implemented correctly.

Spec: docs/06-model-and-quantization.md sections 2-4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RELU = (0, 127)
LINEAR = (-128, 127)


def quantize_multiplier(m: float) -> tuple[int, int]:
    """Express a positive real scale as mult * 2^-shift with mult in [2^30, 2^31).

    Follows gemmlowp [R11]. The docs expect 0 < m < 1 (shift >= 31); larger m
    is accepted as long as the shift stays >= 1.
    """
    if not m > 0:
        raise ValueError(f"requant scale must be positive, got {m}")
    shift = 0
    while m < 0.5:
        m *= 2
        shift += 1
    while m >= 1.0:
        m /= 2
        shift -= 1
    mult = int(round(m * (1 << 31)))
    if mult == (1 << 31):
        mult //= 2
        shift -= 1
    shift += 31
    if not (1 << 30) <= mult < (1 << 31) or not 1 <= shift <= 62:
        raise ValueError(f"scale {m} out of range: mult={mult}, shift={shift}")
    return mult, shift


def requantize(acc: np.ndarray, mult: int, shift: int, lo: int, hi: int) -> np.ndarray:
    """out = clamp((acc * mult + 2^(shift-1)) >> shift, lo, hi), all in int64."""
    prod = acc.astype(np.int64) * np.int64(mult)
    rounded = (prod + (np.int64(1) << np.int64(shift - 1))) >> np.int64(shift)
    return np.clip(rounded, lo, hi).astype(np.int8)


def conv_acc(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same-padded correlation in HWC. x: (H, W, Ci) int8, w: (Co, K, K, Ci) int8, b: (Co,) int32.

    Returns the exact int32 accumulators. The matmul runs in float64 for
    speed; every partial sum is an integer below 2^53, so the result is exact.
    """
    h, wd, ci = x.shape
    co, k, _, ci_w = w.shape
    assert ci == ci_w, (ci, ci_w)
    pad = k // 2
    xp = np.zeros((h + 2 * pad, wd + 2 * pad, ci), dtype=np.float64)
    xp[pad:pad + h, pad:pad + wd] = x
    acc = np.zeros((h * wd, co), dtype=np.float64)
    wf = w.astype(np.float64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[ky:ky + h, kx:kx + wd].reshape(-1, ci)
            acc += patch @ wf[:, ky, kx, :].T
    bound = float(np.abs(w.astype(np.int64)).sum(axis=(1, 2, 3)).max()) * 128 + float(np.abs(b).max())
    assert bound < 2**31, "accumulator could overflow int32"
    return (np.rint(acc).astype(np.int64) + b.astype(np.int64)).reshape(h, wd, co)


def pixel_shuffle2(x: np.ndarray) -> np.ndarray:
    """(H, W, 4) -> (2H, 2W) with out[2y+dy, 2x+dx] = x[y, x, 2*dy + dx]."""
    h, w, c = x.shape
    assert c == 4
    return x.reshape(h, w, 2, 2).transpose(0, 2, 1, 3).reshape(2 * h, 2 * w)


@dataclass
class QLayer:
    w: np.ndarray        # (Co, K, K, Ci) int8
    b: np.ndarray        # (Co,) int32
    mult: int
    shift: int
    out_range: tuple[int, int]


@dataclass
class QModel:
    l1: QLayer
    l2: QLayer
    l3: QLayer
    s_in: float          # input scale: y_float ~= y_int8 * s_in
    s_out: float         # L3 output scale

    def layers(self):
        return (("l1", self.l1), ("l2", self.l2), ("l3", self.l3))


def run_layer(x: np.ndarray, layer: QLayer) -> np.ndarray:
    return requantize(conv_acc(x, layer.w, layer.b), layer.mult, layer.shift, *layer.out_range)


def run_int8(qm: QModel, y_in: np.ndarray) -> dict[str, np.ndarray]:
    """y_in: (H, W) int8 in [0, 127]. Returns every intermediate as int8."""
    l1 = run_layer(y_in[:, :, None], qm.l1)
    l2 = run_layer(l1, qm.l2)
    l3 = run_layer(l2, qm.l3)
    return {"l1_out": l1, "l2_out": l2, "l3_out": l3, "y_out": pixel_shuffle2(l3)}
