"""Gate T0: PSNR of bicubic vs. FP32 ESPCN vs. INT8 ESPCN on Set5 / Set14.

    python -m model.eval_psnr --ckpt model/espcn_fp32.pt

PSNR is measured on Y in [0, 1] with a 2-pixel border crop (docs/03-metrics.md).
T0 passes on Set5 when FP32 beats bicubic, INT8 beats bicubic by at least
1.0 dB, and INT8 is within 0.5 dB of FP32 (D19).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from model.data import DATA, bicubic, fetch, hr_dir, load_y_images
from model.espcn import ESPCN
from model.quant_ref import QModel, run_int8
from model.quantize import quantize_input, quantize_model

BORDER = 2
T0_MAX_DROP_DB = 0.5
T0_MIN_GAIN_OVER_BICUBIC_DB = 1.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a[BORDER:-BORDER, BORDER:-BORDER].astype(np.float64)
    b = b[BORDER:-BORDER, BORDER:-BORDER].astype(np.float64)
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def upscale_fp32(model: ESPCN, lr: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(lr)[None, None])[0, 0].numpy()
    return np.clip(out, 0.0, 1.0)


def upscale_int8(qm: QModel, lr: np.ndarray) -> np.ndarray:
    y = run_int8(qm, quantize_input(lr))["y_out"]
    return np.clip(y.astype(np.float64) * qm.s_out, 0.0, 1.0)


def evaluate(model: ESPCN, qm: QModel | None, dataset: str) -> dict[str, float]:
    """Mean PSNR over a dataset for each method."""
    model.eval()
    scores: dict[str, list[float]] = {"bicubic": [], "fp32": [], "int8": []}
    for lr, hr in load_y_images(hr_dir(dataset)):
        scores["bicubic"].append(psnr(np.clip(bicubic(lr, hr.shape), 0, 1), hr))
        scores["fp32"].append(psnr(upscale_fp32(model, lr), hr))
        if qm is not None:
            scores["int8"].append(psnr(upscale_int8(qm, lr), hr))
    return {k: float(np.mean(v)) for k, v in scores.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--calib", default="div2k", help="dataset used for INT8 calibration")
    ap.add_argument("--n-calib", type=int, default=50)
    args = ap.parse_args()

    for name in ("set5", "set14"):
        fetch(name)
    model = ESPCN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    calib = [lr for lr, _ in load_y_images(hr_dir(args.calib), limit=args.n_calib)]
    qm = quantize_model(model, calib)

    print(f"{'dataset':<8} {'bicubic':>8} {'FP32':>8} {'INT8':>8} {'drop':>6}   (PSNR dB, Y channel)")
    results = {}
    for name in ("set5", "set14"):
        r = evaluate(model, qm, name)
        results[name] = r
        print(f"{name:<8} {r['bicubic']:8.2f} {r['fp32']:8.2f} {r['int8']:8.2f} {r['fp32'] - r['int8']:6.2f}")

    s5 = results["set5"]
    ok = (s5["fp32"] > s5["bicubic"]
          and s5["int8"] - s5["bicubic"] >= T0_MIN_GAIN_OVER_BICUBIC_DB
          and s5["fp32"] - s5["int8"] <= T0_MAX_DROP_DB)
    print("T0 PASSED" if ok else "T0 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
