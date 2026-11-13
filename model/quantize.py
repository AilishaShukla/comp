"""Post-training INT8 quantization of the FP32 ESPCN, and .bin export.

Symmetric, per-tensor, zero-point 0 (docs/06-model-and-quantization.md section 2).

    python -m model.quantize --ckpt model/espcn_fp32.pt --calib data/T91 --out build/model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model.espcn import ESPCN
from model.quant_ref import LINEAR, RELU, QLayer, QModel, quantize_multiplier

S_IN = 1.0 / 127.0
CALIB_PERCENTILE = 99.99


def quantize_input(y: np.ndarray) -> np.ndarray:
    """Float Y in [0, 1] -> int8 in [0, 127]."""
    return np.clip(np.rint(y / S_IN), 0, 127).astype(np.int8)


def calibrate(model: ESPCN, images: list[np.ndarray]) -> list[float]:
    """99.99th-percentile |activation| per layer over the calibration images."""
    samples: list[list[np.ndarray]] = [[], [], []]
    model.eval()
    with torch.no_grad():
        for img in images:
            x = torch.from_numpy(img.astype(np.float32))[None, None]
            for i, a in enumerate(model.features(x)):
                samples[i].append(a.abs().flatten().numpy())
    return [float(np.percentile(np.concatenate(s), CALIB_PERCENTILE)) for s in samples]


def _quant_layer(conv: torch.nn.Conv2d, s_x: float, s_y: float, out_range) -> QLayer:
    w = conv.weight.detach().double().numpy()            # (Co, Ci, K, K)
    b = conv.bias.detach().double().numpy()
    s_w = float(np.abs(w).max()) / 127.0
    wq = np.clip(np.rint(w / s_w), -127, 127).astype(np.int8).transpose(0, 2, 3, 1)  # -> (Co, K, K, Ci)
    bq = np.rint(b / (s_w * s_x)).astype(np.int64)
    assert np.abs(bq).max() < 2**31
    mult, shift = quantize_multiplier(s_w * s_x / s_y)
    return QLayer(np.ascontiguousarray(wq), bq.astype(np.int32), mult, shift, out_range)


def quantize_model(model: ESPCN, calib_images: list[np.ndarray]) -> QModel:
    a1, a2, a3 = calibrate(model, calib_images)
    s1, s2, s3 = a1 / 127.0, a2 / 127.0, a3 / 127.0
    return QModel(
        l1=_quant_layer(model.l1, S_IN, s1, RELU),
        l2=_quant_layer(model.l2, s1, s2, RELU),
        l3=_quant_layer(model.l3, s2, s3, LINEAR),
        s_in=S_IN,
        s_out=s3,
    )


def export(qm: QModel, out_dir: Path) -> None:
    """Write l{1,2,3}_{w,b}.bin and qparams.bin in the docs/06 section 4 format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    qparams = []
    for name, layer in qm.layers():
        layer.w.astype("<i1").tofile(out_dir / f"{name}_w.bin")
        layer.b.astype("<i4").tofile(out_dir / f"{name}_b.bin")
        qparams += [layer.mult, layer.shift]
    np.array(qparams, dtype="<i4").tofile(out_dir / "qparams.bin")
    (out_dir / "scales.txt").write_text(f"s_in {qm.s_in!r}\ns_out {qm.s_out!r}\n")


def main() -> None:
    from model.data import load_y_images  # local import: only the CLI needs image I/O

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True, help="directory of calibration images (HR)")
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = ESPCN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    lr_images = [lr for lr, _ in load_y_images(args.calib, limit=args.n_calib)]
    qm = quantize_model(model, lr_images)
    export(qm, args.out)
    for name, layer in qm.layers():
        print(f"{name}: mult={layer.mult} shift={layer.shift}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
