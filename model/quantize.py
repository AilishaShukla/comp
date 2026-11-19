"""Post-training INT8 quantization of the FP32 ESPCN, and .bin export.

Symmetric, per-tensor, zero-point 0, after cross-layer equalization
(docs/06-model-and-quantization.md section 2).

    python -m model.quantize --ckpt model/espcn_fp32.pt --calib data/T91 --out build/model
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from model.espcn import ESPCN
from model.quant_ref import LINEAR, RELU, QLayer, QModel, quantize_multiplier

S_IN = 1.0 / 127.0
CALIB_PERCENTILE = 99.99
CALIB_SAMPLES_PER_IMAGE = 1_000_000  # per layer; bounds calibration memory (D17)
CALIB_SEED = 0


def quantize_input(y: np.ndarray) -> np.ndarray:
    """Float Y in [0, 1] -> int8 in [0, 127]."""
    return np.clip(np.rint(y / S_IN), 0, 127).astype(np.int8)


def calibrate(model: ESPCN, images: list[np.ndarray]) -> list[float]:
    """99.99th-percentile |activation| per layer over the calibration images.

    Each image contributes at most CALIB_SAMPLES_PER_IMAGE values per layer,
    drawn with a fixed seed, so memory stays bounded on full-size DIV2K images
    (about 286 MB of activations each). Smaller images are used whole.
    """
    rng = np.random.default_rng(CALIB_SEED)
    samples: list[list[np.ndarray]] = [[], [], []]
    model.eval()
    with torch.no_grad():
        for img in images:
            x = torch.from_numpy(img.astype(np.float32))[None, None]
            for i, a in enumerate(model.features(x)):
                v = a.abs().flatten().numpy()
                if v.size > CALIB_SAMPLES_PER_IMAGE:
                    v = v[rng.integers(0, v.size, CALIB_SAMPLES_PER_IMAGE)]
                samples[i].append(v)
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


CLE_ITERS = 20


def _equalize_pair(c1: torch.nn.Conv2d, c2: torch.nn.Conv2d) -> None:
    """Cross-layer equalization of one conv pair joined by ReLU [R28].

    Scales output channel i of c1 by 1/s_i and input channel i of c2 by s_i,
    with s_i chosen so both get the same weight range sqrt(r1_i * r2_i).
    ReLU(x / s) = ReLU(x) / s for s > 0, so the FP32 output is unchanged.
    """
    r1 = c1.weight.abs().amax(dim=(1, 2, 3))
    r2 = c2.weight.abs().amax(dim=(0, 2, 3))
    s = torch.where((r1 > 0) & (r2 > 0), torch.sqrt(r1 * r2) / r2, torch.ones_like(r1))
    c1.weight.div_(s[:, None, None, None])
    c1.bias.div_(s)
    c2.weight.mul_(s[None, :, None, None])


def equalize(model: ESPCN) -> ESPCN:
    """Return a copy of the model with L1-L2 and L2-L3 weight ranges equalized (D18)."""
    eq = copy.deepcopy(model).double()
    with torch.no_grad():
        for _ in range(CLE_ITERS):
            _equalize_pair(eq.l1, eq.l2)
            _equalize_pair(eq.l2, eq.l3)
    return eq.float()


def quantize_model(model: ESPCN, calib_images: list[np.ndarray]) -> QModel:
    model = equalize(model)
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
