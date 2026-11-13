"""Datasets: download, Y-channel conversion, LR/HR pairs and training patches.

    python -m model.data                 # download Set5, Set14 and DIV2K train into data/

Conventions (docs/06-model-and-quantization.md section 1):
  * Y is BT.601 luma as used in SR papers: Y = 16 + (65.481 R + 128.553 G + 24.966 B) / 255,
    then divided by 255 so it lies in [16/255, 235/255].
  * HR images are cropped to even size; LR = bicubic downscale by 2 of the HR Y plane.
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    # name: (url, archive file name, directory of HR images after extraction)
    "set5": ("https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz",
             "Set5_HR.tar.gz", "Set5_HR"),
    "set14": ("https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz",
              "Set14_HR.tar.gz", "Set14_HR"),
    "div2k": ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
              "DIV2K_train_HR.zip", "DIV2K_train_HR"),
}

IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg"}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def fetch(name: str) -> Path:
    """Download and extract a dataset (once). Returns the directory of HR images."""
    url, archive, _ = SOURCES[name]
    dest = DATA / name
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if not marker.exists():
        _download(url, dest / archive)
        print(f"extracting {archive}")
        if archive.endswith(".zip"):
            with zipfile.ZipFile(dest / archive) as z:
                z.extractall(dest)
        else:
            with tarfile.open(dest / archive) as t:
                t.extractall(dest, filter="data")
        marker.touch()
    return hr_dir(name)


def hr_dir(name: str) -> Path:
    """Locate the folder of HR images inside an extracted dataset."""
    base = DATA / name
    for p in sorted([base, *base.rglob("*")]):
        if p.is_dir() and any(f.suffix.lower() in IMG_EXT for f in p.iterdir()):
            return p
    raise FileNotFoundError(f"no images under {base}; run `python -m model.data` first")


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 Y in [16/255, 235/255]."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return (y / 255.0).astype(np.float32)


def bicubic(y: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bicubic resize of a float Y plane to (H, W)."""
    img = Image.fromarray(y.astype(np.float32), mode="F")
    return np.asarray(img.resize((size[1], size[0]), Image.BICUBIC), dtype=np.float32)


def lr_hr_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    hr = rgb_to_y(rgb)
    h, w = (hr.shape[0] // 2) * 2, (hr.shape[1] // 2) * 2
    hr = hr[:h, :w]
    lr = np.clip(bicubic(hr, (h // 2, w // 2)), 0.0, 1.0)
    return lr, hr


def image_paths(folder: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
    return paths[:limit] if limit else paths


def load_y_images(folder: Path, limit: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(lr, hr), ...] float32 Y planes for every image in folder."""
    return [lr_hr_pair(p) for p in image_paths(folder, limit)]


class PatchSampler:
    """Random aligned (LR, HR) patches, with flips and 90-degree rotations.

    Images are held as uint8 to keep DIV2K (800 x ~2K images) in about 3 GB of RAM.
    """

    def __init__(self, folder: Path, lr_patch: int, limit: int | None = None, seed: int = 0):
        self.lr_patch = lr_patch
        self.rng = np.random.default_rng(seed)
        self.lr: list[np.ndarray] = []
        self.hr: list[np.ndarray] = []
        paths = image_paths(folder, limit)
        for i, p in enumerate(paths):
            lr, hr = lr_hr_pair(p)
            self.lr.append(np.rint(lr * 255).astype(np.uint8))
            self.hr.append(np.rint(hr * 255).astype(np.uint8))
            print(f"\r  loaded {i + 1}/{len(paths)} training images", end="", flush=True)
        print()

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.lr_patch
        lrs, hrs = [], []
        for _ in range(n):
            i = int(self.rng.integers(len(self.lr)))
            lr, hr = self.lr[i], self.hr[i]
            y = int(self.rng.integers(lr.shape[0] - p + 1))
            x = int(self.rng.integers(lr.shape[1] - p + 1))
            a = lr[y:y + p, x:x + p]
            b = hr[2 * y:2 * y + 2 * p, 2 * x:2 * x + 2 * p]
            k = int(self.rng.integers(4))
            a, b = np.rot90(a, k), np.rot90(b, k)
            if self.rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            lrs.append(a)
            hrs.append(b)
        lr_t = torch.from_numpy(np.stack(lrs).astype(np.float32) / 255.0)[:, None]
        hr_t = torch.from_numpy(np.stack(hrs).astype(np.float32) / 255.0)[:, None]
        return lr_t, hr_t


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the SR datasets into data/.")
    ap.add_argument("names", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args()
    for name in args.names:
        print(f"{name}: {fetch(name)}")


if __name__ == "__main__":
    main()
