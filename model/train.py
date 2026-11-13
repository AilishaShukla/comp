"""Train the FP32 ESPCN x2 on DIV2K (Y channel).

    python -m model.train                          # full run, writes model/espcn_fp32.pt
    python -m model.train --iters 2000 --n-train 20  # quick smoke test

Validation PSNR on Set5 is logged every --val-every iterations; the best
checkpoint is kept.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model.data import PatchSampler, fetch, hr_dir
from model.espcn import ESPCN
from model.eval_psnr import evaluate

DEFAULT_CKPT = Path(__file__).resolve().parent / "espcn_fp32.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-patch", type=int, default=32, help="LR patch size (HR patch is 2x)")
    ap.add_argument("--n-train", type=int, default=None, help="use only the first N DIV2K images")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    fetch("set5")
    sampler = PatchSampler(fetch("div2k"), args.lr_patch, limit=args.n_train, seed=args.seed)

    model = ESPCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr 1e-3 -> 1e-4 for the last 20 % of training (docs/06 section 1)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.8 * args.iters)], gamma=0.1)
    loss_fn = torch.nn.L1Loss()

    best = float("-inf")
    t0 = time.time()
    running = 0.0
    for it in range(1, args.iters + 1):
        lr_b, hr_b = sampler.batch(args.batch)
        loss = loss_fn(model(lr_b.to(device)), hr_b.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if it % args.val_every == 0 or it == args.iters:
            cpu_model = ESPCN()
            cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            r = evaluate(cpu_model, None, "set5")
            tag = ""
            if r["fp32"] > best:
                best = r["fp32"]
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cpu_model.state_dict(), args.out)
                tag = "  * saved"
            print(f"iter {it:7d}  loss {running / args.val_every:.5f}  "
                  f"Set5 {r['fp32']:.2f} dB (bicubic {r['bicubic']:.2f})  "
                  f"{time.time() - t0:6.0f}s{tag}", flush=True)
            running = 0.0

    print(f"best Set5 PSNR {best:.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
