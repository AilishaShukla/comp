"""ESPCN x2 super-resolution network (Shi et al., CVPR 2016 [R1]).

Changes from the paper, recorded in docs/05-decisions.md:
  * ReLU instead of tanh (D3), so the activation folds into the INT8 clamp.
  * Y channel only (D2).
"""
import torch
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.l1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.l2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(32, scale * scale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def features(self, x: torch.Tensor):
        """Return every intermediate activation (used for calibration)."""
        a1 = torch.relu(self.l1(x))
        a2 = torch.relu(self.l2(a1))
        a3 = self.l3(a2)
        return a1, a2, a3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.features(x)[2])
