"""
Training schedulers.

BetaScheduler       : linear anneal of KL weight β from start→end over N steps
CosineWarmupScheduler : cosine LR decay with optional linear warmup
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BetaScheduler:
    """Linearly anneals β from beta_start to beta_end over total_steps steps."""

    def __init__(
        self,
        beta_start: float = 0.1,
        beta_end: float = 1.0,
        total_steps: int = 50_000,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        t = min(self._step / max(self.total_steps, 1), 1.0)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    @property
    def value(self) -> float:
        t = min(self._step / max(self.total_steps, 1), 1.0)
        return self.beta_start + t * (self.beta_end - self.beta_start)


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine decay with linear warmup.

    lr goes: 0 → peak_lr (over warmup_steps), then cosine decay to min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / max(self.warmup_steps, 1)
            else:
                progress = (step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs
