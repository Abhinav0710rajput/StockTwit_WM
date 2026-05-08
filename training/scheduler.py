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
    """
    β schedule: hold at beta_start for hold_steps, then linearly anneal to
    beta_end over total_steps additional steps.

    Setting hold_steps > 0 lets the reconstruction loss stabilize before the
    KL penalty is applied, which reduces posterior collapse risk.
    """

    def __init__(
        self,
        beta_start: float = 0.1,
        beta_end: float = 0.25,
        total_steps: int = 50_000,
        hold_steps: int = 0,
    ) -> None:
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.total_steps = total_steps
        self.hold_steps  = hold_steps
        self._step = 0

    def _compute(self, step: int) -> float:
        if step <= self.hold_steps:
            return self.beta_start
        t = min((step - self.hold_steps) / max(self.total_steps, 1), 1.0)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def step(self) -> float:
        self._step += 1
        return self._compute(self._step)

    @property
    def value(self) -> float:
        return self._compute(self._step)

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["_step"]


class CyclicalBetaScheduler:
    """
    Cyclical KL annealing (Fu et al. ICLR 2019).

    Each cycle: ramp β from 0 → beta_max over ramp_fraction of the cycle,
    then hold at beta_max for the remainder. After hold_cycles initial full-hold
    cycles at beta_start, cycling begins.

    Why: linear ramp keeps squeezing KL after reconstruction saturates. Cycling
    gives the model bursts of unconstrained reconstruction learning between
    regularization phases, which breaks the plateau.
    """

    def __init__(
        self,
        beta_start: float = 0.1,
        beta_max: float = 0.15,
        cycle_steps: int = 10_000,
        ramp_fraction: float = 0.5,
        hold_steps: int = 0,
    ) -> None:
        self.beta_start    = beta_start
        self.beta_max      = beta_max
        self.cycle_steps   = cycle_steps
        self.ramp_fraction = ramp_fraction
        self.hold_steps    = hold_steps
        self._step = 0

    def _compute(self, step: int) -> float:
        if step <= self.hold_steps:
            return self.beta_start
        cycle_step = (step - self.hold_steps) % self.cycle_steps
        ramp_end   = int(self.cycle_steps * self.ramp_fraction)
        if cycle_step < ramp_end:
            t = cycle_step / max(ramp_end, 1)
            return t * self.beta_max
        return self.beta_max

    def step(self) -> float:
        self._step += 1
        return self._compute(self._step)

    @property
    def value(self) -> float:
        return self._compute(self._step)

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["_step"]


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
