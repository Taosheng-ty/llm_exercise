"""
Solution for Exercise 01: Learning Rate Scheduler
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineDecayLR(_LRScheduler):
    """Warmup + cosine decay learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / self.warmup_steps if self.warmup_steps > 0 else base_lr
            elif step >= self.total_steps:
                lr = self.min_lr
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


class WarmupLinearDecayLR(_LRScheduler):
    """Warmup + linear decay learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / self.warmup_steps if self.warmup_steps > 0 else base_lr
            elif step >= self.total_steps:
                lr = self.min_lr
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = base_lr - (base_lr - self.min_lr) * progress
            lrs.append(lr)
        return lrs
