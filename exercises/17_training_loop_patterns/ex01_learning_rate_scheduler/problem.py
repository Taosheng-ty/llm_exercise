"""
Exercise 01: Learning Rate Scheduler

Difficulty: Medium
Framework: PyTorch

Background:
    Modern LLM training uses learning rate schedules with a warmup phase
    followed by decay. During warmup, the LR linearly increases from 0 to
    the base LR over W steps. After warmup, the LR decays (cosine or linear)
    toward a minimum LR.

    Reference: slime training uses warmup + decay schedules for stable
    optimization of large language models.

Implement two schedulers as subclasses of torch.optim.lr_scheduler._LRScheduler:

1. WarmupCosineDecayLR:
   - Linear warmup from 0 to base_lr over warmup_steps
   - Cosine decay from base_lr to min_lr over remaining steps (total_steps - warmup_steps)
   - After total_steps, LR stays at min_lr

2. WarmupLinearDecayLR:
   - Linear warmup from 0 to base_lr over warmup_steps
   - Linear decay from base_lr to min_lr over remaining steps
   - After total_steps, LR stays at min_lr

Args (constructor):
    optimizer: torch.optim.Optimizer
    warmup_steps: int - number of warmup steps
    total_steps: int - total number of training steps
    min_lr: float - minimum learning rate after decay (default 0.0)

The get_lr() method should return a list of learning rates for each param group.
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
        # TODO: Implement warmup + cosine decay
        # Hint 1: self.last_epoch gives the current step number
        # Hint 2: During warmup (step < warmup_steps): lr = base_lr * step / warmup_steps
        # Hint 3: During decay: use cosine from base_lr to min_lr
        #         progress = (step - warmup_steps) / (total_steps - warmup_steps)
        #         lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        # Hint 4: After total_steps, return min_lr
        raise NotImplementedError("Implement WarmupCosineDecayLR.get_lr")


class WarmupLinearDecayLR(_LRScheduler):
    """Warmup + linear decay learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # TODO: Implement warmup + linear decay
        # Hint 1: Warmup phase is the same as cosine variant
        # Hint 2: During decay: lr = base_lr - (base_lr - min_lr) * progress
        #         where progress = (step - warmup_steps) / (total_steps - warmup_steps)
        raise NotImplementedError("Implement WarmupLinearDecayLR.get_lr")
