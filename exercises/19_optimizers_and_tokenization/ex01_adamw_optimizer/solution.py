"""Solution for Exercise 01: AdamW Optimizer from Scratch"""

import torch


class AdamW:
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Per-parameter state: first moment (m), second moment (v)
        self.state = {}
        for p in self.params:
            self.state[p] = {
                "step": 0,
                "m": torch.zeros_like(p.data),
                "v": torch.zeros_like(p.data),
            }

    def step(self):
        """Perform a single optimization step with decoupled weight decay."""
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data
            s = self.state[p]
            s["step"] += 1
            t = s["step"]

            # Update biased first moment estimate
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
            # Update biased second moment estimate
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * grad * grad

            # Bias correction
            m_hat = s["m"] / (1 - self.beta1 ** t)
            v_hat = s["v"] / (1 - self.beta2 ** t)

            # Adam update + decoupled weight decay
            # Decoupled weight decay: applied directly to param, not via gradient
            p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad = None
