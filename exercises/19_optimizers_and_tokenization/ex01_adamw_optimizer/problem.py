"""Exercise 01: AdamW Optimizer from Scratch (Medium, PyTorch)

AdamW is the standard optimizer for training large language models. Unlike the
original Adam with L2 regularization, AdamW applies DECOUPLED weight decay:
the decay is applied directly to the parameters, not through the gradient.

Original Adam + L2 reg:   grad += weight_decay * param   (coupled)
AdamW (decoupled):        param -= lr * weight_decay * param   (decoupled)

The AdamW update rule per parameter:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)       # bias correction
    v_hat = v_t / (1 - beta2^t)       # bias correction
    param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

Implement the `AdamW` optimizer class with:
    - __init__(params, lr, betas, eps, weight_decay)
    - step(): perform one optimization step
    - zero_grad(): zero all parameter gradients
"""

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
        """Initialize AdamW optimizer.

        Args:
            params: iterable of parameters to optimize (torch.nn.Parameter)
            lr: learning rate
            betas: coefficients for computing running averages of gradient
                   and its square (beta1, beta2)
            eps: term added to denominator for numerical stability
            weight_decay: decoupled weight decay coefficient
        """
        # TODO: store params as a list in self.params, store hyperparameters,
        # initialize self.state as a dict mapping each parameter to a dict with:
        #   "m": first moment (zeros_like), "v": second moment (zeros_like), "step": 0
        raise NotImplementedError("Implement AdamW.__init__")

    def step(self):
        """Perform a single optimization step.

        For each parameter with a gradient:
        1. Update biased first moment estimate (m)
        2. Update biased second moment estimate (v)
        3. Compute bias-corrected first and second moment estimates
        4. Update parameter with Adam step AND decoupled weight decay
        """
        # TODO: implement the AdamW update rule
        raise NotImplementedError("Implement AdamW.step")

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        # TODO: set .grad to None or zero for each parameter
        raise NotImplementedError("Implement AdamW.zero_grad")
