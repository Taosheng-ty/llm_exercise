"""
Exercise 03: Identity Preference Optimization (IPO) Loss

Difficulty: Medium
Framework: PyTorch

Background:
    IPO (Azar et al., 2023) is an alternative to DPO that avoids the overfitting
    issues of DPO by directly optimizing a squared loss on the preference margin.

    While DPO uses: loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    IPO uses: loss = ((log_ratio_chosen - log_ratio_rejected) - 1/(2*beta))^2

    where log_ratio = log(pi(y|x)) - log(pi_ref(y|x)) for chosen/rejected y.

    The key difference: IPO penalizes the margin for deviating from a target
    value 1/(2*beta), rather than pushing it to infinity as DPO can.

Implement:
    ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta) -> (loss, metrics)

Args:
    policy_chosen_logps: (B,) log probs of chosen under policy
    policy_rejected_logps: (B,) log probs of rejected under policy
    ref_chosen_logps: (B,) log probs of chosen under reference
    ref_rejected_logps: (B,) log probs of rejected under reference
    beta: float - regularization strength

Returns:
    loss: scalar tensor, mean squared error from target margin
    metrics: dict with keys:
        'log_ratio_chosen': (B,) = policy_chosen_logps - ref_chosen_logps
        'log_ratio_rejected': (B,) = policy_rejected_logps - ref_rejected_logps
        'margin': (B,) = log_ratio_chosen - log_ratio_rejected
        'target': float = 1 / (2 * beta)
"""

import torch


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute IPO loss.

    Returns:
        (loss, metrics_dict)
    """
    # TODO: Implement IPO loss
    # Hint 1: log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    # Hint 2: log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    # Hint 3: margin = log_ratio_chosen - log_ratio_rejected
    # Hint 4: target = 1.0 / (2.0 * beta)
    # Hint 5: loss = ((margin - target) ** 2).mean()
    raise NotImplementedError("Implement ipo_loss")
