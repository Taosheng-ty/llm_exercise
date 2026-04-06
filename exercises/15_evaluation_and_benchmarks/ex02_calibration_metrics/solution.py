"""
Solution for Exercise 2: Compute Model Calibration (Expected Calibration Error)
"""

import torch


def compute_ece(
    predicted_probs: torch.Tensor,
    true_labels: torch.Tensor,
    n_bins: int = 10,
) -> tuple[torch.Tensor, list[dict]]:
    """Compute Expected Calibration Error (ECE)."""
    confidences, predicted_classes = predicted_probs.max(dim=1)
    correct = (predicted_classes == true_labels).float()
    n_samples = len(true_labels)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece = torch.tensor(0.0)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i].item()
        bin_upper = bin_boundaries[i + 1].item()

        if i < n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        else:
            # Last bin includes the upper boundary
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        count = in_bin.sum().item()
        if count > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()
            gap = abs(avg_accuracy - avg_confidence)
            ece += (count / n_samples) * gap
        else:
            avg_confidence = 0.0
            avg_accuracy = 0.0
            gap = 0.0

        bin_stats.append({
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "count": int(count),
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "gap": gap,
        })

    return ece, bin_stats


def classify_calibration(ece: float) -> str:
    """Classify model calibration quality based on ECE value."""
    if ece < 0.05:
        return "well-calibrated"
    elif ece < 0.15:
        return "moderately-calibrated"
    else:
        return "poorly-calibrated"
