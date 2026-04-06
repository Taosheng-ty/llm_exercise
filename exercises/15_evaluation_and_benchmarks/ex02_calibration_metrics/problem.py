"""
Exercise 2: Compute Model Calibration (Expected Calibration Error)

A well-calibrated model's predicted confidence should match its actual accuracy.
For example, among all predictions where the model says "I'm 80% confident,"
it should be correct ~80% of the time.

Expected Calibration Error (ECE) bins predictions by confidence and measures
the gap between average confidence and accuracy in each bin:

    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

Your task: implement compute_ece() and classify_calibration().

Difficulty: Medium
Framework: PyTorch
"""

import torch


def compute_ece(
    predicted_probs: torch.Tensor,
    true_labels: torch.Tensor,
    n_bins: int = 10,
) -> tuple[torch.Tensor, list[dict]]:
    """Compute Expected Calibration Error (ECE).

    Args:
        predicted_probs: Tensor of shape (N, C) with predicted class probabilities
                         (each row sums to ~1.0). C is the number of classes.
        true_labels: Tensor of shape (N,) with integer class labels in [0, C).
        n_bins: Number of equally-spaced confidence bins in [0, 1].

    Returns:
        A tuple of:
        - ece: Scalar tensor with the ECE value.
        - bin_stats: List of dicts, one per bin, each containing:
            - "bin_lower": float, lower bound of bin
            - "bin_upper": float, upper bound of bin
            - "count": int, number of samples in bin
            - "avg_confidence": float, mean predicted confidence in bin
            - "avg_accuracy": float, fraction of correct predictions in bin
            - "gap": float, |avg_accuracy - avg_confidence|

    Steps:
        1. Get the maximum predicted probability for each sample (confidence)
           and the predicted class (argmax).
        2. Compare predicted class to true_labels to get correctness (0/1).
        3. Bin samples by confidence into n_bins equal-width bins over [0, 1].
           Bin boundaries: bin_i covers (i/n_bins, (i+1)/n_bins].
           The first bin also includes the left boundary 0.0 (i.e., [0, 1/n_bins]).
        4. For each bin, compute average confidence, average accuracy, and gap.
        5. ECE is the weighted sum of gaps: sum(count_b / N * gap_b).
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_ece")


def classify_calibration(ece: float) -> str:
    """Classify model calibration quality based on ECE value.

    Args:
        ece: Expected Calibration Error value.

    Returns:
        One of: "well-calibrated" (ECE < 0.05),
                "moderately-calibrated" (0.05 <= ECE < 0.15),
                "poorly-calibrated" (ECE >= 0.15).
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement classify_calibration")
