"""
Solution for Exercise 07: Throughput Calculator
"""

import numpy as np


def tokens_per_second(
    batch_size: int, seq_len: int, num_gpus: int, step_time_seconds: float
) -> float:
    total_tokens = batch_size * seq_len * num_gpus
    return total_tokens / step_time_seconds


def samples_per_second(
    batch_size: int, num_gpus: int, step_time_seconds: float
) -> float:
    total_samples = batch_size * num_gpus
    return total_samples / step_time_seconds


def achieved_tflops(
    model_flops_per_token: float,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    step_time_seconds: float,
) -> float:
    """Return achieved TFLOPs/s for training (3x forward)."""
    total_flops = model_flops_per_token * batch_size * seq_len * num_gpus
    # Training: 3x forward (forward + backward where backward ~ 2x forward)
    training_flops = 3 * total_flops
    # Convert to TFLOPs
    tflops = training_flops / 1e12
    # Per second
    return tflops / step_time_seconds


def model_flops_utilization(
    achieved_tflops_per_sec: float,
    peak_tflops_per_gpu: float,
    num_gpus: int,
) -> float:
    """Return MFU as a fraction (0 to 1)."""
    total_peak = peak_tflops_per_gpu * num_gpus
    return achieved_tflops_per_sec / total_peak


def compute_all_metrics(
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    step_time_seconds: float,
    model_flops_per_token: float,
    peak_tflops_per_gpu: float,
) -> dict:
    """Compute all throughput metrics."""
    tok_sec = tokens_per_second(batch_size, seq_len, num_gpus, step_time_seconds)
    samp_sec = samples_per_second(batch_size, num_gpus, step_time_seconds)
    tflops_val = achieved_tflops(
        model_flops_per_token, batch_size, seq_len, num_gpus, step_time_seconds
    )
    mfu_val = model_flops_utilization(tflops_val, peak_tflops_per_gpu, num_gpus)

    return {
        "tokens_per_sec": tok_sec,
        "samples_per_sec": samp_sec,
        "tflops": tflops_val,
        "mfu": mfu_val,
    }
