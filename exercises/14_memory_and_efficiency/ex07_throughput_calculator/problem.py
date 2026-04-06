"""
Exercise 07: Throughput Calculator (Easy, numpy)

Training throughput metrics help us understand how efficiently we are using hardware.
Key metrics include:
- Tokens per second: how many tokens are processed per second
- Samples per second: how many complete sequences are processed per second
- TFLOPs: teraFLOPs achieved (actual compute throughput)
- MFU (Model FLOPS Utilization): fraction of peak hardware FLOPS used by model compute

Reference: slime/utils/train_metric_utils.py log_perf_data_raw()

Your tasks:
-----------
1. Implement `tokens_per_second(batch_size, seq_len, num_gpus, step_time_seconds)`:
   - Total tokens per step = batch_size * seq_len * num_gpus
   - Return tokens_per_step / step_time_seconds

2. Implement `samples_per_second(batch_size, num_gpus, step_time_seconds)`:
   - Total samples per step = batch_size * num_gpus
   - Return samples_per_step / step_time_seconds

3. Implement `achieved_tflops(model_flops_per_token, batch_size, seq_len, num_gpus, step_time_seconds)`:
   - Total FLOPs per step = model_flops_per_token * batch_size * seq_len * num_gpus
   - Training: multiply by 3 (forward + backward)
   - Convert to TFLOPs (divide by 1e12)
   - Return TFLOPs / step_time_seconds (this gives TFLOPs/s)

4. Implement `model_flops_utilization(achieved_tflops_per_sec, peak_tflops_per_gpu, num_gpus)`:
   - MFU = achieved_tflops_per_sec / (peak_tflops_per_gpu * num_gpus)
   - Return as fraction (0 to 1)

5. Implement `compute_all_metrics(batch_size, seq_len, num_gpus, step_time_seconds, model_flops_per_token, peak_tflops_per_gpu)`:
   - Return a dict with keys: "tokens_per_sec", "samples_per_sec", "tflops", "mfu"
"""

import numpy as np


def tokens_per_second(
    batch_size: int, seq_len: int, num_gpus: int, step_time_seconds: float
) -> float:
    raise NotImplementedError


def samples_per_second(
    batch_size: int, num_gpus: int, step_time_seconds: float
) -> float:
    raise NotImplementedError


def achieved_tflops(
    model_flops_per_token: float,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    step_time_seconds: float,
) -> float:
    """Return achieved TFLOPs/s for training (3x forward)."""
    raise NotImplementedError


def model_flops_utilization(
    achieved_tflops_per_sec: float,
    peak_tflops_per_gpu: float,
    num_gpus: int,
) -> float:
    """Return MFU as a fraction (0 to 1)."""
    raise NotImplementedError


def compute_all_metrics(
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    step_time_seconds: float,
    model_flops_per_token: float,
    peak_tflops_per_gpu: float,
) -> dict:
    """
    Compute all throughput metrics.

    Returns dict with keys: "tokens_per_sec", "samples_per_sec", "tflops", "mfu"
    """
    raise NotImplementedError
