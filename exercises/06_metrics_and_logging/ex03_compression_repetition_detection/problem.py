"""
Exercise 03: Compression-Based Repetition Detection

In LLM outputs, repetitive text is a common failure mode (e.g., the model
generates the same phrase over and over). In RL-based LLM training, repetition
is a common degenerate behavior where the model learns to maximize length-based
rewards by repeating tokens. Detecting repetition during training enables
automatic filtering of degenerate rollouts and can be used as a penalty signal
in the reward function.

We can detect this using:

1. Compression ratio: highly repetitive text compresses very well (high ratio).
   ratio = len(original) / len(compressed)
   A high ratio (e.g., > 10) indicates repetitive content.

2. N-gram repetition: count how many n-grams appear more than once.
   repetition_fraction = (total n-grams - unique n-grams) / total n-grams

Reference: slime/utils/metric_utils.py - compression_ratio(), has_repetition()
"""

import zlib


def compression_ratio(text: str) -> float:
    """
    Compute the compression ratio of a text string using zlib.

    ratio = len(original_bytes) / len(compressed_bytes)

    A higher ratio means the text is more compressible (more repetitive).

    Args:
        text: Input text string.

    Returns:
        The compression ratio (float). Returns 0.0 for empty strings.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement compression_ratio")


def has_repetition(text: str, threshold: float = 10.0, tail_length: int = 10000) -> bool:
    """
    Detect if a text contains excessive repetition using compression ratio.

    For long texts, only check the last `tail_length` characters to focus
    on recent output (where repetition loops typically manifest).

    Args:
        text: Input text.
        threshold: Compression ratio above which text is considered repetitive.
        tail_length: Number of trailing characters to check for long texts.
            Only applies if len(text) > tail_length.

    Returns:
        True if the text is repetitive, False otherwise.
        Short texts (len <= tail_length) are never considered repetitive.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement has_repetition")


def ngram_repetition_fraction(text: str, n: int = 3) -> float:
    """
    Compute the fraction of repeated n-grams (character-level) in the text.

    For a text of length L, there are (L - n + 1) character n-grams.
    The repetition fraction is:
        (total_ngrams - unique_ngrams) / total_ngrams

    A value of 0.0 means all n-grams are unique (no repetition).
    A value close to 1.0 means almost all n-grams are duplicates.

    Args:
        text: Input text.
        n: The n-gram size.

    Returns:
        Repetition fraction between 0.0 and 1.0.
        Returns 0.0 if text is shorter than n characters.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement ngram_repetition_fraction")


def detect_repetition(
    text: str,
    compression_threshold: float = 10.0,
    ngram_threshold: float = 0.7,
    ngram_size: int = 5,
) -> dict[str, bool | float]:
    """
    Combined repetition detection using both compression ratio and n-gram analysis.

    Args:
        text: Input text.
        compression_threshold: Compression ratio above which text is considered repetitive.
        ngram_threshold: N-gram repetition fraction above which text is considered repetitive.
        ngram_size: Size of n-grams for repetition analysis.

    Returns:
        Dictionary with:
            - "compression_ratio": the computed compression ratio
            - "ngram_repetition": the n-gram repetition fraction
            - "is_repetitive_compression": bool, True if compression ratio > threshold
            - "is_repetitive_ngram": bool, True if ngram fraction > threshold
            - "is_repetitive": bool, True if EITHER method flags repetition

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement detect_repetition")
