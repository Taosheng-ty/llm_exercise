"""
Exercise 03: Compression-Based Repetition Detection - Solution

Reference: slime/utils/metric_utils.py - compression_ratio(), has_repetition()
"""

import zlib


def compression_ratio(text: str) -> float:
    """
    Compute the compression ratio of a text string using zlib.
    """
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, 9)
    return len(raw) / len(compressed)


def has_repetition(text: str, threshold: float = 10.0, tail_length: int = 10000) -> bool:
    """
    Detect if a text contains excessive repetition using compression ratio.
    """
    if len(text) <= tail_length:
        return False
    return compression_ratio(text[-tail_length:]) > threshold


def ngram_repetition_fraction(text: str, n: int = 3) -> float:
    """
    Compute the fraction of repeated n-grams (character-level) in the text.
    """
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return (total - unique) / total


def detect_repetition(
    text: str,
    compression_threshold: float = 10.0,
    ngram_threshold: float = 0.7,
    ngram_size: int = 5,
) -> dict[str, bool | float]:
    """
    Combined repetition detection using both compression ratio and n-gram analysis.
    """
    cr = compression_ratio(text)
    nr = ngram_repetition_fraction(text, ngram_size)
    is_rep_comp = cr > compression_threshold
    is_rep_ngram = nr > ngram_threshold

    return {
        "compression_ratio": cr,
        "ngram_repetition": nr,
        "is_repetitive_compression": is_rep_comp,
        "is_repetitive_ngram": is_rep_ngram,
        "is_repetitive": is_rep_comp or is_rep_ngram,
    }
