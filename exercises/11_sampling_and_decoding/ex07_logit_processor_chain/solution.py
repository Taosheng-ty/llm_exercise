"""Solution for Exercise 07: Logit Processor Chain (NumPy)

Inspired by slime's generate() sampling params pipeline.
"""

import numpy as np
from abc import ABC, abstractmethod


def _softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class LogitProcessor(ABC):
    @abstractmethod
    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TemperatureProcessor(LogitProcessor):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.temperature == 0:
            result = np.full_like(logits, -np.inf)
            result[np.argmax(logits)] = 1e6
            return result
        return logits / self.temperature


class TopKProcessor(LogitProcessor):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.k >= len(logits):
            return logits
        result = logits.copy()
        # Find the k-th largest value
        top_k_indices = np.argpartition(logits, -self.k)[-self.k:]
        threshold = np.min(logits[top_k_indices])
        result[logits < threshold] = -np.inf
        return result


class TopPProcessor(LogitProcessor):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_probs = _softmax(sorted_logits)
        cumulative_probs = np.cumsum(sorted_probs)

        # Mask tokens where cumulative_prob - prob > p (shift check)
        mask = (cumulative_probs - sorted_probs) > self.p
        # Always keep at least the top token
        mask[0] = False

        result = logits.copy()
        masked_indices = sorted_indices[mask]
        result[masked_indices] = -np.inf
        return result


class RepetitionPenaltyProcessor(LogitProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        result = logits.copy()
        if len(input_ids) == 0:
            return result
        unique_ids = np.unique(input_ids)
        for idx in unique_ids:
            if idx < len(result):
                if result[idx] > 0:
                    result[idx] = result[idx] / self.penalty
                else:
                    result[idx] = result[idx] * self.penalty
        return result


class LogitProcessorChain:
    def __init__(self, processors: list):
        self.processors = processors

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        result = logits.copy()
        for processor in self.processors:
            result = processor(input_ids, result)
        return result
