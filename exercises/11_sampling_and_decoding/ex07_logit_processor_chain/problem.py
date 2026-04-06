"""Exercise 07: Logit Processor Chain (Medium) — NumPy

In production LLM serving (like slime's generate() with sampling params),
logit processing is done via a chain of composable processors. Each processor
modifies the logits in sequence before sampling.

Implement the following classes using numpy:

1. LogitProcessor (abstract base):
   - __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray

2. TemperatureProcessor(temperature: float):
   - Divides logits by temperature (handle temperature=0 as argmax)

3. TopKProcessor(k: int):
   - Keeps only top-k logits, sets rest to -inf

4. TopPProcessor(p: float):
   - Keeps smallest set of tokens with cumulative prob >= p

5. RepetitionPenaltyProcessor(penalty: float):
   - Applies repetition penalty to tokens in input_ids

6. LogitProcessorChain(processors: list[LogitProcessor]):
   - __call__ applies each processor in sequence
   - Maintains the chain as a list of processors

Args for __call__:
    input_ids: np.ndarray of shape (seq_len,) — previously generated token IDs
    logits: np.ndarray of shape (vocab_size,) — raw logits for next token

Returns:
    np.ndarray of shape (vocab_size,) — processed logits
"""

import numpy as np
from abc import ABC, abstractmethod


class LogitProcessor(ABC):
    @abstractmethod
    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TemperatureProcessor(LogitProcessor):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # TODO: Implement temperature scaling
        raise NotImplementedError


class TopKProcessor(LogitProcessor):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # TODO: Implement top-k filtering
        raise NotImplementedError


class TopPProcessor(LogitProcessor):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # TODO: Implement top-p filtering
        raise NotImplementedError


class RepetitionPenaltyProcessor(LogitProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # TODO: Implement repetition penalty
        raise NotImplementedError


class LogitProcessorChain:
    def __init__(self, processors: list):
        self.processors = processors

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # TODO: Apply each processor in sequence
        raise NotImplementedError
