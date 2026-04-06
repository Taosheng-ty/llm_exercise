"""Solution for Exercise 08: Stop Criteria for Generation (NumPy)"""

import numpy as np


class StopOnToken:
    def __init__(self, stop_token_id: int):
        self.stop_token_id = stop_token_id

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        if not generated_ids:
            return False
        return generated_ids[-1] == self.stop_token_id


class StopOnMaxLength:
    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        return len(generated_ids) >= self.max_new_tokens


class StopOnString:
    def __init__(self, stop_string: str):
        self.stop_string = stop_string

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        return self.stop_string in decoded_text


class StopCriteriaChain:
    def __init__(self, criteria: list):
        self.criteria = criteria

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        return any(c(generated_ids, decoded_text) for c in self.criteria)
