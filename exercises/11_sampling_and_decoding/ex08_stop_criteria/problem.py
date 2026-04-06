"""Exercise 08: Stop Criteria for Generation (Easy) — NumPy

During text generation, we need to know when to stop. Multiple stop criteria
can be combined, and generation halts when ANY criterion is triggered.

Implement the following classes using numpy:

1. StopOnToken(stop_token_id: int):
   - __call__(generated_ids: list[int], decoded_text: str) -> bool
   - Returns True if the last generated token equals stop_token_id

2. StopOnMaxLength(max_new_tokens: int):
   - __call__(generated_ids: list[int], decoded_text: str) -> bool
   - Returns True if len(generated_ids) >= max_new_tokens

3. StopOnString(stop_string: str):
   - __call__(generated_ids: list[int], decoded_text: str) -> bool
   - Returns True if stop_string appears anywhere in decoded_text

4. StopCriteriaChain(criteria: list):
   - __call__(generated_ids: list[int], decoded_text: str) -> bool
   - Returns True if ANY criterion returns True (short-circuit OR)

Args for __call__:
    generated_ids: list[int] — token IDs generated so far (not including prompt)
    decoded_text: str — the decoded text of generated_ids

Returns:
    bool — True if generation should stop
"""

import numpy as np


class StopOnToken:
    def __init__(self, stop_token_id: int):
        self.stop_token_id = stop_token_id

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        # TODO: Check if last token matches stop_token_id
        raise NotImplementedError


class StopOnMaxLength:
    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        # TODO: Check if we've generated enough tokens
        raise NotImplementedError


class StopOnString:
    def __init__(self, stop_string: str):
        self.stop_string = stop_string

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        # TODO: Check if stop_string appears in decoded_text
        raise NotImplementedError


class StopCriteriaChain:
    def __init__(self, criteria: list):
        self.criteria = criteria

    def __call__(self, generated_ids: list[int], decoded_text: str) -> bool:
        # TODO: Return True if any criterion triggers
        raise NotImplementedError
