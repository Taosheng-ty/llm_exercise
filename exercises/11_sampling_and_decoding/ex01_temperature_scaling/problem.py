"""Exercise 01: Temperature Scaling (Easy)

Temperature scaling is the simplest sampling control in LLM decoding.
Given raw logits from a language model, temperature controls the "sharpness"
of the probability distribution. Temperature is one of the most important knobs in
LLM inference and RL rollout generation. Lower temperatures produce more deterministic
outputs (useful for evaluation), while higher temperatures increase diversity (useful
for exploration during RL training, where diverse rollouts improve learning).

    scaled_logits = logits / temperature

- temperature < 1: sharper distribution (more confident, more deterministic)
- temperature = 1: unchanged distribution
- temperature > 1: flatter distribution (more random, more diverse)
- temperature → 0: equivalent to argmax (greedy decoding)

Implement `temperature_scale(logits, temperature)` that:
1. If temperature == 0, returns a one-hot distribution at the argmax position
   (set argmax logit to a large value like 1e6, rest to -inf)
2. Otherwise, returns logits / temperature

Args:
    logits: torch.Tensor of shape (batch_size, vocab_size) — raw logits
    temperature: float — temperature parameter (>= 0)

Returns:
    torch.Tensor of same shape — scaled logits
"""

import torch


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # TODO: Implement temperature scaling
    # Handle the special case of temperature == 0 (greedy/argmax)
    # For temperature > 0, simply divide logits by temperature
    raise NotImplementedError("Implement temperature_scale")
