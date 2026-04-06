# Validation Report: 08 Attention Mechanisms

## Per-Exercise Results

### Ex01: Scaled Dot-Product Attention
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** None. The problem description is clear and complete.
- **Difficulty rating (Medium):** Accurate. Straightforward application of the attention formula.

### Ex02: Multi-Head Attention
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** None. Well-described module with clear steps.
- **Difficulty rating (Medium):** Accurate. Requires understanding of reshaping and projection layers, but the steps are well-laid-out.

### Ex03: Flash Attention via Tiling
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes, but requires prior knowledge of the online softmax trick.
- **Issues found:** None. The hint about tracking running max (m) and sum (l) is helpful, though a student unfamiliar with the online softmax algorithm may struggle without more detail on the rescaling step.
- **Difficulty rating (Hard):** Accurate. The online softmax trick with running max/sum rescaling is non-trivial to implement correctly.

### Ex04: Causal Mask
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** None. Very clear specification of mask convention (True = masked).
- **Difficulty rating (Easy):** Accurate. Direct use of torch.triu and masked_fill.

### Ex05: Rotary Positional Embedding (RoPE)
- **Tests:** 6/6 passed (after 1 retry)
- **Solvable from description alone:** Yes, but with a subtle pitfall.
- **Issues found:** The description says "Returns cos and sin tensors of shape (max_seq_len, head_dim)" and describes frequencies per dimension pair. A common mistake is using `repeat(1, 2)` instead of `repeat_interleave(2, dim=-1)` to expand from half_dim to head_dim. The description could clarify that each pair of adjacent dimensions shares the same frequency (i.e., cos/sin values at indices 2i and 2i+1 are identical). The current description mentions pairs but does not explicitly state the repeat pattern for the output shape.
- **Difficulty rating (Hard):** Accurate. The interleaving pattern and the rotation formula require careful attention to indexing. Subtle bugs (like repeat vs repeat_interleave) are easy to introduce.

### Ex06: Grouped Query Attention (GQA)
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** None. The input shape convention (flattened heads) is clearly specified with the grouping relationship well-documented.
- **Difficulty rating (Medium):** Accurate. The main challenge is the KV head expansion/repeat logic.

### Ex07: KV Cache
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** The `__init__` signature in problem.py only takes `max_seq_len`, but the test `test_cache_reset` checks `cache.seq_len`, which is an implementation detail not mentioned in the problem description. This is a minor coupling issue -- the test assumes the attribute name `seq_len` exists on the cache object, which should be documented in the problem.
- **Difficulty rating (Medium):** Accurate. Straightforward data structure plus standard attention computation.

### Ex08: Sliding Window Attention
- **Tests:** 5/5 passed
- **Solvable from description alone:** Yes
- **Issues found:** None. The window specification is clear: token i attends to max(0, i-window_size+1) through i.
- **Difficulty rating (Medium):** Accurate. Slightly more than causal masking but the pattern is well-specified.

## Overall Summary

| Exercise | Tests | Pass/Fail | Solvable? | Retries | Difficulty Match |
|----------|-------|-----------|-----------|---------|-----------------|
| Ex01 Scaled Dot-Product Attention | 5/5 | PASS | Yes | 0 | Yes (Medium) |
| Ex02 Multi-Head Attention | 5/5 | PASS | Yes | 0 | Yes (Medium) |
| Ex03 Flash Attention Tiling | 5/5 | PASS | Yes | 0 | Yes (Hard) |
| Ex04 Causal Mask | 5/5 | PASS | Yes | 0 | Yes (Easy) |
| Ex05 Rotary Positional Embedding | 6/6 | PASS | Yes | 1 | Yes (Hard) |
| Ex06 Grouped Query Attention | 5/5 | PASS | Yes | 0 | Yes (Medium) |
| Ex07 KV Cache | 5/5 | PASS | Yes | 0 | Yes (Medium) |
| Ex08 Sliding Window Attention | 5/5 | PASS | Yes | 0 | Yes (Medium) |

**Total: 41/41 tests passed across 8 exercises.**

## Quality Assessment

**Overall quality: High.** All 8 exercises are well-structured, solvable from their problem descriptions alone, and have appropriate difficulty ratings. The test suites are thorough, covering shape checks, correctness against reference implementations, edge cases, and mathematical properties.

## Suggestions

1. **Ex05 (RoPE):** Clarify in the problem description that the returned cos/sin tensors of shape `(max_seq_len, head_dim)` have repeated values for each pair -- i.e., `cos[:, 2i] == cos[:, 2i+1]`. This would reduce a common source of bugs without giving away the solution.

2. **Ex07 (KV Cache):** Document the expected `seq_len` attribute in the class docstring or problem description, since the test checks `cache.seq_len == 0` after reset. Alternatively, add a `__len__` method requirement instead of relying on attribute access.

3. **Ex03 (Flash Attention):** Consider adding a brief note or reference to the online softmax algorithm (e.g., "look up the safe/online softmax trick") for students who may not be familiar with it. The current description mentions "track running max (m) and sum (l)" but does not explain the rescaling formula, which is the core algorithmic insight.
