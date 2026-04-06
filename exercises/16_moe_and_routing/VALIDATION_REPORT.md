# Validation Report: 16_moe_and_routing

## Per-Exercise Results

### ex01_top_k_routing
- **Result**: 6/6 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: None
- **Difficulty rating (Medium)**: Accurate. The problem is straightforward -- matrix multiply, topk, softmax on selected values. Good introductory MoE exercise.

### ex02_load_balancing_loss
- **Result**: 5/5 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: None. The formula is clearly stated in the docstring with all terms defined.
- **Difficulty rating (Medium)**: Accurate. Requires understanding the formula and correctly computing f_i (token fraction) and P_i (mean softmax probability).

### ex03_expert_parallel_dispatch
- **Result**: 5/5 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: None. The dispatch-gather pattern is well described.
- **Difficulty rating (Hard)**: Slightly generous -- the concept is clear and the implementation is a standard loop-based gather/scatter. Could be "Medium" unless the intent is for students to write a more optimized version.

### ex04_routing_replay
- **Result**: 9/9 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: The tests directly access internal attributes (e.g., `cache.top_indices_list`, `cache.forward_index`, `cache.backward_index`), which constrains the implementation to use those exact attribute names. This is documented implicitly in the test but not in the problem description. A minor clarity issue -- the problem could explicitly state the expected attribute names or the tests could avoid accessing internals.
- **Difficulty rating (Medium)**: Accurate. Conceptually simple caching pattern, but the multi-stage interface adds moderate complexity.

### ex05_shared_expert
- **Result**: 7/7 tests passed
- **Solvable from description alone**: Yes, with caveats.
- **Issues found**: The tests access specific attribute names (`layer.shared_expert`, `layer.routed_experts`, `layer.router`) and expect `router` to have a `.weight` attribute (i.e., be an `nn.Linear`). These naming conventions are not specified in `problem.py`. A student who names things differently (e.g., `self.gate` instead of `self.router`) would fail. The problem description should either specify these attribute names or the tests should be less coupled to internals.
- **Difficulty rating (Medium)**: Accurate. Requires combining routing logic with nn.Module architecture.

### ex06_expert_frequency_analysis
- **Result**: 9/9 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: None. Clear function signatures and well-defined metrics.
- **Difficulty rating (Easy)**: Accurate. Pure numpy, straightforward counting and statistics.

### ex07_capacity_factor
- **Result**: 8/8 tests passed
- **Solvable from description alone**: Yes
- **Issues found**: The capacity formula uses `total_tokens / num_experts` but with top_k > 1, the correct denominator involves total assignments (`num_tokens * top_k / num_experts`). The test `test_preserves_kept_weights` uses `capacity = 1.0 * (2*2)/2 = 2` in its comment, confirming total_assignments-based capacity. The problem description says `total_tokens / num_experts` which is slightly ambiguous when top_k > 1. However, the tests make the intended behavior clear.
- **Difficulty rating (Medium)**: Accurate. Requires careful iteration order and capacity tracking.

## Summary

| Exercise | Tests | Passed | Solvable | Difficulty Match |
|----------|-------|--------|----------|-----------------|
| ex01_top_k_routing | 6 | 6 | Yes | Yes |
| ex02_load_balancing_loss | 5 | 5 | Yes | Yes |
| ex03_expert_parallel_dispatch | 5 | 5 | Yes | Slightly overrated |
| ex04_routing_replay | 9 | 9 | Yes | Yes |
| ex05_shared_expert | 7 | 7 | Yes* | Yes |
| ex06_expert_frequency_analysis | 9 | 9 | Yes | Yes |
| ex07_capacity_factor | 8 | 8 | Yes | Yes |

**Total: 49/49 tests passed across all 7 exercises.**

## Overall Quality Assessment

**Strengths:**
- Well-chosen progression of MoE concepts from basic routing to advanced patterns (shared experts, capacity limiting, replay caching).
- Clear problem descriptions with explicit mathematical formulas where needed.
- Good test coverage including edge cases (empty experts, uniform routing, skewed routing).
- Practical relevance to real MoE training pipelines (load balancing, expert collapse detection, routing replay for RLHF).

**Suggestions for Improvement:**
1. **ex04 and ex05**: Tests access internal attribute names (`top_indices_list`, `forward_index`, `backward_index`, `shared_expert`, `routed_experts`, `router`). Either document these required names in `problem.py` or refactor tests to use the public API only.
2. **ex07**: The capacity formula in the problem description (`total_tokens / num_experts`) is ambiguous for top_k > 1. Clarify that the denominator should account for total assignments (`num_tokens * top_k / num_experts`) or state explicitly how top_k interacts with capacity.
3. **ex03**: Consider revising difficulty to "Medium" or adding a requirement for batched (non-loop) dispatch to justify "Hard".
4. All exercises are solvable from the problem description alone, which is the key quality bar. The exercise set is well-designed overall.
