# Validation Report: 05_rollout_pipeline

## Summary

All 4 exercises were solved from the problem description alone. All 34 tests pass.

---

## Exercise 01: Rollout Data Source with Epoch Tracking

- **Solved from problem description alone:** Yes
- **Tests passed:** 8/8
- **Problem description issues:** None. The docstrings and TODO comments are clear and precise. The epoch tracking, wrap-around, and shuffle-seed behavior are all well specified.
- **Test case issues:** None. Good coverage of basic batching, epoch boundary wrap-around, reproducibility, reset, and error cases.
- **Difficulty rating:** Easy-Medium. Straightforward data structure with one subtle case (wrap-around).
- **Suggestions:** Consider adding a test for batch_size == len(data) exact exhaustion followed by another batch (tests the edge where remaining_in_epoch == 0 before a new get_batch call). The existing test_epoch_increments_on_exhaustion partially covers this but a dedicated test would strengthen it.

---

## Exercise 02: Experience Replay Buffer

- **Solved from problem description alone:** Yes
- **Tests passed:** 8/8
- **Problem description issues:** None. The FIFO eviction, priority sampling formula, and with-replacement behavior are all clearly stated.
- **Test case issues:** Minor: `test_fifo_eviction` directly accesses `buf.buffer`, which couples the test to internal implementation (the attribute name). The problem.py does not specify that the internal storage must be named `buffer`. A student using a different attribute name (e.g., `_buffer` or `storage`) would fail this test despite correct behavior.
- **Difficulty rating:** Easy-Medium. Standard replay buffer pattern.
- **Suggestions:** Either specify in problem.py that the internal list must be named `buffer`, or change the test to use a public method (e.g., iterate/inspect via sampling or a `to_list()` method).

---

## Exercise 03: Dynamic Sampling Filters

- **Solved from problem description alone:** Yes, but with one non-obvious requirement.
- **Tests passed:** 9/9
- **Problem description issues:** The `test_filter_chain_drop_reason_counts` test expects `len(counts) == 2` when two groups are dropped by `filter_identical_rewards` with rewards [1.0, 1.0] and [2.0, 2.0]. This implies the reason string must include the specific reward value (e.g., "identical rewards: 1.0" vs "identical rewards: 2.0") so the two reasons are distinct strings. This is NOT stated in the problem description. The problem only says to check if std < 1e-8 and return a reason containing "identical". A student who uses a generic reason like "identical rewards" would pass all tests except this one. This is the most significant quality issue found.
- **Test case issues:** The `test_filter_chain_drop_reason_counts` test has a comment "two different reasons (1.0 and 2.0)" which hints at the requirement but is insufficient -- the hint is in the test, not in the problem description where the student would look for guidance.
- **Difficulty rating:** Easy, except for the hidden reason-string requirement which is tricky.
- **Suggestions:** Add explicit guidance in `filter_identical_rewards` that the reason string should include the actual reward value, e.g., `"identical rewards: {value}"`. Alternatively, change the test to use two different filters so the two drop reasons are naturally different, which would test the counting mechanism without relying on implementation-specific reason formatting.

---

## Exercise 04: Best-of-N Sampling with Rejection

- **Solved from problem description alone:** Yes
- **Tests passed:** 9/9
- **Problem description issues:** None. The three strategies (greedy, weighted, rejection) are clearly defined. The weight formula with min-shift + epsilon is specified.
- **Test case issues:** None. Good mix of unit tests for individual strategies and batch-level integration tests with stats verification.
- **Difficulty rating:** Easy-Medium. Multiple functions but each is straightforward.
- **Suggestions:** Consider adding a test for `batch_best_of_n` with an empty `prompt_responses` list to verify edge case handling (division by zero in acceptance_rate). Also consider testing `weighted_sample` with all equal scores to ensure it still works (uniform fallback).

---

## Overall Assessment

The exercise set is well-designed and progressively builds on rollout pipeline concepts. The problem descriptions are generally clear with good function signatures and docstrings. The main quality issue is in Exercise 03 where the test for `drop_reason_counts` implicitly requires the reason string to contain the reward value, which is not documented in the problem. All exercises are solvable at roughly Easy-Medium difficulty, which seems appropriate for a module-level exercise set. No difficulty level was explicitly stated in the problems, so no comparison can be made.
