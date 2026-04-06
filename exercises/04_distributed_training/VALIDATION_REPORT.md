# Validation Report: 04_distributed_training

## Exercise 01: GPU Placement for Actor-Critic Training

- **Solvable from problem description alone:** Yes
- **Tests passed:** 7 / 7
- **Problem description issues:** None. The rules are clearly enumerated, the allocation order is specified, and edge cases (colocate, no critic) are well-defined. The required ValueError messages are implied but the exact match strings ("Not enough GPUs", "Colocate mode requires") are only discoverable from the test file -- the problem description should specify the exact error messages expected.
- **Test case issues:** None. Good coverage of happy paths (basic, colocate, no critic, exact count) and error paths (insufficient GPUs, colocate with too-small actor). The overlap disjointness test is a nice structural check.
- **Difficulty rating:** Easy. Straightforward conditional logic with simple arithmetic. Appropriate for an introductory distributed training exercise.
- **Suggestions for improvement:**
  - Specify the exact ValueError message strings in the problem description so students do not have to guess.
  - Consider adding a test for colocate=True with use_critic=False.
  - Could add a test where total_gpus exceeds what is needed (leftover GPUs) to verify only the required GPUs are allocated.

---

## Exercise 02: Model Weight Sharding and Gathering

- **Solvable from problem description alone:** Yes
- **Tests passed:** 7 / 7
- **Problem description issues:** None. The padding strategy (pad with zeros along axis 0 to make divisible) is clearly stated. The return types are well-specified.
- **Test case issues:** None. Good coverage including even splits, uneven splits with padding verification, roundtrip tests, single shard, metadata shape check, and 1D tensors.
- **Difficulty rating:** Easy-Medium. Requires familiarity with numpy (pad, split, concatenate) but the algorithm is straightforward. Appropriate difficulty.
- **Suggestions for improvement:**
  - Consider adding a test for 0-dimensional or scalar tensors (edge case).
  - Could test with num_shards > dim-0 size (e.g., 2 rows split into 4 shards) to verify padding works for extreme cases.
  - Could add a test verifying that shard dtypes match the original tensor dtype.

---

## Exercise 03: Async Training Scheduler

- **Solvable from problem description alone:** Yes
- **Tests passed:** 7 / 7
- **Problem description issues:** The problem description is clear about the constraints (one generate at a time, one train at a time, different types can overlap). The tie-breaking rule (sort by task name) is mentioned in the test comments but not explicitly in the problem description -- it should be stated in the docstring.
- **Test case issues:** None. The test_sequential_same_type test relies on alphabetical tie-breaking (gen1 before gen2 at time 0), which is reasonable but should be documented. The async prefetch pattern and three-stage pipeline tests are excellent real-world scenarios.
- **Difficulty rating:** Medium. Requires implementing a scheduling algorithm with resource constraints and dependency resolution. The greedy approach (pick earliest-available ready task) works but requires careful thought. Well-calibrated difficulty.
- **Suggestions for improvement:**
  - Explicitly state the tie-breaking rule in the problem description: "When multiple tasks can start at the same time, break ties alphabetically by task name."
  - The greedy scheduling approach (always pick the earliest-startable task) happens to produce optimal results for all test cases, but the problem does not clarify whether the schedule must be optimal or just valid. Consider clarifying whether "minimize total time" is a hard requirement or whether any valid schedule is accepted.
  - Consider adding a test with tasks given in non-alphabetical order to verify the scheduler handles input ordering correctly.

---

## Overall Summary

| Exercise | Solvable | Tests Passed | Difficulty |
|----------|----------|-------------|------------|
| ex01_gpu_placement | Yes | 7/7 | Easy |
| ex02_weight_sharding | Yes | 7/7 | Easy-Medium |
| ex03_async_training_scheduler | Yes | 7/7 | Medium |

**Overall quality:** High. All three exercises have clear problem descriptions, well-defined function signatures, and comprehensive test suites. The exercises form a nice progression from simple allocation logic to numpy array manipulation to algorithm design. The main improvement needed across all exercises is making implicit requirements (exact error messages, tie-breaking rules) explicit in the problem descriptions rather than only discoverable through the tests.
