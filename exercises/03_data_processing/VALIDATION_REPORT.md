# Validation Report: 03_data_processing Exercises

**Validator:** Student Agent (Claude)
**Date:** 2026-04-06
**Method:** Solved each exercise from `problem.py` alone (did NOT read `solution.py`), then ran all tests.

---

## Exercise 01: Chat Template Builder

- **Solvable from problem description alone:** Yes
- **Tests passed:** 12 / 12
- **Problem description issues:** None. The docstrings are clear, the ChatML format is precisely specified with an example, and the edge cases (system message dedup, generation prompt, empty messages) are all documented.
- **Test case issues:** None. Good coverage of edge cases including empty messages, system message deduplication, and generation prompt appending.
- **Stated difficulty:** Easy
- **Actual difficulty assessment:** Easy. Matches the stated difficulty. Straightforward string formatting with a couple of conditional branches.
- **Suggestions for improvement:**
  - Could add a test for `add_generation_prompt=True` combined with `system_message` to test the combination of both features.
  - Could add a test where `build_messages` receives a list prompt with `as_conversation=False` to clarify that lists are always returned as-is regardless of the flag.

---

## Exercise 02: Sequence Packing

- **Solvable from problem description alone:** Yes
- **Tests passed:** 11 / 11
- **Problem description issues:** None. The FFD algorithm is well-described in the docstring. The return types (numpy arrays with specific shapes) are clearly specified.
- **Test case issues:** None. Tests are appropriately behavioral -- they check bin counts and total real tokens rather than exact token orderings, which is the right approach since FFD can produce different valid packings depending on tie-breaking.
- **Stated difficulty:** Medium
- **Actual difficulty assessment:** Easy-to-Medium. The FFD algorithm is clearly described step-by-step, making it more of a coding exercise than an algorithmic puzzle. The numpy array construction adds a small wrinkle. Matches stated difficulty.
- **Suggestions for improvement:**
  - The `compute_packing_efficiency` function description does not explicitly state whether it should call `pack_sequences` internally or compute efficiency independently. The test for `test_multiple_sequences` with `seqs = [[1,2,3], [4,5]]` and `max_seq_len=6` expects `5/6`, which implies the two sequences should pack into one bin. This works correctly with FFD but the coupling between the two functions could be made more explicit.
  - Consider adding a test where sequences of the same length still get packed correctly (tests tie-breaking in the sort).

---

## Exercise 03: Sequence Length Balancing with Karmarkar-Karp

- **Solvable from problem description alone:** Yes
- **Tests passed:** 10 / 10
- **Problem description issues:** The algorithm description is good but required careful reading. The key insight -- that merging pairs partition `i` of state0 with partition `k-1-i` of state1 -- could benefit from a small concrete example. The initial state construction ("sequence is placed in one partition and the rest are empty") is slightly ambiguous about which partition gets the sequence, though it does not matter due to the sorting step.
- **Test case issues:** The tests use relaxed assertions (`spread <= 2`, `spread < 120`) which is appropriate for a heuristic algorithm. However, the `test_equal_lengths` test asserts exact equality (`sums[0] == sums[1] == 20`), which could be fragile if an implementation produces valid but differently-ordered partitions.
- **Stated difficulty:** Hard
- **Actual difficulty assessment:** Hard. Matches the stated difficulty. The Karmarkar-Karp algorithm is non-trivial to implement correctly -- managing the heap with states containing k partitions, the merge logic, and keeping partitions sorted. The use of `heapq` (min-heap in Python) to simulate a max-heap also requires care.
- **Suggestions for improvement:**
  - Add a brief worked example in the docstring showing one or two merge steps.
  - Consider adding a test with `k_partitions` greater than the number of sequences to verify behavior (should produce some empty partitions).
  - Consider adding a test for a single-element `seqlen_list`.
  - The problem does not specify behavior for empty `seqlen_list`; a test for that edge case would be useful.

---

## Exercise 04: Loss Mask Generation for SFT Training

- **Solvable from problem description alone:** Yes
- **Tests passed:** 12 / 12
- **Problem description issues:** None. The algorithm is precisely described, and the inline example in the docstring is very helpful for understanding the expected output. The distinction between header tokens (masked out) and content tokens (masked in) is clear.
- **Test case issues:** None. Good coverage including multi-turn conversations, system+user+assistant combinations, empty assistant content, and multi-token role IDs. The `_make_turn` helper in tests is well-designed.
- **Stated difficulty:** Medium
- **Actual difficulty assessment:** Medium. Matches stated difficulty. The scanning/state-machine logic requires careful index management, but the algorithm is well-specified.
- **Suggestions for improvement:**
  - Consider adding a test where `im_start_id` or `im_end_id` appears as a regular content token (edge case for robustness).
  - Consider adding a test for a conversation that ends without a trailing newline after `im_end` (truncated sequence).
  - The `get_response_lengths` function's definition of "from first 1 to end" is slightly unusual (it includes trailing zeros). This is clearly documented and tested, but a note explaining WHY this definition is used (e.g., to detect truncation) would add pedagogical value.

---

## Overall Summary

| Exercise | Difficulty | Solvable? | Tests Passed | Quality |
|----------|-----------|-----------|-------------|---------|
| ex01_chat_template | Easy | Yes | 12/12 | Excellent |
| ex02_sequence_packing | Medium | Yes | 11/11 | Excellent |
| ex03_seqlen_balancing | Hard | Yes | 10/10 | Very Good |
| ex04_loss_mask_generation | Medium | Yes | 12/12 | Excellent |

**Overall assessment:** All four exercises are well-crafted. The problem descriptions are clear and complete enough to implement correct solutions without seeing the reference solution. The difficulty ratings are accurate. Test suites provide good coverage with appropriate assertion strategies (behavioral checks rather than overly rigid exact-match checks, especially for the packing and partitioning exercises). The exercises form a coherent progression through real LLM training data processing concerns.

**Top suggestions across all exercises:**
1. Add edge case tests for empty inputs and boundary conditions (especially ex03).
2. Ex03 would benefit from a small worked example in the docstring.
3. Consider adding `__init__.py` files to the exercise directories so tests can be run out of the box with relative imports.
