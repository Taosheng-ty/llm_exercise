# Validation Report: 02_reward_functions

Validated by: Student Agent (Claude)
Date: 2026-04-06

## Summary

| Exercise | Solvable from description? | Tests passed | Stated difficulty | Assessed difficulty |
|----------|---------------------------|-------------|-------------------|---------------------|
| ex01_math_answer_normalization | Yes | 19/19 | Medium | Medium |
| ex02_f1_score | Yes | 12/12 | Easy | Easy |
| ex03_reward_shaping | Yes | 17/17 | Easy | Easy |
| ex04_outcome_reward_model | Yes | 24/24 | Medium | Medium |

**Overall: All 72 tests passed. All exercises are solvable from problem descriptions alone.**

---

## Exercise 1: Math Answer Normalization

- **Solvable from description alone:** Yes
- **Tests passed:** 19/19
- **Difficulty rating:** Medium (matches stated difficulty)

### Problem description quality
The problem description is excellent. The 12-step normalization pipeline is clearly enumerated with specific ordering requirements. Each step is concrete and unambiguous. The examples in the docstring further clarify expected behavior.

### Test case quality
Good coverage of individual normalization steps plus equivalence comparisons. Tests are appropriately strict.

### Suggestions for improvement
- The `\text{...}` removal (step 3) only handles non-nested braces. Consider documenting that limitation or adding a test for `\text{hello \textbf{world}}` to clarify expected behavior.
- Step 7 (leading decimal) could have an edge case test like `".123"` to confirm it handles arbitrary decimal lengths.
- Consider adding a test for `answers_are_equivalent` where both are fractions with different representations (e.g., `"2/4"` vs `"1/2"`), which would currently fail since the float comparison path would handle it but it tests a more interesting case.
- The `\frac{a}{b}` regex replacement (step 9) is described as "simple replacement" but does not handle nested fracs like `\frac{\frac{1}{2}}{3}`. This is fine for the exercise scope but worth documenting explicitly.

---

## Exercise 2: Token-level F1 Score

- **Solvable from description alone:** Yes
- **Tests passed:** 12/12
- **Difficulty rating:** Easy (matches stated difficulty)

### Problem description quality
Clear and well-structured. The 5-step F1 computation pipeline is standard and well-documented. The special token handling and None case are explicitly called out.

### Test case quality
Good basic coverage. Tests verify the core algorithm, edge cases (None, special tokens), and article removal effects.

### Suggestions for improvement
- Could add a test for empty string prediction (not None, but `""`).
- Could add a test where both prediction and gold normalize to empty strings after article/punctuation removal.
- The special token list `("yes", "no", "noanswer")` is somewhat arbitrary without context -- a brief note about why these are special (HotpotQA convention) would be helpful.
- Consider testing with duplicate tokens to verify Counter-based intersection works correctly (e.g., `"cat cat"` vs `"cat"`).

---

## Exercise 3: Reward Shaping

- **Solvable from description alone:** Yes
- **Tests passed:** 17/17
- **Difficulty rating:** Easy (matches stated difficulty)

### Problem description quality
Excellent. The formula for length penalty is given explicitly. The format bonus markers are listed exhaustively with case-sensitivity and comma requirements clearly stated. Very straightforward to implement.

### Test case quality
Good coverage of each marker type, case insensitivity, custom parameters, and the combined `shape_reward` function.

### Suggestions for improvement
- This exercise is almost too straightforward -- it is essentially direct formula implementation with no algorithmic thinking required. Could be labeled "Warm-up" rather than "Easy" if the scale allows it.
- Could add a test verifying that `shape_reward` result is NOT clipped (as documented), e.g., a case where the result goes below 0.
- Could add a test for "Step 10" vs "Step 1" to verify partial matching behavior (does "Step 1" in "Step 10" still match? Yes it does, which is fine, but worth being explicit).
- The marker "First," requires a comma but "Therefore" does not. This asymmetry is documented but could trip students up -- consider adding a test for "First" without comma returning 0.0.

---

## Exercise 4: Outcome Reward Model

- **Solvable from description alone:** Yes
- **Tests passed:** 24/24
- **Difficulty rating:** Medium (matches stated difficulty)

### Problem description quality
Well-structured with clear extraction strategy ordering (boxed -> answer phrase -> last line). The nested brace handling requirement for `\boxed{}` and "last occurrence" semantics are both explicitly stated. The `normalize_for_comparison` function is intentionally simple and clearly described.

### Test case quality
Good coverage including priority ordering, nested braces, and edge cases. The `test_boxed_takes_priority` test is particularly good at verifying strategy ordering.

### Suggestions for improvement
- Could add a test for `extract_from_answer_phrase` where the answer contains a period mid-value (e.g., "The answer is 3.14." -- should this return "3" or "3.14"?). The current regex stops at the first period, which would return "3" for "3.14." -- this might be unintentional.
- Could add a test for `normalize_for_comparison` with multiple trailing periods (e.g., "42..").
- Could add an edge case test for `extract_from_boxed` with unmatched braces.
- The `normalize_for_comparison` description says "remove trailing periods" but only one period is shown in tests. Clarify whether `rstrip(".")` (removing all trailing periods) or removing exactly one is intended.

---

## Overall Assessment

All four exercises are well-designed, clearly documented, and solvable from problem descriptions alone without consulting the reference solution. The difficulty ratings are accurate. The test suites provide adequate coverage for validating correct implementations.

The strongest exercise is ex01 (most interesting algorithmic content) and ex04 (real-world pattern with strategy ordering). The weakest is ex03 (almost trivially formulaic, though still pedagogically useful as a warm-up).

One cross-cutting concern: none of the exercises test for particularly adversarial inputs (empty strings, very long inputs, unicode, etc.). Adding 1-2 edge case tests per exercise would improve robustness validation without significantly increasing difficulty.
