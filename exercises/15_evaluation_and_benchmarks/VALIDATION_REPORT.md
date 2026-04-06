# Validation Report: 15_evaluation_and_benchmarks

## Per-Exercise Results

### ex01_perplexity
- **Tests**: 9/9 passed
- **Solvable from description alone**: Yes
- **Issues found**: None. The formula is clearly specified with examples.
- **Difficulty rating**: Medium -- accurate. Requires understanding of masked tensor operations but the formula is given.

### ex02_calibration_metrics
- **Tests**: 9/9 passed
- **Solvable from description alone**: Yes
- **Issues found**: The binning boundary logic (inclusive/exclusive on bin edges) is not fully specified in the problem description. The student must infer a reasonable approach. However, the tests are lenient enough that standard approaches work. One minor note: `test_none_metadata_overrides_becomes_dict` is in the wrong exercise's test file context -- actually it is in ex06, so no issue here.
- **Difficulty rating**: Medium -- accurate. ECE binning requires care with edge cases.

### ex03_majority_voting
- **Tests**: 20/20 passed
- **Solvable from description alone**: Yes
- **Issues found**: None. Tie-breaking rules and edge cases are well-documented. The examples in docstrings are helpful.
- **Difficulty rating**: Medium -- slightly generous. This is closer to Easy since it is mostly Counter usage with straightforward logic.

### ex04_exact_match
- **Tests**: 22/22 passed
- **Solvable from description alone**: Yes
- **Issues found**: None. The normalization steps are clearly ordered and well-specified. Examples confirm expected behavior.
- **Difficulty rating**: Easy -- accurate. Standard string normalization with clear step-by-step instructions.

### ex05_mcqa_evaluation
- **Tests**: 22/22 passed
- **Solvable from description alone**: Yes
- **Issues found**: Minor ambiguity -- the problem description shows regex patterns with double-escaped backslashes (e.g., `\\s*` instead of `\s*`), which could confuse students reading raw docstrings vs rendered text. The `test_no_match` test relies on the fact that lowercase letters do not match `\b[A-Z]\b`, which is correct but could trip up students who use case-insensitive fallback. The example in `weighted_majority_vote` docstring is misleading (says "A wins" then corrects to "B wins") -- while technically showing the correction, it could be cleaner. Overall the exercise is well-designed.
- **Difficulty rating**: Medium -- accurate. Regex pattern matching with multiple fallback strategies requires careful implementation.

### ex06_eval_config_builder
- **Tests**: 23/23 passed
- **Solvable from description alone**: Yes
- **Issues found**: The `resolve_field` function uses `None` as sentinel for "not set," but the `test_zero_is_valid` and `test_empty_string_is_valid` tests verify that 0 and "" are treated as valid values. This is well-documented but creates a subtle design issue: a user cannot intentionally set a field to `None` at the dataset level to fall through to defaults. This is a known limitation of the pattern, not a bug in the exercise.
- **Difficulty rating**: Medium -- accurate. The config resolution hierarchy and field aliasing add real complexity.

### ex07_benchmark_aggregation
- **Tests**: 21/21 passed
- **Solvable from description alone**: Yes
- **Issues found**: None. All functions are clearly specified with straightforward algorithms.
- **Difficulty rating**: Easy -- accurate. Standard statistical aggregations with clear specs.

## Summary

| Exercise | Pass/Fail | Solvable | Difficulty Claimed | Difficulty Actual |
|----------|-----------|----------|-------------------|-------------------|
| ex01_perplexity | 9/9 | Yes | Medium | Medium |
| ex02_calibration_metrics | 9/9 | Yes | Medium | Medium |
| ex03_majority_voting | 20/20 | Yes | Medium | Easy-Medium |
| ex04_exact_match | 22/22 | Yes | Easy | Easy |
| ex05_mcqa_evaluation | 22/22 | Yes | Medium | Medium |
| ex06_eval_config_builder | 23/23 | Yes | Medium | Medium |
| ex07_benchmark_aggregation | 21/21 | Yes | Easy | Easy |

**Total**: 126/126 tests passed across all 7 exercises.

## Overall Quality Assessment

**Rating: Excellent**

All 7 exercises are well-designed and fully solvable from their problem descriptions alone without needing to peek at solutions. The docstrings contain clear specifications including formulas, step-by-step instructions, type signatures, edge case behavior, and examples.

### Strengths
- Comprehensive test coverage with good edge cases (empty inputs, boundary conditions, tie-breaking)
- Clear progression from simpler (ex04, ex07) to more complex (ex05, ex06) exercises
- Practical relevance to LLM evaluation workflows
- Consistent code style and structure across exercises

### Suggestions
1. **ex03**: Consider downgrading difficulty label to Easy, as it primarily uses Counter with simple logic.
2. **ex05**: Clean up the misleading example in `weighted_majority_vote` docstring (the "actually B wins" correction is awkward -- it is in ex03 actually). Fix the double-escaped regex patterns in docstrings to avoid confusion.
3. **ex02**: Consider adding a note about bin boundary convention (left-exclusive, right-inclusive vs. other schemes) to reduce ambiguity.
4. **General**: The exercises could benefit from a brief note about which Python version features are expected (e.g., `str | None` union types require Python 3.10+).
