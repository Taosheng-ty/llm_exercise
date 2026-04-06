# Validation Report: 20_inference_and_serving

## Exercise 01: Paged Attention KV Cache (ex01_paged_attention)

**Verdict: FAIL -- hidden attribute name requirements**

Issues:
- The test (`test_cache_storage_shape`, line 26-27) checks `cache.key_cache.shape` and `cache.value_cache.shape`. The problem.py docstring does not specify that the attributes must be named `key_cache` and `value_cache`. A student could reasonably name them `k_cache`/`v_cache`, `keys`/`values`, etc., and fail.
- The internal tracking data structures (free set, block fill counts) are not prescribed, which is fine since tests only observe them indirectly.
- Otherwise, the behavioral specification (allocate, free, append, read, num_free_blocks) is clear and sufficient.

**Fix needed:** Add to `__init__` docstring: "Store the physical KV storage as `self.key_cache` and `self.value_cache`."

---

## Exercise 02: INT8 Quantization (ex02_int8_quantization)

**Verdict: PASS**

The problem.py provides clear formulas for all five functions. The docstring specifies:
- Symmetric: per-channel along dim 0, scale = max(|row|) / 127, clamp to [-127,127], int8 output.
- Asymmetric: per-channel along dim 0, scale/zero_point formulas, clamp to [0,255], uint8 output.
- Error metrics: MSE, max abs error, SNR dB with explicit formula.

Minor notes:
- The zero-tensor edge case (division by zero in scale) is not explicitly mentioned, but clamping the scale denominator is standard practice and reasonably inferred.
- The `scale.numel() == 4` test is consistent with the docstring shape `(num_channels, 1, ...)` for a 2D input.

A competent student can pass all tests from the problem.py alone.

---

## Exercise 03: ALiBi Attention (ex03_alibi_attention)

**Verdict: PASS**

The problem.py is well-specified:
- Slopes formula: `slopes[i] = 2^(-8*(i+1)/num_heads)` -- explicit and unambiguous.
- Bias formula: `bias[h, q, k] = slopes[h] * (k - q)` -- explicit.
- Attention: standard scaled dot-product + ALiBi bias + optional causal mask with -inf.

All three functions have clear input/output shapes and formulas. Tests align perfectly with the docstring. No hidden requirements.

---

## Exercise 04: Cross-Attention (ex04_cross_attention)

**Verdict: PASS (borderline)**

The problem.py names the projections explicitly as `W_q`, `W_k`, `W_v`, `W_o`, and the tests check for these attribute names (line 27-30). This is consistent.

Minor notes:
- The test `test_parameter_count` (line 37-38) expects `4 * (d*d + d)`, implying each Linear layer has bias=True. The problem.py does not explicitly state whether to use bias. However, `nn.Linear` defaults to `bias=True`, so a student using default parameters would pass.
- The encoder_mask convention (True=valid, False=pad) is explicitly stated.

A careful student following the problem.py and using standard PyTorch defaults will pass.

---

## Exercise 05: Multi-Token Prediction (ex05_multi_token_prediction)

**Verdict: FAIL -- hidden attribute name requirement + ambiguous total_loss**

Issues:
1. **Hidden attribute name:** The test `test_gradients_flow_to_all_heads` (line 58) accesses `head.heads` to iterate over individual prediction heads. The problem.py does not specify that the nn.ModuleList must be named `heads`. A student could name it `prediction_heads`, `linears`, `layers`, etc., and fail.

2. **Ambiguous total_loss:** The problem.py says `total_loss` is "scalar mean" but does not clarify mean of what. The reference solution computes `sum(per_head_losses) / count_of_nonzero_losses`. A student might reasonably compute `sum(per_head_losses) / len(per_head_losses)` (simple mean) or `sum(per_head_losses) / num_futures`. Since the test `test_correct_target_shifting` checks `total.item() < 0.01`, both approaches would likely pass that test, but semantic correctness is unclear.

3. **Target shifting clarity:** The docstring says "head i (0-indexed), the target at position t is targets[t+i+1]". This is correct and clear, but students may be confused about whether the truncation applies to the predictions or the targets or both.

**Fix needed:** Add to docstring: "Store the linear layers in `self.heads` as an nn.ModuleList." Clarify the total_loss formula.

---

## Summary

| Exercise | Verdict | Key Issue |
|----------|---------|-----------|
| ex01_paged_attention | FAIL | Tests require `key_cache`/`value_cache` attribute names not specified in problem.py |
| ex02_int8_quantization | PASS | Well-specified formulas |
| ex03_alibi_attention | PASS | Well-specified formulas and shapes |
| ex04_cross_attention | PASS | Attribute names are specified; default bias works |
| ex05_multi_token_prediction | FAIL | Tests require `heads` attribute name not specified in problem.py; ambiguous total_loss |

**Solvable from problem.py alone: 3 out of 5** (ex02, ex03, ex04)

**Fixable with minor docstring additions: 5 out of 5** -- both failing exercises only need attribute name hints added to the problem.py docstrings.
