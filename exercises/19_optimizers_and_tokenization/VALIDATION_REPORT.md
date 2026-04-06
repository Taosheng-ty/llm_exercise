# Validation Report: 19_optimizers_and_tokenization

## Exercise 01: AdamW Optimizer (ex01_adamw_optimizer)

**Verdict: FAIL (minor ambiguity)**

The algorithm description in problem.py is excellent -- the AdamW update rule is given explicitly and correctly. However, the tests depend on specific internal attribute names that are NOT documented in problem.py:

- Tests access `opt.params` (must be a list). Problem.py says "store params as a list" in TODO but does not name the attribute.
- Tests access `opt.state[p]["m"]`, `opt.state[p]["v"]`, `opt.state[p]["step"]` -- a dict keyed by parameter objects with specific string sub-keys. Problem.py says "initialize state (step count, first moment m, second moment v)" but never specifies this exact structure.
- Tests check `model.weight.grad is None` after `zero_grad()`, meaning the implementation must set `.grad = None` rather than zeroing it. Problem.py says "set .grad to None or zero" -- this is fine since setting to None would pass.

A student could reasonably implement the correct algorithm but use different attribute names (e.g., `self.parameters`, `self.m`, `self.v`, `self.t`) and fail the state-inspection tests. The algorithmic behavior tests (matching torch.optim.AdamW) would still pass, but 3 out of 8 tests inspect internal state.

**Suggested fix:** Add to docstrings that `self.params` should be a list attribute and `self.state` should be a `dict` mapping each parameter to `{"m": ..., "v": ..., "step": ...}`.

---

## Exercise 02: BPE Tokenizer (ex02_bpe_tokenizer)

**Verdict: PASS**

Problem.py clearly describes:
- The `</w>` word-end marker convention (stated in TODO comments for both train and encode)
- The training algorithm (character-level start, count pairs, merge most frequent)
- The encoding algorithm (split into chars with `</w>`, apply merges in order, map to IDs)
- The decoding algorithm (map IDs to tokens, concatenate, replace `</w>` with spaces, strip)

All tests are consistent with the problem description. The roundtrip test builds its own vocab, so there is no hidden vocab-building convention. The word splitting convention (split on spaces) is implicit but standard.

No issues found. A student with knowledge of BPE could implement this from problem.py alone.

---

## Exercise 03: ZeRO Optimizer Sharding (ex03_zero_optimizer_sharding)

**Verdict: PASS**

Problem.py is very clear. All five functions have well-specified signatures and docstrings:
- The "last rank gets remainder" convention for uneven splits is explicitly stated in `shard_optimizer_state`
- The same convention is implicitly expected for `reduce_scatter_gradients` and `shard_parameters` (tests confirm via roundtrip)
- `gather_for_step` and `all_gather_params` are simple concatenations

The use of `np.array_split` (which gives the last shard the remainder) is the natural implementation. Tests confirm this behavior (e.g., 10 elements / 3 ranks = [3, 3, 4]).

No issues found. Fully solvable from problem.py.

---

## Exercise 04: Special Token Handler (ex04_special_token_handler)

**Verdict: PASS**

Problem.py is clear and complete:
- `encode_with_specials`: trivial prepend/append, well-documented
- `pad_batch`: padding side, max_len defaulting, and truncation are all described (truncation in TODO step 2)
- `create_attention_mask`: simple comparison, well-documented
- Return types (numpy arrays, dtypes) are specified

The test for non-mutation of input (`test_does_not_mutate_input`) is a standard Python expectation. All behavior is inferable from the docstrings.

No issues found. Fully solvable from problem.py.

---

## Exercise 05: Constrained Decoding (ex05_constrained_decoding)

**Verdict: PASS**

Problem.py provides an excellent worked example of the FSM structure, which is the most complex part. Key details are well-specified:
- The trie structure with state IDs and `"is_terminal"` marker is shown by example
- `get_valid_token_mask` explicitly states terminal/unknown states return all-False mask
- `constrained_sample` steps are enumerated (check no valid tokens, mask with -inf, temperature scale, softmax, sample)
- The return type of -1 for no valid tokens is documented

One minor note: `get_valid_token_mask` must filter out the `"is_terminal"` key when iterating state transitions, but this is clear from the example (terminal states only contain `"is_terminal"`, not token transitions).

No issues found. Fully solvable from problem.py.

---

## Summary

| Exercise | Verdict | Issue |
|----------|---------|-------|
| 01 - AdamW Optimizer | FAIL | Internal state attribute names (`params`, `state[p]["m"]`, etc.) not documented |
| 02 - BPE Tokenizer | PASS | None |
| 03 - ZeRO Optimizer Sharding | PASS | None |
| 04 - Special Token Handler | PASS | None |
| 05 - Constrained Decoding | PASS | None |

**4 out of 5 exercises are fully solvable from problem.py alone.**

Exercise 01 is solvable algorithmically but 3/8 tests inspect internal state using attribute names not specified in the problem. A student who implements the correct algorithm but uses different attribute names would fail those tests.
