"""Tests for Exercise 04: Special Token Handler"""

import importlib.util
import os

import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
encode_with_specials = _mod.encode_with_specials
pad_batch = _mod.pad_batch
create_attention_mask = _mod.create_attention_mask


class TestEncodeWithSpecials:
    def test_both_bos_and_eos(self):
        """Should prepend BOS and append EOS."""
        result = encode_with_specials([10, 20, 30], bos_id=1, eos_id=2)
        assert result == [1, 10, 20, 30, 2]

    def test_bos_only(self):
        """Should prepend BOS but not append EOS."""
        result = encode_with_specials([10, 20], bos_id=1, eos_id=2, add_eos=False)
        assert result == [1, 10, 20]

    def test_eos_only(self):
        """Should append EOS but not prepend BOS."""
        result = encode_with_specials([10, 20], bos_id=1, eos_id=2, add_bos=False)
        assert result == [10, 20, 2]

    def test_neither(self):
        """With both flags False, should return original sequence."""
        result = encode_with_specials([10, 20], bos_id=1, eos_id=2,
                                      add_bos=False, add_eos=False)
        assert result == [10, 20]

    def test_empty_sequence(self):
        """Should work on empty input."""
        result = encode_with_specials([], bos_id=1, eos_id=2)
        assert result == [1, 2]

    def test_does_not_mutate_input(self):
        """Should not modify the input list."""
        original = [10, 20, 30]
        copy = list(original)
        encode_with_specials(original, bos_id=1, eos_id=2)
        assert original == copy


class TestPadBatch:
    def test_right_padding(self):
        """Right padding should add pad tokens after content."""
        seqs = [[1, 2, 3], [4, 5]]
        result = pad_batch(seqs, pad_id=0, padding_side="right")
        expected = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_left_padding(self):
        """Left padding should add pad tokens before content."""
        seqs = [[1, 2, 3], [4, 5]]
        result = pad_batch(seqs, pad_id=0, padding_side="left")
        expected = np.array([[1, 2, 3], [0, 4, 5]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_explicit_max_len(self):
        """Should pad to specified max_len."""
        seqs = [[1, 2], [3]]
        result = pad_batch(seqs, pad_id=0, padding_side="right", max_len=5)
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1, 2, 0, 0, 0])

    def test_truncation(self):
        """Sequences longer than max_len should be truncated."""
        seqs = [[1, 2, 3, 4, 5]]
        result = pad_batch(seqs, pad_id=0, padding_side="right", max_len=3)
        np.testing.assert_array_equal(result[0], [1, 2, 3])

    def test_output_dtype(self):
        """Output should be int64."""
        seqs = [[1, 2]]
        result = pad_batch(seqs, pad_id=0)
        assert result.dtype == np.int64

    def test_single_sequence(self):
        """Should handle a single sequence with no padding needed."""
        seqs = [[10, 20, 30]]
        result = pad_batch(seqs, pad_id=0)
        np.testing.assert_array_equal(result, [[10, 20, 30]])


class TestCreateAttentionMask:
    def test_right_padded(self):
        """Mask should be 1 for content, 0 for right padding."""
        padded = np.array([[5, 6, 7, 0, 0], [5, 6, 0, 0, 0]], dtype=np.int64)
        mask = create_attention_mask(padded, pad_id=0)
        expected = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)

    def test_left_padded(self):
        """Mask should be 0 for left padding, 1 for content."""
        padded = np.array([[0, 0, 5, 6, 7], [0, 0, 0, 5, 6]], dtype=np.int64)
        mask = create_attention_mask(padded, pad_id=0)
        expected = np.array([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)

    def test_no_padding(self):
        """Mask should be all 1s when there is no padding."""
        padded = np.array([[5, 6, 7]], dtype=np.int64)
        mask = create_attention_mask(padded, pad_id=0)
        np.testing.assert_array_equal(mask, [[1, 1, 1]])

    def test_mask_dtype(self):
        """Mask should be int64."""
        padded = np.array([[1, 0]], dtype=np.int64)
        mask = create_attention_mask(padded, pad_id=0)
        assert mask.dtype == np.int64
