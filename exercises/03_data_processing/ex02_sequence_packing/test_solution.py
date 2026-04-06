"""Tests for Exercise 02: Sequence Packing"""

import numpy as np
import pytest
from .solution import pack_sequences, compute_packing_efficiency


class TestPackSequences:
    def test_single_sequence(self):
        seqs = [[1, 2, 3]]
        ids, mask = pack_sequences(seqs, max_seq_len=5)
        assert ids.shape == (1, 5)
        assert mask.shape == (1, 5)
        np.testing.assert_array_equal(ids[0, :3], [1, 2, 3])
        np.testing.assert_array_equal(ids[0, 3:], [0, 0])
        np.testing.assert_array_equal(mask[0], [1, 1, 1, 0, 0])

    def test_two_sequences_fit_in_one_bin(self):
        seqs = [[1, 2, 3], [4, 5]]
        ids, mask = pack_sequences(seqs, max_seq_len=6)
        # Both should fit in one bin (total length = 5 <= 6)
        assert ids.shape[0] == 1
        # Real tokens should sum to 5
        assert mask.sum() == 5

    def test_sequences_need_multiple_bins(self):
        seqs = [[1, 2, 3], [4, 5, 6], [7, 8]]
        ids, mask = pack_sequences(seqs, max_seq_len=4)
        # [1,2,3] takes 3/4 of a bin, [4,5,6] takes 3/4, [7,8] can fit with either
        # FFD sorts: [1,2,3], [4,5,6], [7,8] (all len 3,3,2)
        # Bin 0: [1,2,3] -> remains 1, can't fit [4,5,6]
        # Bin 1: [4,5,6] -> remains 1, can't fit [7,8]
        # Actually [7,8] len 2 doesn't fit in remainder 1
        # So we need 3 bins? No: FFD sorts descending: len 3, len 3, len 2
        # Bin 0: [1,2,3] (rem=1), Bin 1: [4,5,6] (rem=1), Bin 2: [7,8] (rem=2)
        assert ids.shape[0] == 3
        assert mask.sum() == 8

    def test_packing_merges_small_sequences(self):
        seqs = [[1], [2], [3], [4]]
        ids, mask = pack_sequences(seqs, max_seq_len=4)
        # All 4 sequences (length 1 each) should fit in 1 bin
        assert ids.shape[0] == 1
        np.testing.assert_array_equal(mask[0], [1, 1, 1, 1])

    def test_sequence_too_long_raises(self):
        seqs = [[1, 2, 3, 4, 5]]
        with pytest.raises(ValueError, match="length"):
            pack_sequences(seqs, max_seq_len=3)

    def test_empty_input(self):
        ids, mask = pack_sequences([], max_seq_len=5)
        assert ids.shape == (0, 5)
        assert mask.shape == (0, 5)

    def test_custom_pad_token(self):
        seqs = [[1, 2]]
        ids, mask = pack_sequences(seqs, max_seq_len=4, pad_token_id=99)
        np.testing.assert_array_equal(ids[0], [1, 2, 99, 99])


class TestPackingEfficiency:
    def test_perfect_packing(self):
        seqs = [[1, 2, 3, 4]]
        eff = compute_packing_efficiency(seqs, max_seq_len=4)
        assert eff == pytest.approx(1.0)

    def test_half_efficiency(self):
        seqs = [[1, 2]]
        eff = compute_packing_efficiency(seqs, max_seq_len=4)
        assert eff == pytest.approx(0.5)

    def test_multiple_sequences(self):
        # 3 tokens + 2 tokens = 5 real tokens in 1 bin of size 6
        seqs = [[1, 2, 3], [4, 5]]
        eff = compute_packing_efficiency(seqs, max_seq_len=6)
        assert eff == pytest.approx(5.0 / 6.0)

    def test_empty_sequences(self):
        eff = compute_packing_efficiency([], max_seq_len=4)
        assert eff == pytest.approx(0.0)
