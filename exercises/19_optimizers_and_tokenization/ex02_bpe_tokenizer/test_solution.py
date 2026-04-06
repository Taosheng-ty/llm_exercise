"""Tests for Exercise 02: BPE Tokenizer from Scratch"""

import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_bpe = _mod.train_bpe
bpe_encode = _mod.bpe_encode
bpe_decode = _mod.bpe_decode


class TestTrainBpe:
    def test_returns_list_of_tuples(self):
        """train_bpe should return a list of (str, str) tuples."""
        corpus = ["ab ab ab"]
        merges = train_bpe(corpus, num_merges=1)
        assert isinstance(merges, list)
        assert all(isinstance(m, tuple) and len(m) == 2 for m in merges)

    def test_most_frequent_pair_first(self):
        """The first merge should be the most frequent adjacent pair."""
        corpus = ["aa aa aa bb"]
        merges = train_bpe(corpus, num_merges=1)
        # 'a' and 'a</w>' appear 3 times (end of each 'aa' word),
        # but 'a' 'a' also appears 3 times as the first two chars of 'aa'
        # The most frequent pair should be ('a', 'a</w>') since each 'aa' has it
        first_merge = merges[0]
        assert isinstance(first_merge[0], str) and isinstance(first_merge[1], str)

    def test_num_merges_respected(self):
        """Should return exactly num_merges merge rules (or fewer if exhausted)."""
        corpus = ["abcd abcd abcd"]
        merges = train_bpe(corpus, num_merges=3)
        assert len(merges) == 3

    def test_zero_merges(self):
        """Zero merges should return empty list."""
        corpus = ["hello world"]
        merges = train_bpe(corpus, num_merges=0)
        assert merges == []

    def test_merge_reduces_tokens(self):
        """After merging, repeated patterns should consolidate."""
        corpus = ["lo lo lo lo"]
        merges = train_bpe(corpus, num_merges=1)
        # 'l' and 'o</w>' should be the first merge (4 occurrences)
        assert merges[0] == ("l", "o</w>")


class TestBpeEncode:
    def test_encode_simple(self):
        """Encoding should produce integer token IDs."""
        corpus = ["ab ab ab"]
        merges = train_bpe(corpus, num_merges=1)
        # Build vocab from characters + merges
        vocab = {"a": 0, "b</w>": 1, "ab</w>": 2}
        ids = bpe_encode("ab", merges, vocab)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_uses_merges(self):
        """Encoding should apply merges to reduce token count."""
        merges = [("l", "o</w>")]
        vocab = {"h": 0, "e": 1, "l": 2, "lo</w>": 3}
        ids = bpe_encode("helo", merges, vocab)
        # 'helo' -> ['h', 'e', 'l', 'o</w>'] -> ['h', 'e', 'lo</w>'] after merge
        assert ids == [0, 1, 3]

    def test_encode_multiple_words(self):
        """Encoding should handle multiple words."""
        merges = [("l", "o</w>")]
        vocab = {"h": 0, "e": 1, "lo</w>": 2}
        ids = bpe_encode("helo helo", merges, vocab)
        assert ids == [0, 1, 2, 0, 1, 2]


class TestBpeDecode:
    def test_decode_simple(self):
        """Decoding should reconstruct the original text."""
        id_to_token = {0: "h", 1: "e", 2: "lo</w>"}
        text = bpe_decode([0, 1, 2], id_to_token)
        assert text == "helo"

    def test_decode_multiple_words(self):
        """Decoding should reconstruct multiple words with spaces."""
        id_to_token = {0: "h", 1: "i</w>", 2: "b", 3: "ye</w>"}
        text = bpe_decode([0, 1, 2, 3], id_to_token)
        assert text == "hi bye"

    def test_roundtrip(self):
        """Encoding then decoding should reconstruct the original text."""
        corpus = ["low low low lowest newest"]
        merges = train_bpe(corpus, num_merges=5)

        # Build vocabulary from the corpus after all merges
        # Start with character tokens
        all_tokens = set()
        for text in corpus:
            for word in text.split():
                chars = list(word[:-1]) + [word[-1] + "</w>"]
                all_tokens.update(chars)
        # Add merge results
        for a, b in merges:
            all_tokens.add(a + b)

        vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}
        id_to_token = {i: tok for tok, i in vocab.items()}

        original = "low lowest"
        ids = bpe_encode(original, merges, vocab)
        decoded = bpe_decode(ids, id_to_token)
        assert decoded == original
