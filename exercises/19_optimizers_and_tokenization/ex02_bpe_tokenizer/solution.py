"""Solution for Exercise 02: BPE Tokenizer from Scratch"""

from collections import Counter


def _tokenize_word(word: str) -> list[str]:
    """Split a word into characters with end-of-word marker on last char."""
    if not word:
        return []
    chars = list(word[:-1]) + [word[-1] + "</w>"]
    return chars


def _get_pair_counts(tokenized_corpus: list[list[list[str]]]) -> Counter:
    """Count all adjacent token pairs across the corpus."""
    pairs = Counter()
    for sentence in tokenized_corpus:
        for word_tokens in sentence:
            for i in range(len(word_tokens) - 1):
                pairs[(word_tokens[i], word_tokens[i + 1])] += 1
    return pairs


def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    """Merge all occurrences of pair in a token list."""
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Train BPE merges from a corpus.

    Tokenizes each word into characters (with '</w>' end-of-word marker),
    then iteratively finds and merges the most frequent adjacent pair.
    """
    # Tokenize corpus: list of sentences, each sentence is list of words,
    # each word is list of character tokens
    tokenized = []
    for text in corpus:
        words = text.split()
        tokenized.append([_tokenize_word(w) for w in words])

    merges = []
    for _ in range(num_merges):
        pairs = _get_pair_counts(tokenized)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Apply merge across entire corpus
        for sent_idx, sentence in enumerate(tokenized):
            for word_idx, word_tokens in enumerate(sentence):
                tokenized[sent_idx][word_idx] = _merge_pair(word_tokens, best_pair)

    return merges


def bpe_encode(text: str, merges: list[tuple[str, str]], vocab: dict[str, int]) -> list[int]:
    """Encode text using learned BPE merges."""
    words = text.split()
    all_ids = []
    for word in words:
        tokens = _tokenize_word(word)
        # Apply merges in order
        for pair in merges:
            tokens = _merge_pair(tokens, pair)
        # Map tokens to IDs
        for tok in tokens:
            if tok in vocab:
                all_ids.append(vocab[tok])
    return all_ids


def bpe_decode(token_ids: list[int], id_to_token: dict[int, str]) -> str:
    """Decode token IDs back to text."""
    tokens = [id_to_token[tid] for tid in token_ids]
    text = "".join(tokens)
    # Replace end-of-word markers with spaces
    text = text.replace("</w>", " ")
    return text.rstrip(" ")
