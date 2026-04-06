"""Exercise 02: BPE Tokenizer from Scratch (Medium, stdlib)

Byte Pair Encoding (BPE) is the tokenization algorithm used by GPT-2, GPT-3/4,
and many other LLMs. The core idea is:

Training phase:
    1. Start with a character-level vocabulary (each character is a token)
    2. Find the most frequent adjacent pair of tokens in the corpus
    3. Merge that pair into a new token
    4. Repeat for num_merges iterations

Encoding phase:
    1. Split text into characters
    2. Apply learned merges in order: for each merge (a, b), replace all
       adjacent occurrences of a followed by b with the merged token ab
    3. Map the resulting tokens to their integer IDs

Decoding phase:
    1. Map integer IDs back to token strings
    2. Concatenate them

Implement:
    - train_bpe(corpus, num_merges) -> list of merge rules
    - bpe_encode(text, merges, vocab) -> list of token IDs
    - bpe_decode(token_ids, id_to_token) -> reconstructed string
"""

from collections import Counter


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Train BPE merges from a corpus.

    Args:
        corpus: list of strings (training texts)
        num_merges: number of merge operations to learn

    Returns:
        Ordered list of merge rules as (token_a, token_b) tuples.
        Each merge means: wherever token_a is followed by token_b,
        replace them with token_a+token_b.
    """
    # TODO:
    # 1. Tokenize each word in corpus into characters (use a word boundary marker
    #    like appending '</w>' to the last character of each word)
    # 2. For each merge iteration:
    #    a. Count all adjacent pairs across the corpus
    #    b. Find the most frequent pair
    #    c. Merge that pair everywhere in the corpus
    #    d. Record the merge rule
    raise NotImplementedError("Implement train_bpe")


def bpe_encode(text: str, merges: list[tuple[str, str]], vocab: dict[str, int]) -> list[int]:
    """Encode text using learned BPE merges.

    Args:
        text: input string to encode
        merges: ordered list of merge rules from train_bpe
        vocab: mapping from token string to integer ID

    Returns:
        List of integer token IDs.
    """
    # TODO:
    # 1. Split text into words, then each word into characters
    #    (with '</w>' appended to last char of each word)
    # 2. For each merge rule in order, merge all adjacent pairs that match
    # 3. Map resulting tokens to IDs using vocab
    raise NotImplementedError("Implement bpe_encode")


def bpe_decode(token_ids: list[int], id_to_token: dict[int, str]) -> str:
    """Decode token IDs back to text.

    Args:
        token_ids: list of integer token IDs
        id_to_token: mapping from integer ID to token string

    Returns:
        Reconstructed string. '</w>' markers become spaces (or end of string).
    """
    # TODO:
    # 1. Map each ID to its token string
    # 2. Concatenate tokens
    # 3. Replace '</w>' with spaces, strip trailing space
    raise NotImplementedError("Implement bpe_decode")
