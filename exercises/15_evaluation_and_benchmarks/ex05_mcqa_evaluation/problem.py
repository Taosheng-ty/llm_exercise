"""
Exercise 5: Multiple-Choice QA Evaluation

Evaluate multiple-choice question answering by extracting answer letters from
free-form model responses and comparing to ground truth.

This mirrors the pattern in slime/rollout/rm_hub/gpqa.py compute_gpqa_reward(),
which uses regex patterns to extract answer letters and falls back to text
similarity when extraction fails.

Your task: implement extract_answer_letter(), text_similarity_fallback(), and
evaluate_mcqa_dataset().

Difficulty: Medium
Framework: numpy / stdlib
"""

import re
import string


def extract_answer_letter(
    response: str,
    valid_letters: list[str] | None = None,
) -> str | None:
    """Extract the answer letter from a free-form model response.

    Try these regex patterns in order on the text AFTER any </think> tag:
    1. r"(?:answer|option|choice)\\s*(?:is|:)?\\s*([A-Z])" (case insensitive)
    2. r"([A-Z])\\s*(?:is\\s*(?:the)?\\s*correct)" (case insensitive)
    3. r"final\\s*(?:answer|option)\\s*(?:is|:)?\\s*([A-Z])" (case insensitive)
    4. Fallback: last standalone uppercase letter (\\b[A-Z]\\b) in the text

    For each match, check if the letter is in valid_letters (uppercased).
    If valid_letters is None, default to A-J.

    Also strip chain-of-thought: if "</think>" is in the response, only
    consider text after the last "</think>".

    Args:
        response: The model's full response text.
        valid_letters: List of valid answer letters (e.g., ["A","B","C","D"]).

    Returns:
        The extracted letter (uppercase), or None if extraction fails.

    Examples:
        >>> extract_answer_letter("The answer is B.", ["A","B","C","D"])
        'B'
        >>> extract_answer_letter("I think... </think> C", ["A","B","C","D"])
        'C'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement extract_answer_letter")


def text_similarity_fallback(
    response: str,
    choices: list[str],
) -> int:
    """When letter extraction fails, find the most similar choice by word overlap.

    Compute Jaccard similarity (intersection over union of word sets) between
    the normalized response and each choice. Return the index of the most
    similar choice. Normalization: lowercase, split on non-alphanumeric chars.

    Args:
        response: The model's response text (after stripping chain-of-thought).
        choices: List of choice texts (e.g., ["photosynthesis", "mitosis", ...]).

    Returns:
        Index (0-based) of the most similar choice.
        Ties broken by lowest index.
        If response is empty or all similarities are 0, return 0.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement text_similarity_fallback")


def evaluate_mcqa_dataset(
    responses: list[str],
    correct_letters: list[str],
    choices_list: list[list[str]] | None = None,
    valid_letters: list[str] | None = None,
) -> dict:
    """Evaluate a multiple-choice QA dataset.

    For each (response, correct_letter) pair:
    1. Try extract_answer_letter(). If successful, compare to correct_letter.
    2. If extraction fails AND choices_list is provided, use
       text_similarity_fallback() to get a choice index, convert to letter,
       and compare.
    3. If extraction fails and no choices, count as incorrect.

    Args:
        responses: List of N model response strings.
        correct_letters: List of N correct answer letters (uppercase).
        choices_list: Optional list of N lists of choice texts. Used for
                      fallback similarity matching.
        valid_letters: Valid answer letters. Defaults to ["A","B","C","D"].

    Returns:
        Dict with:
        - "accuracy": float, fraction correct
        - "extracted_count": int, number where letter extraction succeeded
        - "fallback_count": int, number where fallback was used
        - "failed_count": int, number where no answer could be determined
        - "total": int, total number of questions
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement evaluate_mcqa_dataset")
