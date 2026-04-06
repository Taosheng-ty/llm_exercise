"""
Solution for Exercise 5: Multiple-Choice QA Evaluation
"""

import re
import string


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _normalize_for_similarity(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def extract_answer_letter(
    response: str,
    valid_letters: list[str] | None = None,
) -> str | None:
    """Extract the answer letter from a free-form model response."""
    if not response:
        return None

    if valid_letters is None:
        valid_letters = list(string.ascii_uppercase[:10])

    valid_set = {letter.upper() for letter in valid_letters}
    text = _strip_chain_of_thought(response)

    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_set:
                return letter

    # Fallback: last standalone capital letter
    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid_set:
            return letter

    return None


def text_similarity_fallback(
    response: str,
    choices: list[str],
) -> int:
    """When letter extraction fails, find the most similar choice by word overlap."""
    response_words = _normalize_for_similarity(response)
    if not response_words:
        return 0

    best_idx = 0
    best_sim = -1.0

    for i, choice in enumerate(choices):
        choice_words = _normalize_for_similarity(choice)
        if not choice_words:
            continue
        intersection = len(response_words & choice_words)
        union = len(response_words | choice_words)
        sim = intersection / union if union > 0 else 0.0
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx


def evaluate_mcqa_dataset(
    responses: list[str],
    correct_letters: list[str],
    choices_list: list[list[str]] | None = None,
    valid_letters: list[str] | None = None,
) -> dict:
    """Evaluate a multiple-choice QA dataset."""
    if valid_letters is None:
        valid_letters = ["A", "B", "C", "D"]

    correct = 0
    extracted_count = 0
    fallback_count = 0
    failed_count = 0
    total = len(responses)

    for i in range(total):
        response = responses[i]
        correct_letter = correct_letters[i].upper()

        letter = extract_answer_letter(response, valid_letters)
        if letter is not None:
            extracted_count += 1
            if letter == correct_letter:
                correct += 1
        elif choices_list is not None:
            fallback_count += 1
            idx = text_similarity_fallback(
                _strip_chain_of_thought(response), choices_list[i]
            )
            predicted_letter = valid_letters[idx] if idx < len(valid_letters) else None
            if predicted_letter and predicted_letter.upper() == correct_letter:
                correct += 1
        else:
            failed_count += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "extracted_count": extracted_count,
        "fallback_count": fallback_count,
        "failed_count": failed_count,
        "total": total,
    }
