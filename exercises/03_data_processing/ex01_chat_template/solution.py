"""
Exercise 01: Chat Template Builder - Solution
"""


def apply_chatml_template(
    messages: list[dict],
    system_message: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Apply ChatML template to a list of messages."""
    all_messages = list(messages)

    # Prepend system message if provided and no existing system message at start
    if system_message is not None:
        if not all_messages or all_messages[0]["role"] != "system":
            all_messages.insert(0, {"role": "system", "content": system_message})

    result = ""
    for msg in all_messages:
        role = msg["role"]
        content = msg["content"]
        result += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        result += "<|im_start|>assistant\n"

    return result


def build_messages(
    data: dict,
    prompt_key: str = "prompt",
    as_conversation: bool = True,
) -> list[dict]:
    """Build a list of message dicts from raw data."""
    if prompt_key not in data:
        raise KeyError(f"Key '{prompt_key}' not found in data")

    prompt = data[prompt_key]

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        else:
            return [{"role": "user", "content": prompt}]

    # Already a list of message dicts
    return list(prompt)
