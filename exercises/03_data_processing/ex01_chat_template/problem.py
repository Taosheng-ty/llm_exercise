"""
Exercise 01: Chat Template Builder (Easy)

In LLM training pipelines (like slime's _build_messages), raw conversation data
must be converted into a formatted string that the model can understand. The most
common format is ChatML, used by models like Qwen and many others.

ChatML format wraps each message like this:
    <|im_start|>{role}\n{content}<|im_end|>\n

Your task: implement functions to apply ChatML-style templates to conversation data.

Reference: slime/utils/data.py _build_messages() and tokenizer.apply_chat_template()
"""


def apply_chatml_template(
    messages: list[dict],
    system_message: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Apply ChatML template to a list of messages.

    Each message is a dict with "role" and "content" keys.
    The ChatML format for each message is:
        <|im_start|>{role}\n{content}<|im_end|>\n

    Args:
        messages: List of {"role": str, "content": str} dicts.
            Roles are typically "system", "user", or "assistant".
        system_message: If provided, prepend a system message at the start.
            Only prepend if there is no existing system message as the first message.
        add_generation_prompt: If True, append "<|im_start|>assistant\n" at the end
            to prompt the model to generate an assistant response.

    Returns:
        The formatted string with all messages in ChatML format.

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> apply_chatml_template(messages)
        '<|im_start|>user\\nHello<|im_end|>\\n'
    """
    # TODO: Implement this function
    raise NotImplementedError


def build_messages(
    data: dict,
    prompt_key: str = "prompt",
    as_conversation: bool = True,
) -> list[dict]:
    """Build a list of message dicts from raw data, similar to slime's _build_messages.

    The data dict may contain the prompt in different formats:
    1. A string -> wrap in [{"role": "user", "content": prompt}] if as_conversation is True,
       otherwise return the string as-is.
    2. A list of message dicts -> return as-is (already in conversation format).

    Args:
        data: Dict containing the prompt data.
        prompt_key: Key to look up the prompt in data. Defaults to "prompt".
        as_conversation: If True, convert string prompts to conversation format.
            If False, return string prompts as-is.

    Returns:
        A list of message dicts if as_conversation is True,
        or the raw string if as_conversation is False and prompt is a string.

    Raises:
        KeyError: If prompt_key is not found in data.
    """
    # TODO: Implement this function
    raise NotImplementedError
