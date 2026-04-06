"""Tests for Exercise 01: Chat Template Builder"""

import pytest
from .solution import apply_chatml_template, build_messages


class TestApplyChatmlTemplate:
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = apply_chatml_template(messages)
        assert result == "<|im_start|>user\nHello<|im_end|>\n"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = apply_chatml_template(messages)
        expected = (
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello!<|im_end|>\n"
            "<|im_start|>user\nHow are you?<|im_end|>\n"
        )
        assert result == expected

    def test_system_message_prepend(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = apply_chatml_template(messages, system_message="You are helpful.")
        expected = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
        )
        assert result == expected

    def test_system_message_no_duplicate(self):
        """If messages already start with a system message, don't prepend another."""
        messages = [
            {"role": "system", "content": "Existing system msg"},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_chatml_template(messages, system_message="New system msg")
        # Should keep the existing system message, not prepend a new one
        assert result.count("<|im_start|>system") == 1
        assert "Existing system msg" in result

    def test_add_generation_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = apply_chatml_template(messages, add_generation_prompt=True)
        assert result.endswith("<|im_start|>assistant\n")

    def test_empty_messages_with_system(self):
        result = apply_chatml_template([], system_message="Be helpful.")
        assert result == "<|im_start|>system\nBe helpful.<|im_end|>\n"

    def test_empty_messages_no_system(self):
        result = apply_chatml_template([])
        assert result == ""


class TestBuildMessages:
    def test_string_prompt_as_conversation(self):
        data = {"prompt": "What is 2+2?"}
        result = build_messages(data, as_conversation=True)
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_string_prompt_not_conversation(self):
        data = {"prompt": "What is 2+2?"}
        result = build_messages(data, as_conversation=False)
        assert result == "What is 2+2?"

    def test_list_prompt_passthrough(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        data = {"prompt": messages}
        result = build_messages(data)
        assert result == messages

    def test_custom_prompt_key(self):
        data = {"question": "Tell me a joke"}
        result = build_messages(data, prompt_key="question")
        assert result == [{"role": "user", "content": "Tell me a joke"}]

    def test_missing_key_raises(self):
        data = {"other_field": "value"}
        with pytest.raises(KeyError):
            build_messages(data, prompt_key="prompt")
