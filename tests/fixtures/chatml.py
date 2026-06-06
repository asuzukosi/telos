"""shared chatml tokenizer fixtures for harness tests."""

from __future__ import annotations

import json
from typing import Any

from transformers import PreTrainedTokenizerBase


class FakeChatMLTokenizer(PreTrainedTokenizerBase):
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tools: list[dict[str, Any] | Any] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        _ = (
            tools,
            documents,
            chat_template,
            continue_final_message,
            padding,
            truncation,
            max_length,
            return_tensors,
            return_dict,
            return_assistant_tokens_mask,
            tokenizer_kwargs,
            kwargs,
        )
        text = json.dumps(conversation) + ("<|assistant|>" if add_generation_prompt else "")
        if tokenize:
            return [ord(c) for c in text]
        return text

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> str:
        _ = skip_special_tokens, clean_up_tokenization_spaces, kwargs
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(i) for i in token_ids if 32 <= i < 127)

    def convert_tokens_to_ids(self, token: str) -> int:
        _ = token
        return 99

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        _ = kwargs
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return ord(token) if len(token) == 1 else 0

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index)

    def get_vocab(self) -> dict[str, int]:
        return {}

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        _ = save_directory, filename_prefix
        return ()
