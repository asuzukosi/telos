"""tests for tokenizer_helpers."""

from __future__ import annotations

from agenticml.constants import END_MARKER_TOKEN_ID
from agenticml.tokenizer_helpers import (
    agenticml_stop_token_ids,
    chat_template_ids,
    chatml_stop_token_ids,
    eos_token_id,
    pad_token_id,
    require_single_token_id,
    single_token_id,
)
from tests.fake_tokenizer import FakeTokenizer


class _StubTokenizer(FakeTokenizer):
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def convert_tokens_to_ids(self, token: str) -> int | None:
        mapping = {
            "<|eot_id|>": 10,
            "<|eom_id|>": 11,
            "<|reserved_special_token_0|>": 20,
        }
        return mapping.get(token)


def test_single_token_id_reads_stub_method():
    tok = _StubTokenizer()
    assert single_token_id(tok, "<|eot_id|>") == 10
    assert single_token_id(tok, "missing") is None


def test_require_single_token_id_raises_for_missing():
    tok = _StubTokenizer()
    assert require_single_token_id(tok, "<|reserved_special_token_0|>") == 20


def test_chat_template_ids_render_then_encode():
    tok = FakeTokenizer()
    frames = [{"type": "goal", "content": "hi"}]
    ids = chat_template_ids(tok, frames, add_generation_prompt=False, add_special_tokens=False)
    wire = tok.apply_chat_template(frames, tokenize=False, add_generation_prompt=False)
    assert ids == tok.encode(wire, add_special_tokens=False)
    assert len(ids) > 1


def test_pad_and_stop_ids():
    tok = _StubTokenizer()
    assert pad_token_id(tok) == 0
    assert eos_token_id(tok) == 1
    assert chatml_stop_token_ids(tok) == [10, 11, 1]
    assert agenticml_stop_token_ids(tok) == [END_MARKER_TOKEN_ID, 1]
