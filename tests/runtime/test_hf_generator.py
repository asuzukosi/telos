# tests/runtime/test_hf_generator.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agenticml.constants import DEFAULT_BASE_MODEL, END_MARKER_TOKEN_ID
from agenticml.evaluation.harness.load import as_causal_lm, causal_lm_load_kwargs
from agenticml.runtime.hf_generator import HfGenerator
from agenticml.tokenizer_helpers import agenticml_stop_token_ids, chat_template_ids, pad_token_id


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for hf_generator integration tests",
)


def _prelude_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    goal: str,
    mission: str,
) -> list[int]:
    return chat_template_ids(
        tokenizer,
        [
            {"type": "goal", "content": goal},
            {"type": "mission", "content": mission},
        ],
        add_generation_prompt=False,
        add_special_tokens=False,
    )


def _require_init_model() -> None:
    try:
        AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    except OSError as e:
        pytest.skip(f"init model not on hub yet: {e}")


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    _require_init_model()
    return AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)


@pytest.fixture(scope="module")
def gen() -> HfGenerator:
    _require_init_model()
    return HfGenerator.from_pretrained(DEFAULT_BASE_MODEL, dtype=torch.bfloat16)


def test_hf_generator_returns_suffix_only(tokenizer: PreTrainedTokenizerBase, gen: HfGenerator):
    prompt_ids = _prelude_input_ids(
        tokenizer,
        goal="You are concise.",
        mission="Say hi.",
    )
    out = gen.generate(
        prompt_ids,
        pad_token_id=pad_token_id(tokenizer),
        eos_token_id=agenticml_stop_token_ids(tokenizer),
        max_new_tokens=24,
    )
    assert isinstance(out, list)
    assert len(out) > 0
    assert out != prompt_ids


def test_hf_generator_respects_max_new_tokens(tokenizer: PreTrainedTokenizerBase, gen: HfGenerator):
    prompt_ids = _prelude_input_ids(tokenizer, goal="test", mission="test")
    out = gen.generate(
        prompt_ids,
        pad_token_id=pad_token_id(tokenizer),
        eos_token_id=END_MARKER_TOKEN_ID,
        max_new_tokens=8,
    )
    assert len(out) <= 8


def test_hf_generator_emits_attention_mask(
    monkeypatch,
    tokenizer: PreTrainedTokenizerBase,
    gen: HfGenerator,
):
    captured: dict = {}

    original_generate = as_causal_lm(gen.model).generate

    def wrapped_generate(*args, **kwargs):
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["pad_token_id"] = kwargs.get("pad_token_id")
        captured["eos_token_id"] = kwargs.get("eos_token_id")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(gen.model, "generate", wrapped_generate)
    pad = pad_token_id(tokenizer)
    stops = agenticml_stop_token_ids(tokenizer)
    _ = gen.generate(
        _prelude_input_ids(tokenizer, goal="x", mission="y"),
        pad_token_id=pad,
        eos_token_id=stops,
        max_new_tokens=4,
    )

    assert captured["attention_mask"] is not None
    assert captured["attention_mask"].dtype == torch.long
    assert captured["pad_token_id"] == pad
    assert captured["eos_token_id"] == stops


def test_from_pretrained_uses_shared_load_kwargs(monkeypatch):
    seen: dict = {}

    def fake_load_model(model_id, dtype=torch.bfloat16):
        seen["model_id"] = model_id
        seen["load_kw"] = causal_lm_load_kwargs(dtype)
        return MagicMock()

    monkeypatch.setattr(
        "agenticml.runtime.hf_generator.load_model",
        fake_load_model,
    )
    HfGenerator.from_pretrained("test-model", dtype=torch.float16)
    assert seen["model_id"] == "test-model"
    assert "max_memory" in seen["load_kw"] or seen["load_kw"].get("device_map") == "cpu"
