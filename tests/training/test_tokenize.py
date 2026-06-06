"""tests for training.tokenize."""

from __future__ import annotations

import json

from agenticml.training.tokenize import parse_training_prompt, tokenize_data_for_training
from agenticml.training.types import TrainingPromptField
from tests.fake_tokenizer import FakeTokenizer


def test_parse_training_prompt_frames():
    frames = [{"type": "goal", "content": "g"}, {"type": "action", "content": {"tool": "x"}}]
    out = parse_training_prompt({"frames": json.dumps(frames)}, TrainingPromptField.FRAMES)
    assert out == frames


def test_parse_training_prompt_drops_invalid():
    assert parse_training_prompt({}, TrainingPromptField.FRAMES) is None
    assert parse_training_prompt({"frames": "not json"}, TrainingPromptField.FRAMES) is None


def test_tokenize_drops_empty_row():
    out = tokenize_data_for_training(
        {},
        tokenizer=FakeTokenizer(),
        max_length=512,
        prompt_field=TrainingPromptField.FRAMES,
    )
    assert out["input_ids"] == []
