from __future__ import annotations

import json
from typing import List

from transformers import PreTrainedTokenizerBase

from agenticml.constants import MODEL_MARKER_TOKEN_IDS, RUNTIME_MARKER_TOKEN_IDS
from agenticml.training.labels import mask_labels
from agenticml.training.types import TrainingPromptField
from agenticml.trajectory import Trajectory

_EMPTY_BATCH = {"input_ids": [], "labels": [], "attention_mask": []}


def parse_training_prompt(
    ex: dict,
    prompt_field: TrainingPromptField,
) -> list[dict] | None:
    raw = ex.get(prompt_field.value)
    if raw is None:
        return None
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    if prompt_field is TrainingPromptField.FRAMES:
        return Trajectory(items).to_dict()
    return items


def normalize_token_ids(ids) -> list[int] | None:
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
        return None
    return ids


def truncate(ids: List[int], labels: List[int], max_length: int):
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]
    return ids, labels


def tokenize_data_for_training(
    ex: dict,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    prompt_field: TrainingPromptField = TrainingPromptField.FRAMES,
) -> dict:
    """tokenize one dataset row via apply_chat_template."""
    prompt = parse_training_prompt(ex, prompt_field)
    if prompt is None:
        return dict(_EMPTY_BATCH)

    encode_kwargs: dict = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if prompt_field is TrainingPromptField.FRAMES:
        encode_kwargs["add_special_tokens"] = False
    try:
        ids = tokenizer.apply_chat_template(prompt, **encode_kwargs)
    except Exception:
        return dict(_EMPTY_BATCH)

    ids = normalize_token_ids(ids)
    if ids is None:
        return dict(_EMPTY_BATCH)

    labels = mask_labels(
        ids,
        prompt,
        tokenizer,
        prompt_field,
        agent_marker_ids=MODEL_MARKER_TOKEN_IDS,
        runtime_marker_ids=RUNTIME_MARKER_TOKEN_IDS,
    )

    ids, labels = truncate(ids, labels, max_length)
    if not ids or all(x == -100 for x in labels):
        return dict(_EMPTY_BATCH)

    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
