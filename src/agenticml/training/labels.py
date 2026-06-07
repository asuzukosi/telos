from __future__ import annotations

from typing import AbstractSet

from transformers import PreTrainedTokenizerBase

from agenticml.training.types import TrainingPromptField


def mask_agenticml_runtime_labels(
    input_ids: list[int],
    agent_marker_ids: AbstractSet[int],
    runtime_marker_ids: AbstractSet[int],
) -> list[int]:
    labels = list(input_ids)
    in_model_block = False
    for i, tok in enumerate(input_ids):
        if tok in runtime_marker_ids:
            in_model_block = False
        elif tok in agent_marker_ids:
            in_model_block = True
        if not in_model_block:
            labels[i] = -100
    return labels


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def mask_assistant_only(input_ids: list[int], tokenizer) -> list[int]:
    """build labels so loss applies only on assistant turns (chatml fallback).

    the model still sees the full conversation in input_ids. labels use -100 on
    positions we do not train on (system, user, tool, etc.). assistant message
    content and its closing <|eot_id|> copy input_ids so the model learns to
    predict those tokens.
    """
    labels = [-100] * len(input_ids)

    start_header = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    end_header = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
    end_of_turn = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    assistant_role = tokenizer.encode("assistant", add_special_tokens=False)

    pos = 0
    while pos < len(input_ids):
        header_start = _find_subseq(input_ids, start_header, pos)
        if header_start == -1:
            break

        role_start = header_start + len(start_header)
        header_end = _find_subseq(input_ids, end_header, role_start)
        if header_end == -1:
            break

        role_token_ids = input_ids[role_start:header_end]
        content_start = header_end + len(end_header)

        turn_end = _find_subseq(input_ids, end_of_turn, content_start)
        if turn_end == -1:
            turn_end = len(input_ids)

        if role_token_ids == assistant_role:
            supervise_until = min(turn_end + len(end_of_turn), len(input_ids))
            for j in range(content_start, supervise_until):
                labels[j] = input_ids[j]

        pos = turn_end + len(end_of_turn)

    return labels


def mask_chatml_labels(
    ids: list[int],
    prompt: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    _ = prompt
    return mask_assistant_only(ids, tokenizer)


def mask_labels(
    ids: list[int],
    prompt: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    prompt_field: TrainingPromptField,
    *,
    agent_marker_ids: AbstractSet[int],
    runtime_marker_ids: AbstractSet[int],
) -> list[int]:
    if prompt_field is TrainingPromptField.FRAMES:
        return mask_agenticml_runtime_labels(ids, agent_marker_ids, runtime_marker_ids)
    return mask_chatml_labels(ids, prompt, tokenizer)
