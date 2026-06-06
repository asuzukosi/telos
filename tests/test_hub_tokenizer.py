"""smoke tests for published hub agenticml tokenizers (skipped when hub unreachable)."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from agenticml.agentic_template import (
    AGENTIC_CHAT_TEMPLATE,
    parse_reserved_wire,
    render_trajectory,
)
from agenticml.constants import END_MARKER_TOKEN_ID
from agenticml.frames import action, end, goal, mission
from agenticml.trajectory import Trajectory

INIT_REPO = "kosiasuzu/agenticml-agent-llama-3.1-8b-init"
MERGED_REPO = "kosiasuzu/agenticml-llama3.1-8b-lora-merged"


def _load_hub_tokenizer(repo_id: str):
    try:
        return AutoTokenizer.from_pretrained(repo_id)
    except Exception as exc:
        pytest.skip(f"hub tokenizer unavailable: {exc}")


@pytest.mark.parametrize("repo_id", [INIT_REPO, MERGED_REPO])
def test_hub_tokenizer_has_agentic_chat_template(repo_id: str):
    tokenizer = _load_hub_tokenizer(repo_id)
    assert tokenizer.chat_template
    assert tokenizer.chat_template.strip() == AGENTIC_CHAT_TEMPLATE.strip()


@pytest.mark.parametrize("repo_id", [INIT_REPO, MERGED_REPO])
def test_hub_tokenizer_round_trip(repo_id: str):
    tokenizer = _load_hub_tokenizer(repo_id)
    traj = Trajectory([
        goal("You are helpful."),
        mission("Say hi."),
        action({"tool": "answer", "text": "hi"}),
        end(),
    ])
    wire = render_trajectory(tokenizer, traj)
    assert wire
    ids = tokenizer.apply_chat_template(
        traj.to_dict(),
        tokenize=True,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    assert END_MARKER_TOKEN_ID in ids
    parsed = parse_reserved_wire(tokenizer.decode(ids), strict=False)
    assert parsed[-1].type.value == "<|end|>"
