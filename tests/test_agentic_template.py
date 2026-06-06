"""tests for agenticml chat-template rendering and init bake."""

from __future__ import annotations

from jinja2 import Template

from agenticml.agentic_template import (
    AGENTIC_CHAT_TEMPLATE,
    bake_agentic_template,
    parse_reserved_wire,
    render_trajectory,
    reserved_to_markers,
)
from agenticml.constants import FRAME_WIRE_MARKERS, FrameType
from agenticml.frames import action, end, goal, mission, result
from agenticml.trajectory import Trajectory
from tests.fake_tokenizer import FakeTokenizer


def test_to_dict_is_template_input():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "bash", "command": "ls"}),
        end(),
        result({"tool": "bash", "value": "out"}),
    ]
    trajectory = Trajectory(frames)
    assert [m["type"] for m in trajectory.to_dict()] == [
        "goal", "mission", "action", "end", "result",
    ]


def test_bake_agentic_template_sets_jinja():
    tok = FakeTokenizer()
    tok.chat_template = "{{ not agenticml }}"
    bake_agentic_template(tok)
    assert tok.chat_template == AGENTIC_CHAT_TEMPLATE


def test_render_trajectory_uses_apply_chat_template():
    frames = [goal("g"), action({"tool": "answer", "text": "ok"}), end()]
    trajectory = Trajectory(frames)
    tokenizer = FakeTokenizer()
    wire = render_trajectory(tokenizer, trajectory)
    template_wire = Template(AGENTIC_CHAT_TEMPLATE).render(
        messages=trajectory.to_dict()
    )
    assert wire == template_wire
    assert "<|reserved_special_token_0|>g" in wire
    assert "<|reserved_special_token_6|>" in wire
    assert FRAME_WIRE_MARKERS[FrameType.END] in wire
    assert reserved_to_markers(wire) == (
        '<|goal|>g<|action|>{"text": "ok", "tool": "answer"}<|end|>'
    )


def test_parse_reserved_wire_from_rendered_trajectory():
    trajectory = Trajectory([
        goal("g"),
        action({"tool": "answer", "text": "ok"}),
        end(),
    ])
    tokenizer = FakeTokenizer()
    wire = render_trajectory(tokenizer, trajectory)
    parsed = parse_reserved_wire(wire)
    assert len(parsed) == 3
    assert parsed[1].content == {"tool": "answer", "text": "ok"}
    assert parsed[2].type is FrameType.END
