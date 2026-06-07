"""tests for agenticml <-> chatml FormatBridge."""

from __future__ import annotations
import pytest
from agenticml.bridge import FormatBridge
from agenticml.frames import action, goal, mission
from tests.fake_tokenizer import FakeTokenizer


@pytest.fixture
def b() -> FormatBridge:
    return FormatBridge()


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


def test_frames_to_messages_basic(b: FormatBridge):
    frames = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    messages = b.frames_to_messages(frames)
    assert messages[0] == {"role": "system", "content": "g"}
    assert messages[1] == {"role": "user", "content": "m"}


def test_frames_to_messages_tool_loop(b: FormatBridge):
    frames = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
        {"type": "think", "content": "hmm"},
        {"type": "action", "content": {"tool": "bash", "command": "ls"}},
        {"type": "result", "content": {"tool": "bash", "value": "out"}},
    ]
    messages = b.frames_to_messages(frames)
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "bash"
    assert messages[2]["content"] == "hmm"
    assert messages[3]["role"] == "tool"


def test_frames_to_messages_skips_malformed_action(b: FormatBridge):
    frames = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
        {"type": "action", "content": None},
        {"type": "action", "content": {"tool": "answer", "text": "ok"}},
    ]
    messages = b.frames_to_messages(frames)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "ok"


def test_messages_to_frames_round_trip(b: FormatBridge):
    frames = [
        {"type": "goal", "content": "system goal"},
        {"type": "mission", "content": "do task"},
        {"type": "think", "content": "reason"},
        {"type": "action", "content": {"tool": "echo", "value": "x"}},
        {"type": "result", "content": {"tool": "echo", "value": "x"}},
        {"type": "feedback", "content": "nice"},
        {"type": "action", "content": {"tool": "answer", "text": "done"}},
    ]
    messages = b.frames_to_messages(frames)
    back = b.messages_to_frames(messages)
    assert back[0]["type"] == "goal"
    assert back[1]["type"] == "mission"
    assert back[2]["type"] == "think"
    assert back[3]["type"] == "action"
    assert back[3]["content"]["tool"] == "echo"
    assert back[4]["type"] == "result"
    assert back[5]["type"] == "feedback"
    assert back[6]["content"]["tool"] == "answer"


def test_coerce_messages_from_frames(b: FormatBridge):
    msgs = b.coerce_messages([{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}])
    assert msgs[0]["role"] == "system"


def test_coerce_frames_from_messages(b: FormatBridge):
    frames = b.coerce_frames([
        {"role": "system", "content": "g"},
        {"role": "user", "content": "m"},
    ])
    assert frames[0]["type"] == "goal"
    assert frames[1]["type"] == "mission"


def test_agenticml_wire_round_trip(b: FormatBridge, tok: FakeTokenizer):
    frames = [goal("g"), mission("m"), action({"tool": "answer", "text": "ok"})]
    wire = b.frames_to_agenticml_wire(frames, tokenizer=tok)
    back = b.agenticml_wire_to_frames(wire)
    assert back[0]["type"] == "goal"
    action_frames = [f for f in back if f["type"] == "action"]
    assert action_frames[-1]["content"]["tool"] == "answer"


def test_agenticml_wire_to_messages_marker_wire(b: FormatBridge):
    wire = "<|goal|>g<|mission|>m"
    messages = b.agenticml_wire_to_messages(wire)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_parse_chatml_generation_tool_call(b: FormatBridge):
    text = '<|python_tag|>{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}<|eom_id|>'
    parsed = b.parse_chatml_generation(text)
    assert parsed.stop_reason == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call["name"] == "bash"
