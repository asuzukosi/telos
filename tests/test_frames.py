import pytest
from agenticml.constants import FrameType
from agenticml.frames import (
    AgenticMLOwnershipError,
    AgenticMLParseError,
    action,
    belief,
    goal,
    reward,
    result,
    parse,
)


def test_constructors_set_type_and_content():
    assert goal("hi").type is FrameType.GOAL
    assert goal("hi").content == "hi"
    assert action({"tool": "x"}).content == {"tool": "x"}
    assert reward(0.5).content == 0.5
    assert reward(1).content == 1.0


def test_owner_property():
    assert goal("x").owner == "runtime"
    assert belief("x").owner == "model"
    assert action({"t": "x"}).owner == "model"
    assert result({"tool": "bash", "value": "out"}).owner == "runtime"


def test_frametype_string_equivalence():
    assert FrameType.GOAL == "<|goal|>"
    assert "<|goal|>" == FrameType.GOAL


def test_parse_single_prose_frame():
    frames = parse("<|goal|>Hello world")
    assert len(frames) == 1
    assert frames[0].type is FrameType.GOAL
    assert frames[0].content == "Hello world"
    assert frames[0].error is None


def test_parse_multiple_frames():
    text = "<|goal|>be helpful<|mission|>find the answer"
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.GOAL
    assert frames[0].content == "be helpful"
    assert frames[1].type is FrameType.MISSION
    assert frames[1].content == "find the answer"


def test_parse_action_json_frame_includes_end_marker():
    text = '<|action|>{"tool":"read_file","path":"main.py"}<|end|>'
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.ACTION
    assert frames[0].content == {"tool": "read_file", "path": "main.py"}
    assert frames[1].type is FrameType.END


def test_parse_reward_frame():
    frames = parse("<|reward|>0.75")
    assert len(frames) == 1
    assert frames[0].content == 0.75


def test_parse_negative_reward():
    frames = parse("<|reward|>-1.5")
    assert frames[0].content == -1.5


def test_parse_only_end_marker_yields_end_frame():
    frames = parse("<|end|>")
    assert len(frames) == 1
    assert frames[0].type is FrameType.END
    assert frames[0].content is None


def test_malformed_json_captured_in_error():
    text = '<|action|>{"tool":bad json}<|end|>'
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.ACTION
    assert frames[0].content is None
    assert frames[0].error is not None
    assert "json" in frames[0].error.lower()


def test_malformed_reward_captured_in_error():
    frames = parse("<|reward|>not_a_number")
    assert frames[0].error is not None
    assert frames[0].content is None


def test_parse_rejects_garbage_before_first_marker():
    with pytest.raises(AgenticMLParseError):
        parse("not a frame <|goal|>hi")


def test_parse_empty_string_returns_no_frames():
    assert parse("") == []
    assert parse("   \n   ") == []


def test_strict_mode_rejects_runtime_frame():
    text = '<|result|>{"tool":"bash","value":1}'
    frames = parse(text)
    assert len(frames) == 1
    with pytest.raises(AgenticMLOwnershipError, match="runtime-owned"):
        parse(text, strict=True)


def test_strict_mode_accepts_pure_model_output():
    text = (
        '<|belief|>I see three files.'
        '<|action|>{"tool":"answer","text":"ok"}'
        "<|end|>"
    )
    frames = parse(text, strict=True)
    assert len(frames) == 3
    assert [f.type for f in frames] == [FrameType.BELIEF, FrameType.ACTION, FrameType.END]


def test_reserved_to_markers_round_trip():
    from agenticml.agentic_template import parse_reserved_wire, reserved_to_markers
    from agenticml.constants import FRAME_WIRE_MARKERS, WIRE_END_MARKER

    wire = (
        f"{FRAME_WIRE_MARKERS[FrameType.GOAL]}hi"
        f'{FRAME_WIRE_MARKERS[FrameType.ACTION]}{{"tool":"answer","text":"ok"}}'
        f"{WIRE_END_MARKER}"
    )
    marker_wire = reserved_to_markers(wire)
    assert marker_wire == (
        '<|goal|>hi<|action|>{"tool":"answer","text":"ok"}<|end|>'
    )
    parsed = parse_reserved_wire(wire)
    assert len(parsed) == 3
    assert parsed[0].type is FrameType.GOAL
    assert parsed[1].type is FrameType.ACTION
    assert parsed[2].type is FrameType.END
