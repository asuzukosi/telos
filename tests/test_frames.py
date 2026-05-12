import pytest
from telos.constants import FrameType
from telos.frames import Frame, PayloadKind, TELOS_MARKERS, TelosOwnershipError, TelosParseError, action, belief, end, feedback, goal, mission, obs, plan, reward, result, think, TelosParseError, TelosOwnershipError, render_frame, render, parse

 
def test_constructors_set_type_and_content():
    assert goal("hi").type is FrameType.GOAL
    assert goal("hi").content == "hi"
    assert action({"tool": "x"}).content == {"tool": "x"}
    assert reward(0.5).content == 0.5
    assert reward(1).content == 1.0
    assert end().content is None
 
 
def test_owner_property():
    assert goal("x").owner == "runtime"
    assert belief("x").owner == "model"
    assert action({"t": "x"}).owner == "model"
    assert result({"ok": 1}).owner == "runtime"
 
 
def test_payload_kind_property():
    assert goal("x").kind == PayloadKind.PROSE
    assert action({"t": "x"}).kind == PayloadKind.JSON
    assert reward(1.0).kind == PayloadKind.NUMBER
    assert end().kind == PayloadKind.EMPTY

def test_frametype_string_equivalence():
    """FrameType inherits from str, so equality with marker strings holds."""
    assert FrameType.GOAL == "<|goal|>"
    assert "<|goal|>" == FrameType.GOAL

def test_render_prose_frame():
    assert render_frame(goal("Hello world")) == "<|goal|>Hello world"

def test_render_json_frame_is_compact():
    rendered = render_frame(action({"tool": "list_dir", "path": "/tmp"}))
    assert rendered == '<|action|>{"tool":"list_dir","path":"/tmp"}'

def test_render_number_frame_integer():
    assert render_frame(reward(1)) == "<|reward|>1"
    assert render_frame(reward(0)) == "<|reward|>0"
    assert render_frame(reward(-1)) == "<|reward|>-1"
 
 
def test_render_number_frame_float():
    assert render_frame(reward(0.5)) == "<|reward|>0.5"
 
 
def test_render_end_has_no_payload():
    assert render_frame(end()) == "<|end|>"

def test_render_trajectory_with_newline_separator():
    frames = [
        goal("You are an assistant."),
        mission("What is 2+2?"),
        action({"tool": "answer", "text": "4"}),
        end(),
    ]
    out = render(frames)
    assert "<|goal|>You are an assistant." in out
    assert "<|mission|>What is 2+2?" in out
    assert '<|action|>{"tool":"answer","text":"4"}' in out
    assert out.endswith("<|end|>")

def test_render_no_double_separator_when_content_ends_in_newline():
    frames = [
        goal("ends with newline\n"),
        mission("next frame"),
    ]
    out = render(frames)
    assert "\n\n<|mission|>" not in out
    assert "\n<|mission|>" in out
 
def test_parse_single_prose_frame():
    frames = parse("<|goal|>Hello world")
    assert len(frames) == 1
    assert frames[0].type is FrameType.GOAL
    assert frames[0].content == "Hello world"
    assert frames[0].error is None

def test_parse_multiple_frames():
    text = "<|goal|>be helpful\n<|mission|>find the answer"
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.GOAL
    assert frames[0].content == "be helpful"
    assert frames[1].type is FrameType.MISSION
    assert frames[1].content == "find the answer"

def test_parse_action_json_frame():
    text = '<|action|>{"tool":"read_file","path":"main.py"}<|end|>'
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.ACTION
    assert frames[0].content == {"tool": "read_file", "path": "main.py"}
    assert frames[1].type is FrameType.END
    assert frames[1].content is None
 
 
def test_parse_reward_frame():
    frames = parse("<|reward|>0.75")
    assert len(frames) == 1
    assert frames[0].content == 0.75
 
 
def test_parse_negative_reward():
    frames = parse("<|reward|>-1.5")
    assert frames[0].content == -1.5
 
 
def test_parse_end_frame_has_empty_content():
    frames = parse("<|end|>")
    assert len(frames) == 1
    assert frames[0].type is FrameType.END
    assert frames[0].content is None
    assert frames[0].error is None

def test_round_trip_full_trajectory():
    original = [
        goal("You are a file assistant."),
        mission("Find the largest file."),
        action({"tool": "list_dir", "path": "/tmp"}),
        end(),
        result({"ok": 1, "value": ["a.txt", "b.bin"]}),
        belief("Two files in /tmp; sizes unknown."),
        action({"tool": "answer", "text": "done"}),
        end(),
    ]
    rendered = render(original)
    parsed = parse(rendered)
    assert len(parsed) == len(original)
    for a, b in zip(original, parsed):
        assert a.type is b.type
        assert a.content == b.content
 
def test_malformed_json_captured_in_error():
    text = '<|action|>{"tool":bad json}<|end|>'
    frames = parse(text)
    assert len(frames) == 2
    assert frames[0].type is FrameType.ACTION
    assert frames[0].content is None
    assert frames[0].error is not None
    assert "json" in frames[0].error.lower()
    assert frames[1].type is FrameType.END
 
 
def test_malformed_reward_captured_in_error():
    frames = parse("<|reward|>not_a_number")
    assert frames[0].error is not None
    assert frames[0].content is None
 
 
def test_parse_rejects_garbage_before_first_marker():
    with pytest.raises(TelosParseError):
        parse("not a frame <|goal|>hi")
 
 
def test_parse_empty_string_returns_no_frames():
    assert parse("") == []
    assert parse("   \n   ") == []
 


def test_strict_mode_rejects_runtime_frame():
    text = '<|result|>{"ok":1,"value":1}'
    frames = parse(text)
    assert len(frames) == 1
    with pytest.raises(TelosOwnershipError, match="runtime-owned"):
        parse(text, strict=True)
 
 
def test_strict_mode_accepts_pure_model_output():
    text = '<|belief|>I see three files.<|action|>{"tool":"answer","text":"ok"}<|end|>'
    frames = parse(text, strict=True)
    assert len(frames) == 3
    assert [f.type for f in frames] == [FrameType.BELIEF, FrameType.ACTION, FrameType.END]

def test_telos_markers_count():
    assert len(TELOS_MARKERS) == 11
