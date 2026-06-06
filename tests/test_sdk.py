"""Tests for agenticml.sdk."""
import json
from agenticml.sdk import StepResult, has_tool_obs, render_tool_schema, step, with_tool_obs
from agenticml.constants import WIRE_END_MARKER
from tests.wire_fixtures import W_ACTION, W_BELIEF, W_RESULT
from agenticml.trajectory import Trajectory
from tests.fake_tokenizer import FakeTokenizer
from tests.fixtures.generators import SdkScriptedGenerator


def test_render_tool_schema_empty_returns_empty():
    assert render_tool_schema([]) == ""


def test_render_tool_schema_emits_full_signature():
    tools = [{
        "name": "read_file",
        "description": "Read a file's contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read"},
                "encoding": {"type": "string"},
            },
            "required": ["path"],
        },
    }]
    out = render_tool_schema(tools)
    assert out.startswith("tools:\nnamespace tools {")
    assert "// Read a file's contents." in out
    assert "type read_file = (_: {" in out
    assert "path: string," in out
    assert "// Path to read" in out
    assert "encoding?: string," in out
    assert "}) => any;" in out
    assert out.endswith("}")


def test_render_tool_schema_handles_enum_and_no_args():
    no_args = render_tool_schema([{"name": "ping", "description": "no args"}])
    assert "type ping = () => any;" in no_args

    enum_out = render_tool_schema([{
        "name": "set_mode",
        "parameters": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
            "required": ["mode"],
        },
    }])
    assert "mode: 'fast' | 'slow'" in enum_out or 'mode: "fast" | "slow"' in enum_out


def test_step_accepts_list_of_dicts():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = SdkScriptedGenerator(f'{W_ACTION}{{"tool":"answer","text":"ok"}}')
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)


def test_step_accepts_trajectory_instance():
    from agenticml.frames import goal, mission
    trajectory = Trajectory([goal("g"), mission("m")])
    gen = SdkScriptedGenerator(f'{W_ACTION}{{"tool":"answer","text":"ok"}}')
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)


def test_step_accepts_list_of_frame_objects():
    from agenticml.frames import goal, mission
    trajectory = [goal("g"), mission("m")]
    gen = SdkScriptedGenerator(f'{W_ACTION}{{"tool":"answer","text":"ok"}}')
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result.trajectory, Trajectory)


def test_step_appends_model_frames():
    trajectory = [
        {"type": "goal", "content": "be helpful"},
        {"type": "mission", "content": "answer briefly"},
    ]
    gen = SdkScriptedGenerator(
        f'{W_BELIEF}I know this.{W_ACTION}{{"tool":"answer","text":"42"}}'
    )
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    assert isinstance(result, StepResult)
    assert result.stopped_on == WIRE_END_MARKER
    assert isinstance(result.new_frames, Trajectory)
    types_short = [f["type"] for f in result.new_frames.to_dict()]
    assert types_short == ["belief", "action", "end"]


def test_step_extended_trajectory_starts_with_input():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = SdkScriptedGenerator(f'{W_ACTION}{{"tool":"answer","text":"ok"}}')
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    extended_dicts = result.trajectory.to_dict()
    assert extended_dicts[:2] == trajectory


def test_with_tool_obs_inserts_obs_after_goal():
    trajectory = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    tools = [{
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }]
    out = with_tool_obs(trajectory, tools)
    types = [f["type"] for f in out.to_dict()]
    assert types == ["goal", "obs", "mission"]
    assert has_tool_obs(out)


def test_with_tool_obs_skips_when_obs_already_present():
    trajectory = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "obs", "content": "tools:\nnamespace tools {\n  type ping = () => any;\n}"},
        {"type": "mission", "content": "m"},
    ])
    tools = [{"name": "read_file", "parameters": {"type": "object", "properties": {}, "required": []}}]
    out = with_tool_obs(trajectory, tools)
    assert [f["type"] for f in out.to_dict()] == ["goal", "obs", "mission"]


def test_step_result_to_dict_is_json_serializable():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = SdkScriptedGenerator(f'{W_ACTION}{{"tool":"answer","text":"ok"}}')
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    d = result.to_dict()
    serialized = json.dumps(d)
    assert "trajectory" in d
    assert "new_frames" in d
    assert "stopped_on" in d
    assert "raw_text" in d
    assert json.loads(serialized) == d


def test_step_captures_parse_error():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = SdkScriptedGenerator("this is not a frame")
    result = step(trajectory, tokenizer=FakeTokenizer(), generate=gen)
    assert result.stopped_on.startswith("parse_error")
    assert len(result.new_frames) == 0
    assert result.trajectory.to_dict() == trajectory


def test_step_strict_mode_rejects_runtime_owned_frame():
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    gen = SdkScriptedGenerator(f'{W_RESULT}{{"tool":"bash","value":1}}')
    result = step(
        trajectory,
        tokenizer=FakeTokenizer(),
        generate=gen,
        strict=True,
    )
    assert "parse_error" in result.stopped_on
    assert len(result.new_frames) == 0
