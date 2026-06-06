from __future__ import annotations

import pytest

from agenticml.constants import WIRE_END_MARKER
from tests.wire_fixtures import W_ACTION, W_BELIEF
from agenticml.evaluation.harness.backends import AgenticMLBackend
from agenticml.runtime import Tool, ToolRegistry
from agenticml.trajectory import Trajectory
from tests.fake_tokenizer import FakeTokenizer
from tests.fixtures.generators import HfScriptedGenerator


@pytest.fixture
def backend() -> AgenticMLBackend:
    return AgenticMLBackend(
        tokenizer=FakeTokenizer(),
        generator=HfScriptedGenerator([]),
    )


def test_agenticml_backend_format(backend: AgenticMLBackend):
    assert backend.format == "agenticml"


def test_agenticml_backend_step(backend: AgenticMLBackend):
    backend.generator = HfScriptedGenerator(
        [f'{W_BELIEF}ok{W_ACTION}{{"tool":"answer","text":"42"}}']
    )
    trajectory = [
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ]
    out = backend.step(trajectory, max_new_tokens=256)
    step = out.step
    assert step is not None
    assert step.stopped_on == WIRE_END_MARKER
    assert out.generated_tokens > 0
    assert out.prompt_tokens > 0
    assert out.inference_sec >= 0.0
    types = [f["type"] for f in step.new_frames.to_dict()]
    assert types == ["belief", "action", "end"]


def test_agenticml_backend_run_terminal_answer(backend: AgenticMLBackend):
    backend.generator = HfScriptedGenerator(
        [f'{W_ACTION}{{"tool":"answer","text":"42"}}']
    )
    reg = ToolRegistry()
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    out = backend.run(initial, reg)
    assert out.stopped_on == "terminal_action"
    assert out.final_answer == "42"
    assert out.iterations == 1
    assert out.generated_tokens > 0
    assert out.total_sec >= 0.0


def test_agenticml_backend_run_multi_step_tool(backend: AgenticMLBackend):
    backend.generator = HfScriptedGenerator([
        f'{W_ACTION}{{"tool":"echo","value":"a"}}',
        f'{W_ACTION}{{"tool":"answer","text":"done"}}',
    ])
    reg = ToolRegistry()
    reg.register(
        Tool(
            "echo",
            lambda value: value,
            {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        )
    )
    initial = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    out = backend.run(initial, reg)
    assert out.stopped_on == "terminal_action"
    assert out.iterations == 2
    assert out.tool_sec >= 0.0


def test_agenticml_backend_satisfies_protocol(backend: AgenticMLBackend):
    from agenticml.evaluation.harness.backend import ModelBackend

    assert isinstance(backend, ModelBackend)
