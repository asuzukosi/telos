from __future__ import annotations
import json
import pytest
from agenticml.evaluation.harness.backend import ModelBackend
from agenticml.evaluation.harness.backends import ChatMLBackend
from agenticml.evaluation.harness.backends.chatml_backend import _parse, _to_messages
from agenticml.runtime import Tool, ToolRegistry
from agenticml.trajectory import Trajectory
from tests.fixtures.chatml import FakeChatMLTokenizer
from tests.fixtures.generators import HfScriptedGenerator


@pytest.fixture
def backend() -> ChatMLBackend:
    return ChatMLBackend(
        FakeChatMLTokenizer(),
        HfScriptedGenerator([], append_stop=False),
    )


def test_to_messages():
    msgs = _to_messages([{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}])
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_parse_tool_and_text():
    call, _, stop = _parse('<|python_tag|>{"name": "answer", "arguments": "{}"}<|eom_id|>')
    assert stop == "tool_call"
    assert call is not None
    assert call["name"] == "answer"
    _, text, stop = _parse("hello<|eot_id|>")
    assert text == "hello"
    assert stop == "assistant_text"


def test_run_terminal(backend: ChatMLBackend):
    backend.generator = HfScriptedGenerator([
        '<|python_tag|>{"name": "answer", "arguments": "{\\"text\\": \\"42\\"}"}<|eom_id|>',
    ], append_stop=False)
    out = backend.run(
        [{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}],
        ToolRegistry(),
    )
    assert out.stopped_on == "terminal_action"
    assert out.final_answer == "42"
    assert isinstance(backend, ModelBackend)


def test_run_tool_loop(backend: ChatMLBackend):
    backend.generator = HfScriptedGenerator([
        '<|python_tag|>{"name": "echo", "arguments": "{\\"value\\": \\"a\\"}"}<|eom_id|>',
        '<|python_tag|>{"name": "answer", "arguments": "{\\"text\\": \\"done\\"}"}<|eom_id|>',
    ], append_stop=False)
    reg = ToolRegistry()
    reg.register(Tool("echo", lambda value: value, {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}))
    out = backend.run(Trajectory([{"type": "goal", "content": "g"}, {"type": "mission", "content": "m"}]), reg)
    assert out.final_answer == "done"
    assert out.iterations == 2
