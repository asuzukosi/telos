"""integration tests for swe chatml model adapter."""

from __future__ import annotations

import json

import pytest

from agenticml.evaluation.benchmarks.suite import RunContext
from agenticml.evaluation.benchmarks.swe.chatml import run_one_task
from agenticml.evaluation.benchmarks.swe.loop import run_chatml_swe
from agenticml.evaluation.benchmarks.swe.prelude import instance_to_messages
from agenticml.evaluation.benchmarks.swe.registry import SweEnvBridge
from agenticml.evaluation.harness.backends.chatml_backend import ChatMLBackend
from tests.fixtures.chatml import FakeChatMLTokenizer
from tests.fixtures.generators import HfScriptedGenerator
from tests.fixtures.swe import SweFakeEnv


def _instance() -> dict:
    return {
        "instance_id": "django__django-11099",
        "problem_statement": "fix the url validator bug",
    }


def _backend(responses: list[str]) -> ChatMLBackend:
    return ChatMLBackend(
        FakeChatMLTokenizer(),
        HfScriptedGenerator(responses, append_stop=False),
    )


def _bash_call(command: str) -> str:
    args = json.dumps({"command": command})
    return f'<|python_tag|>{{"name": "bash", "arguments": {json.dumps(args)}}}<|eom_id|>'


def test_instance_to_messages():
    msgs = instance_to_messages(_instance())
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "fix the url validator bug" in msgs[1]["content"]


def test_run_chatml_swe_bash_then_submit():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@\n"
    env = SweFakeEnv(
        [{"output": "foo.py\n", "returncode": 0, "exception_info": ""}],
        submit_patch=patch,
    )
    backend = _backend(
        [
            _bash_call("ls"),
            _bash_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    )
    result = run_chatml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=5)
    assert result.stopped_on == "submitted"
    assert result.model_patch == patch
    assert result.iterations == 2
    assert result.messages is not None
    assert any(m.get("role") == "tool" for m in result.messages)
    assert env.commands[0] == "ls"


def test_run_chatml_swe_stops_on_repeated_command():
    cmd = "git fetch origin"
    backend = _backend([_bash_call(cmd), _bash_call(cmd), _bash_call(cmd)])
    env = SweFakeEnv([{"output": "fatal\n", "returncode": 128, "exception_info": ""}])
    result = run_chatml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=10)
    assert result.stopped_on == "repeated_command"
    assert result.iterations == 3
    assert env.commands == [cmd, cmd]


def test_run_one_task_chatml_with_injected_env():
    patch = "--- a/x.py\n"
    env = SweFakeEnv(submit_patch=patch)
    backend = _backend([_bash_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")])
    ctx = RunContext(model_id="org/model", format="chatml")
    row = run_one_task(backend, _instance(), ctx, env=env, max_iterations=3)
    assert row["format"] == "chatml"
    assert row["success"] is True
    assert row["model_patch"] == patch
    assert len(row["messages"]) > 0
