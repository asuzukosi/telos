"""integration tests for swe chatml model adapter."""

from __future__ import annotations

import json
from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.swe.chatml import run_one_task
from telos.evaluation.benchmarks.swe.loop import run_chatml_swe
from telos.evaluation.benchmarks.swe.prelude import instance_to_messages
from telos.evaluation.benchmarks.swe.registry import SweEnvBridge
from telos.evaluation.harness.backends.chatml_backend import ChatMLBackend
from telos.runtime.hf_generator import HfGenerator


class _FakeChatMLTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_):
        text = json.dumps(messages) + ("<|assistant|>" if add_generation_prompt else "")
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def convert_tokens_to_ids(self, token: str):
        del token
        return 99


class _ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, input_ids, eos_token_id, max_new_tokens, *, pad_token_id=None, **_):
        del input_ids, eos_token_id, max_new_tokens, pad_token_id
        return [ord(c) for c in self.responses.pop(0)]


class _FakeEnv:
    def __init__(self, outputs: list[dict] | None = None, *, submit_patch: str | None = None):
        self.outputs = list(outputs or [])
        self.commands: list[str] = []
        self.submit_patch = submit_patch

    def execute(self, action: dict, cwd: str = "") -> dict:
        del cwd
        cmd = action.get("command", "")
        self.commands.append(cmd)
        if self.submit_patch is not None and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            from minisweagent.exceptions import Submitted

            raise Submitted(
                {
                    "role": "exit",
                    "content": self.submit_patch,
                    "extra": {"exit_status": "Submitted", "submission": self.submit_patch},
                }
            )
        if self.outputs:
            return self.outputs.pop(0)
        return {"output": "ok\n", "returncode": 0, "exception_info": ""}


def _instance() -> dict:
    return {
        "instance_id": "django__django-11099",
        "problem_statement": "fix the url validator bug",
    }


def _backend(responses: list[str]) -> ChatMLBackend:
    return ChatMLBackend(
        cast(PreTrainedTokenizerBase, _FakeChatMLTokenizer()),
        cast(HfGenerator, _ScriptedGenerator(responses)),
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
    env = _FakeEnv(
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
    env = _FakeEnv([{"output": "fatal\n", "returncode": 128, "exception_info": ""}])
    result = run_chatml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=10)
    assert result.stopped_on == "repeated_command"
    assert result.iterations == 3
    assert env.commands == [cmd, cmd]


def test_run_one_task_chatml_with_injected_env():
    patch = "--- a/x.py\n"
    env = _FakeEnv(submit_patch=patch)
    backend = _backend([_bash_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")])
    ctx = RunContext(model_id="org/model", format="chatml")
    row = run_one_task(backend, _instance(), ctx, env=env, max_iterations=3)
    assert row["format"] == "chatml"
    assert row["success"] is True
    assert row["model_patch"] == patch
    assert len(row["messages"]) > 0
