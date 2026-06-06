"""integration tests for swe telos model adapter (run_telos_swe, run_one_task)."""

from __future__ import annotations

from typing import cast

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.swe.loop import run_telos_swe
from telos.evaluation.benchmarks.swe.registry import SweEnvBridge
from telos.evaluation.benchmarks.swe.telos import run_one_task
from telos.evaluation.harness.backends.telos_backend import TelosBackend
from telos.tokenizer import TelosTokenizer
from telos.runtime.hf_generator import HfGenerator


class _FakeTokenizer:
    end_id = 999_999

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.end_id:
                out.append("<|end|>")
            else:
                out.append(chr(i))
        return "".join(out)

    @property
    def hf(self):
        return self


class _ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, input_ids, eos_token_id, max_new_tokens, *, pad_token_id=None, return_full_sequence=False):
        del input_ids, pad_token_id, return_full_sequence
        text = self.responses.pop(0)
        ids = [ord(c) for c in text]
        stop = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        ids.append(stop)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        return ids


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


def _backend(responses: list[str]) -> TelosBackend:
    return TelosBackend(tokenizer=cast(TelosTokenizer, _FakeTokenizer()), generator=cast(HfGenerator, _ScriptedGenerator(responses)))


def test_run_telos_swe_bash_then_submit():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@\n"
    env = _FakeEnv(
        [{"output": "foo.py\n", "returncode": 0, "exception_info": ""}],
        submit_patch=patch,
    )
    backend = _backend(
        [
            '<|action|>{"tool":"bash","command":"ls"}',
            '<|action|>{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}',
        ]
    )
    result = run_telos_swe(backend, SweEnvBridge(env), _instance(), max_iterations=5)
    assert result.stopped_on == "submitted"
    assert result.model_patch == patch
    assert result.iterations == 2
    assert env.commands == [
        "ls",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
    ]
    assert result.trajectory is not None
    types = [f["type"] for f in result.trajectory.to_dict()]
    assert "result" in types


def test_run_telos_swe_no_action_stops():
    backend = _backend(['<|belief|>thinking only'])
    env = _FakeEnv()
    result = run_telos_swe(backend, SweEnvBridge(env), _instance(), max_iterations=3)
    assert result.stopped_on == "no_action"
    assert env.commands == []


def test_run_one_task_with_injected_env():
    patch = "--- a/x.py\n"
    env = _FakeEnv(submit_patch=patch)
    backend = _backend(
        ['<|action|>{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}']
    )
    ctx = RunContext(model_id="org/model", format="telos")
    row = run_one_task(backend, _instance(), ctx, env=env, max_iterations=3)
    assert row["format"] == "telos"
    assert row["success"] is True
    assert row["model_patch"] == patch
    assert row["instance_id"] == "django__django-11099"


def test_run_telos_swe_stops_on_repeated_command():
    same = '<|action|>{"tool":"bash","command":"git fetch origin"}'
    backend = _backend([same, same, same])
    env = _FakeEnv([{"output": "fatal\n", "returncode": 128, "exception_info": ""}])
    result = run_telos_swe(backend, SweEnvBridge(env), _instance(), max_iterations=10)
    assert result.stopped_on == "repeated_command"
    assert result.iterations == 3
    assert env.commands == ["git fetch origin", "git fetch origin"]


def test_run_telos_swe_records_token_stats():
    backend = _backend(['<|action|>{"tool":"bash","command":"pwd"}'])
    env = _FakeEnv([{"output": "/testbed\n", "returncode": 0, "exception_info": ""}])
    result = run_telos_swe(
        backend,
        SweEnvBridge(env),
        _instance(),
        max_iterations=1,
        max_new_tokens=64,
    )
    assert result.stopped_on == "max_iterations"
    assert result.prompt_tokens > 0
    assert result.generated_tokens > 0
    assert result.inference_sec >= 0.0
