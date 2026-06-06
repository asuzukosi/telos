"""integration tests for swe agenticml model adapter (run_agenticml_swe, run_one_task)."""
from __future__ import annotations
from agenticml.evaluation.benchmarks.suite import RunContext
from agenticml.evaluation.benchmarks.swe.loop import run_agenticml_swe
from agenticml.evaluation.benchmarks.swe.registry import SweEnvBridge
from agenticml.evaluation.benchmarks.swe.agenticml import run_one_task
from agenticml.evaluation.harness.backends.agenticml_backend import AgenticMLBackend
from tests.fake_tokenizer import FakeTokenizer
from tests.fixtures.generators import HfScriptedGenerator
from tests.fixtures.swe import SweFakeEnv
from tests.wire_fixtures import W_ACTION, W_BELIEF


def _instance() -> dict:
    return {
        "instance_id": "django__django-11099",
        "problem_statement": "fix the url validator bug",
    }


def _backend(responses: list[str]) -> AgenticMLBackend:
    return AgenticMLBackend(
        tokenizer=FakeTokenizer(),
        generator=HfScriptedGenerator(responses),
    )


def test_run_agenticml_swe_bash_then_submit():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@\n"
    env = SweFakeEnv(
        [{"output": "foo.py\n", "returncode": 0, "exception_info": ""}],
        submit_patch=patch,
    )
    backend = _backend(
        [
            f'{W_ACTION}{{"tool":"bash","command":"ls"}}',
            f'{W_ACTION}{{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}}',
        ]
    )
    result = run_agenticml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=5)
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


def test_run_agenticml_swe_no_action_stops():
    backend = _backend([f"{W_BELIEF}thinking only"])
    env = SweFakeEnv()
    result = run_agenticml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=3)
    assert result.stopped_on == "no_action"
    assert env.commands == []


def test_run_one_task_with_injected_env():
    patch = "--- a/x.py\n"
    env = SweFakeEnv(submit_patch=patch)
    backend = _backend(
        [f'{W_ACTION}{{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}}']
    )
    ctx = RunContext(model_id="org/model", format="agenticml")
    row = run_one_task(backend, _instance(), ctx, env=env, max_iterations=3)
    assert row["format"] == "agenticml"
    assert row["success"] is True
    assert row["model_patch"] == patch
    assert row["instance_id"] == "django__django-11099"


def test_run_agenticml_swe_stops_on_repeated_command():
    same = f'{W_ACTION}{{"tool":"bash","command":"git fetch origin"}}'
    backend = _backend([same, same, same])
    env = SweFakeEnv([{"output": "fatal\n", "returncode": 128, "exception_info": ""}])
    result = run_agenticml_swe(backend, SweEnvBridge(env), _instance(), max_iterations=10)
    assert result.stopped_on == "repeated_command"
    assert result.iterations == 3
    assert env.commands == ["git fetch origin", "git fetch origin"]


def test_run_agenticml_swe_records_token_stats():
    backend = _backend([f'{W_ACTION}{{"tool":"bash","command":"pwd"}}'])
    env = SweFakeEnv([{"output": "/testbed\n", "returncode": 0, "exception_info": ""}])
    result = run_agenticml_swe(
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
