import pytest

from agenticml.evaluation.benchmarks.swe.common import format_command_output
from agenticml.evaluation.benchmarks.swe.prelude import instance_mission, instance_to_prelude
from agenticml.evaluation.benchmarks.swe.registry import SweEnvBridge, registry_from_bridge
from agenticml.runtime.tools import ToolError
from tests.fixtures.swe import SweFakeEnv


def test_instance_to_prelude():
    instance = {
        "instance_id": "django__django-11099",
        "problem_statement": "fix the bug in urls",
    }
    frames = instance_to_prelude(instance)
    assert frames[0]["type"] == "goal"
    assert "fix the bug in urls" in frames[1]["content"]
    assert "<pr_description>" in frames[1]["content"]
    assert "no git remotes" in frames[1]["content"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in frames[1]["content"]


def test_instance_mission_requires_problem_statement():
    with pytest.raises(ValueError):
        instance_mission({})


def test_format_command_output_truncates_long_text():
    text = "x" * 20_000
    rendered = format_command_output({"output": text, "returncode": 0, "exception_info": ""})
    assert "<output_head>" in rendered
    assert "<output_tail>" in rendered
    assert "10000 characters elided" in rendered


def test_format_command_output_includes_exception():
    rendered = format_command_output(
        {"output": "fail", "returncode": 1, "exception_info": "timeout"}
    )
    assert "<exception>timeout</exception>" in rendered
    assert "<returncode>1</returncode>" in rendered


def test_registry_bash_executes_command():
    env = SweFakeEnv([{"output": "hello\n", "returncode": 0, "exception_info": ""}])
    bridge = SweEnvBridge(env)
    reg = registry_from_bridge(bridge)
    out = reg.call("bash", {"command": "echo hello"})
    assert "<returncode>0</returncode>" in out
    assert "hello" in out
    assert env.commands == ["echo hello"]


def test_registry_bash_rejects_empty_command():
    bridge = SweEnvBridge(SweFakeEnv())
    reg = registry_from_bridge(bridge)
    with pytest.raises(ToolError, match="non-empty"):
        reg.call("bash", {"command": "  "})


def test_registry_bash_captures_submission():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@\n"
    env = SweFakeEnv(submit_patch=patch)
    bridge = SweEnvBridge(env)
    reg = registry_from_bridge(bridge)
    out = reg.call("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"})
    assert out == patch
    assert bridge.submission == patch
    assert bridge.exit_status == "Submitted"


def test_pred_entry_and_result_row():
    from agenticml.evaluation.benchmarks.swe.io import pred_entry, result_row
    from agenticml.evaluation.benchmarks.swe.loop import SweRunResult
    from agenticml.trajectory import Trajectory

    run = SweRunResult(
        instance_id="django__django-11099",
        trajectory=Trajectory([]),
        stopped_on="submitted",
        iterations=3,
        model_patch="--- a/x.py\n",
        exit_status="Submitted",
    )
    assert pred_entry(run, model_id="org/model")["model_patch"] == "--- a/x.py\n"
    row = result_row(run, model_id="org/model")
    assert row["success"] is True
    assert row["format"] == "agenticml"
