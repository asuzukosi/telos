import pytest

from telos.evaluation.benchmarks.swe.common import format_command_output
from telos.evaluation.benchmarks.swe.prelude import instance_mission, instance_to_prelude
from telos.evaluation.benchmarks.swe.registry import SweEnvBridge, registry_from_bridge
from telos.runtime.tools import ToolError


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


def test_registry_bash_executes_command():
    env = _FakeEnv([{"output": "hello\n", "returncode": 0, "exception_info": ""}])
    bridge = SweEnvBridge(env)
    reg = registry_from_bridge(bridge)
    out = reg.call("bash", {"command": "echo hello"})
    assert "<returncode>0</returncode>" in out
    assert "hello" in out
    assert env.commands == ["echo hello"]


def test_registry_bash_rejects_empty_command():
    bridge = SweEnvBridge(_FakeEnv())
    reg = registry_from_bridge(bridge)
    with pytest.raises(ToolError, match="non-empty"):
        reg.call("bash", {"command": "  "})


def test_registry_bash_captures_submission():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@\n"
    env = _FakeEnv(submit_patch=patch)
    bridge = SweEnvBridge(env)
    reg = registry_from_bridge(bridge)
    out = reg.call("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"})
    assert out == patch
    assert bridge.submission == patch
    assert bridge.exit_status == "Submitted"


def test_pred_entry_and_result_row():
    from telos.evaluation.benchmarks.swe.io import pred_entry, result_row
    from telos.evaluation.benchmarks.swe.loop import SweRunResult
    from telos.trajectory import Trajectory

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
    assert row["format"] == "telos"
