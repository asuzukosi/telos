"""shared swe benchmark test fixtures."""

from __future__ import annotations


class SweFakeEnv:
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
