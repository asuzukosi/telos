"""map mini-swe bash environment to agenticml ToolRegistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agenticml.evaluation.benchmarks.swe.common import BASH_TOOL_SCHEMA, ensure_miniswe_on_path, format_command_output
from agenticml.runtime.tools import Tool, ToolError, ToolRegistry


class BashEnvironment(Protocol):
    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]: ...


@dataclass
class SweEnvBridge:
    """wrap mini-swe Environment.execute and capture SWE-bench submission."""

    env: BashEnvironment
    submission: str | None = field(default=None, init=False)
    exit_status: str | None = field(default=None, init=False)

    def run_bash(self, command: str) -> str:
        ensure_miniswe_on_path()
        from minisweagent.exceptions import Submitted

        try:
            output = self.env.execute({"command": command})
        except Submitted as exc:
            msg = exc.messages[0] if exc.messages else {}
            extra = msg.get("extra", {})
            self.submission = str(extra.get("submission") or msg.get("content") or "")
            self.exit_status = str(extra.get("exit_status") or "Submitted")
            return self.submission
        return format_command_output(output)


def registry_from_bridge(bridge: SweEnvBridge) -> ToolRegistry:
    reg = ToolRegistry()

    def bash(command: str) -> str:
        cmd = (command or "").strip()
        if not cmd:
            raise ToolError("bash requires non-empty command")
        return bridge.run_bash(cmd)

    reg.register(Tool("bash", bash, BASH_TOOL_SCHEMA))
    return reg


def registry_from_env(env: BashEnvironment) -> tuple[ToolRegistry, SweEnvBridge]:
    bridge = SweEnvBridge(env)
    return registry_from_bridge(bridge), bridge
