"""map cachedtoolenv functions to telos ToolRegistry."""

from __future__ import annotations

import json
from typing import Any

from telos.evaluation.benchmarks.toolbench.cache import CachedToolEnv
from telos.runtime.tools import Tool, ToolError, ToolRegistry


def _action_input(args: dict[str, Any]) -> str:
    return json.dumps(args)


def registry_from_env(env: CachedToolEnv) -> ToolRegistry:
    reg = ToolRegistry()
    for spec in env.functions:
        name = str(spec["name"])
        if name == "Finish":

            def finish(**kwargs: Any) -> str:
                obs, code = env.step("Finish", _action_input(kwargs))
                if code == 3:
                    return str(kwargs.get("final_answer") or obs)
                if code == 4:
                    raise ToolError("give_up_and_restart")
                raise ToolError(obs)

            reg.register(Tool("Finish", finish, spec))
            continue

        reg.register(Tool(name, _make_api_fn(env, name), spec))
    return reg


def _make_api_fn(env: CachedToolEnv, action_name: str):
    def api_fn(**kwargs: Any) -> str:
        obs, _code = env.step(action_name, _action_input(kwargs))
        return obs

    return api_fn
