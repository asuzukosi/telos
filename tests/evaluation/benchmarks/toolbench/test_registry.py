import json

import pytest

from agenticml.evaluation.benchmarks.toolbench.cache import CachedToolEnv
from agenticml.evaluation.benchmarks.toolbench.registry import registry_from_env
from agenticml.runtime.tools import ToolError


def _entry() -> dict:
    return {
        "query": "say hi",
        "query_id": "1",
        "api_list": [
            {"category_name": "Demo", "tool_name": "demo_tool", "api_name": "hello"},
        ],
    }


def test_registry_finish_success(data_root):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    reg = registry_from_env(env)
    out = reg.call("Finish", {"return_type": "give_answer", "final_answer": "done"})
    assert out == "done"
    assert env.check_success() == 1


def test_registry_finish_give_up(data_root):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    reg = registry_from_env(env)
    with pytest.raises(ToolError, match="give_up"):
        reg.call("Finish", {"return_type": "give_up_and_restart"})


def test_registry_passes_full_function_name_to_env(data_root):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    seen: list[str] = []

    def fake_step(action_name: str = "", action_input: str = "") -> tuple[str, int]:
        del action_input
        seen.append(action_name)
        return '{"response": "ok"}', 0

    env.step = fake_step  # type: ignore[method-assign]
    reg = registry_from_env(env)
    fn_name = next(n for n in reg.list_names() if n != "Finish")
    reg.call(fn_name, {})
    assert seen == [fn_name]


def test_registry_api_delegates_to_env(data_root):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)

    def fake_step(action_name: str = "", action_input: str = "") -> tuple[str, int]:
        del action_input
        if action_name == "Finish":
            return '{"response":"finish"}', 3
        return '{"error": "", "response": "tool ok"}', 0

    env.step = fake_step  # type: ignore[method-assign]
    reg = registry_from_env(env)
    fn_name = next(n for n in reg.list_names() if n != "Finish")
    raw = reg.call(fn_name, {})
    assert json.loads(raw)["response"] == "tool ok"
