from pathlib import Path

from agenticml.evaluation.benchmarks.toolbench.cache import (
    CachedToolEnv,
    _read_response_cache,
    _write_response_cache,
    call_tool_local,
    execute_tool_call,
)


def test_finish_give_answer(data_root: tuple[Path, dict]):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    obs, code = env.step(
        "Finish",
        '{"return_type": "give_answer", "final_answer": "done"}',
    )
    assert code == 3
    assert "successfully" in obs
    assert env.check_success() == 1


def test_finish_give_up(data_root: tuple[Path, dict]):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    obs, code = env.step("Finish", '{"return_type": "give_up_and_restart"}')
    assert code == 4
    assert "give up" in obs


def test_unknown_action_returns_code_1(data_root: tuple[Path, dict]):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    obs, code = env.step("missing_fn", "{}")
    assert code == 1
    assert "No such function name" in obs


def test_functions_include_finish_and_subfunction(data_root: tuple[Path, dict]):
    root, entry = data_root
    env = CachedToolEnv(entry, data_root=root)
    names = [f["name"] for f in env.functions]
    assert "Finish" in names
    assert any("hello" in n for n in names)


def test_response_cache_roundtrip(data_root: tuple[Path, dict]):
    root, _ = data_root
    _write_response_cache(
        root,
        "Demo",
        "demo_tool",
        "hello",
        "{}",
        {"error": "", "response": "cached"},
    )
    cached = _read_response_cache(root, "Demo", "demo_tool", "hello", "{}")
    assert cached == {"error": "", "response": "cached"}


def test_call_tool_local_uses_cache_without_upstream(monkeypatch, data_root: tuple[Path, dict]):
    root, _ = data_root
    def fail_import(*_args, **_kwargs):
        raise AssertionError("should not import upstream when cache hit")

    monkeypatch.setattr(
        "agenticml.evaluation.benchmarks.toolbench.cache.ensure_toolbench_on_path",
        fail_import,
    )
    _write_response_cache(
        root,
        "Demo",
        "demo_tool",
        "hello",
        "{}",
        {"error": "", "response": "from cache"},
    )
    out = call_tool_local(
        category="Demo",
        tool_name="demo_tool",
        api_name="hello",
        tool_input="{}",
        data_root=root,
    )
    assert out["response"] == "from cache"


def test_execute_tool_call_delegates_to_env(data_root: tuple[Path, dict]):
    root, entry = data_root
    obs, code = execute_tool_call(
        entry,
        "Finish",
        '{"return_type": "give_answer", "final_answer": "ok"}',
        data_root=root,
    )
    assert code == 3
