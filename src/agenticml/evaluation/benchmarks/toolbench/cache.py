"""cache-only tool execution for toolbench (subclass upstream rapidapi_wrapper)."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.toolbench.common import (
    ensure_toolbench_on_path,
    response_cache_path,
    response_examples_path,
    toolenv_path,
)
from agenticml.evaluation.benchmarks.toolbench.subset import default_data_root


def _cache_key(category: str, tool_name: str, api_name: str, tool_input: str) -> str:
    payload = json.dumps(
        {"category": category, "tool_name": tool_name, "api_name": api_name, "tool_input": tool_input},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_file(
    data_root: Path,
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: str,
) -> Path:
    key = _cache_key(category, tool_name, api_name, tool_input)
    return response_cache_path(data_root) / category / tool_name / api_name / f"{key}.json"


def _read_response_cache(
    data_root: Path,
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: str,
) -> dict[str, str] | None:
    path = _cache_file(data_root, category, tool_name, api_name, tool_input)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return None
    return {"error": str(raw.get("error", "")), "response": str(raw.get("response", ""))}


def _write_response_cache(
    data_root: Path,
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: str,
    response: dict[str, str],
) -> None:
    path = _cache_file(data_root, category, tool_name, api_name, tool_input)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response, indent=2))


def _status_code_from_response(response: dict[str, str]) -> int:
    err = response.get("error") or ""
    if err == "API not working error...":
        return 6
    if err == "Unauthorized error...":
        return 7
    if err == "Unsubscribed error...":
        return 8
    if err == "Too many requests error...":
        return 9
    if err in ("Rate limit per minute error...", "Rate limit error..."):
        return 10
    if err == "Message error...":
        return 11
    return 0


def call_tool_local(
    *,
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: str,
    data_root: Path,
    observ_compress_method: str = "truncate",
    rapidapi_key: str = "",
    use_cache: bool = True,
) -> dict[str, str]:
    if use_cache:
        cached = _read_response_cache(data_root, category, tool_name, api_name, tool_input)
        if cached is not None:
            return cached

    ensure_toolbench_on_path()
    from toolbench.inference.server import get_rapidapi_response

    payload = {
        "category": category,
        "tool_name": tool_name,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": observ_compress_method,
        "rapidapi_key": rapidapi_key,
    }
    response = get_rapidapi_response(
        payload,
        api_customization=True,
        tools_root="data.toolenv.tools",
        schema_root=str(response_examples_path(data_root)),
    )
    out = {"error": str(response.get("error", "")), "response": str(response.get("response", ""))}
    if use_cache:
        _write_response_cache(data_root, category, tool_name, api_name, tool_input, out)
    return out


def _toolbench_args(
    data_root: Path,
    *,
    observ_compress_method: str,
    max_observation_length: int,
    rapidapi_key: str,
) -> Namespace:
    return Namespace(
        tool_root_dir=str(toolenv_path(data_root)),
        toolbench_key="",
        rapidapi_key=rapidapi_key,
        use_rapidapi_key=False,
        api_customization=True,
        observ_compress_method=observ_compress_method,
        max_observation_length=max_observation_length,
    )


def _rapidapi_wrapper_cls():
    ensure_toolbench_on_path()
    from toolbench.inference.Downstream_tasks.rapidapi import rapidapi_wrapper

    return rapidapi_wrapper


class CachedRapidapiWrapper:
    """cache-only rapidapi env; delegates setup/finish to upstream, tools to local cache."""

    def __init__(
        self,
        entry: dict[str, Any],
        *,
        data_root: Path | None = None,
        observ_compress_method: str = "truncate",
        max_observation_length: int = 1024,
        rapidapi_key: str = "",
        use_cache: bool = True,
    ) -> None:
        rapidapi_wrapper = _rapidapi_wrapper_cls()
        self.data_root = data_root or default_data_root()
        self.use_cache = use_cache
        self.rapidapi_key = rapidapi_key
        self.observ_compress_method = observ_compress_method

        args = _toolbench_args(
            self.data_root,
            observ_compress_method=observ_compress_method,
            max_observation_length=max_observation_length,
            rapidapi_key=rapidapi_key,
        )
        env = rapidapi_wrapper.__new__(rapidapi_wrapper)
        env.tool_root_dir = args.tool_root_dir
        data_dict = rapidapi_wrapper.fetch_api_json(env, entry)
        tool_descriptions = rapidapi_wrapper.build_tool_description(env, data_dict)
        rapidapi_wrapper.__init__(env, entry, tool_descriptions, None, args, process_id=0)
        self._env = env

    @property
    def functions(self) -> list[dict[str, Any]]:
        return self._env.functions

    @property
    def task_description(self) -> str:
        return self._env.task_description

    @property
    def input_description(self) -> str:
        return self._env.input_description

    def check_success(self) -> int:
        return self._env.check_success()

    def step(self, action_name: str = "", action_input: str = "") -> tuple[str, int]:
        if action_name == "Finish":
            return self._env.step(action_name=action_name, action_input=action_input)
        for k, function in enumerate(self._env.functions):
            if function["name"].endswith(action_name):
                pure_api_name = self._env.api_name_reflect[function["name"]]
                response = call_tool_local(
                    category=self._env.cate_names[k],
                    tool_name=self._env.tool_names[k],
                    api_name=pure_api_name,
                    tool_input=action_input,
                    data_root=self.data_root,
                    observ_compress_method=self.observ_compress_method,
                    rapidapi_key=self.rapidapi_key,
                    use_cache=self.use_cache,
                )
                obs = json.dumps(response)
                if len(obs) > self._env.max_observation_length:
                    obs = obs[: self._env.max_observation_length] + "..."
                return obs, _status_code_from_response(response)
        obs = json.dumps({"error": f"No such function name: {action_name}", "response": ""})
        return obs, 1


CachedToolEnv = CachedRapidapiWrapper


def execute_tool_call(
    entry: dict[str, Any],
    action_name: str,
    action_input: str,
    *,
    data_root: Path | None = None,
    observ_compress_method: str = "truncate",
    max_observation_length: int = 1024,
    rapidapi_key: str = "",
    use_cache: bool = True,
) -> tuple[str, int]:
    env = CachedRapidapiWrapper(
        entry,
        data_root=data_root,
        observ_compress_method=observ_compress_method,
        max_observation_length=max_observation_length,
        rapidapi_key=rapidapi_key,
        use_cache=use_cache,
    )
    return env.step(action_name, action_input)
