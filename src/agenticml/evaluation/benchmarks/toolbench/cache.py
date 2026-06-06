"""cache-only tool execution for toolbench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from telos.evaluation.benchmarks.toolbench.common import (
    change_name,
    contain,
    ensure_toolbench_on_path,
    get_white_list,
    response_cache_path,
    response_examples_path,
    standardize,
    toolenv_path,
)

_FINISH_DESCRIPTION = (
    'If you believe that you have obtained a result that can answer the task, please call '
    'this function to provide the final answer. Alternatively, if you recognize that you are '
    'unable to proceed with the task in the current state, call this function to restart. '
    'Remember: you must ALWAYS call this function at the end of your attempt, and the only part '
    'that will be shown to the user is the final answer, so it should contain sufficient information.'
)


def _finish_function() -> dict[str, Any]:
    return {
        "name": "Finish",
        "description": _FINISH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "return_type": {
                    "type": "string",
                    "enum": ["give_answer", "give_up_and_restart"],
                },
                "final_answer": {
                    "type": "string",
                    "description": (
                        'The final answer you want to give the user. You should have this field '
                        'if "return_type"=="give_answer"'
                    ),
                },
            },
            "required": ["return_type"],
        },
    }


def fetch_api_json(query_json: dict[str, Any], tool_root: Path) -> dict[str, Any]:
    data_dict: dict[str, Any] = {"api_list": []}
    for item in query_json.get("api_list") or []:
        cate_name = item["category_name"]
        tool_name = standardize(item["tool_name"])
        api_name = change_name(standardize(item["api_name"]))
        tool_json = json.loads((tool_root / cate_name / f"{tool_name}.json").read_text())
        append_flag = False
        for api_dict in tool_json["api_list"]:
            pure_api_name = change_name(standardize(api_dict["name"]))
            if pure_api_name != api_name:
                continue
            data_dict["api_list"].append(
                {
                    "category_name": cate_name,
                    "api_name": api_dict["name"],
                    "api_description": api_dict.get("description", ""),
                    "required_parameters": api_dict.get("required_parameters", []),
                    "optional_parameters": api_dict.get("optional_parameters", []),
                    "tool_name": tool_json["tool_name"],
                }
            )
            append_flag = True
            break
        if not append_flag:
            raise KeyError(f"api {api_name!r} not found for tool {tool_name!r} in {cate_name!r}")
    return data_dict


def api_json_to_openai_json(api_json: dict[str, Any], standard_tool_name: str) -> tuple[dict[str, Any], str, str]:
    description_max_length = 256
    map_type = {"NUMBER": "integer", "STRING": "string", "BOOLEAN": "boolean"}
    template: dict[str, Any] = {
        "name": "",
        "description": "",
        "parameters": {"type": "object", "properties": {}, "required": [], "optional": []},
    }
    pure_api_name = change_name(standardize(api_json["api_name"]))
    template["name"] = (pure_api_name + f"_for_{standard_tool_name}")[-64:]
    template["description"] = (
        f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
    )
    if api_json.get("api_description", "").strip():
        truncated = api_json["api_description"].strip().replace(api_json["api_name"], template["name"])[
            :description_max_length
        ]
        template["description"] += f' The description of this function is: "{truncated}"'

    for para in api_json.get("required_parameters") or []:
        name = change_name(standardize(para["name"]))
        param_type = map_type.get(para.get("type"), "string")
        default_value = para.get("default")
        if len(str(default_value)) != 0:
            prompt: dict[str, Any] = {
                "type": param_type,
                "description": para["description"][:description_max_length],
                "example_value": default_value,
            }
        else:
            prompt = {"type": param_type, "description": para["description"][:description_max_length]}
        template["parameters"]["properties"][name] = prompt
        template["parameters"]["required"].append(name)

    for para in api_json.get("optional_parameters") or []:
        name = change_name(standardize(para["name"]))
        param_type = map_type.get(para.get("type"), "string")
        default_value = para.get("default")
        if len(str(default_value)) != 0:
            prompt = {
                "type": param_type,
                "description": para["description"][:description_max_length],
                "example_value": default_value,
            }
        else:
            prompt = {"type": param_type, "description": para["description"][:description_max_length]}
        template["parameters"]["properties"][name] = prompt
        template["parameters"]["optional"].append(name)

    return template, api_json["category_name"], pure_api_name


def build_tool_descriptions(data_dict: dict[str, Any], tool_root: Path) -> list[list[str]]:
    white_list = get_white_list(tool_root)
    origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
    tool_des = contain(origin_tool_names, white_list)
    if not isinstance(tool_des, list):
        raise ValueError("task api_list references tools missing from toolenv")
    return [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]


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


def _handle_finish(action_input: str) -> tuple[str, int]:
    try:
        json_data = json.loads(action_input, strict=False)
    except json.JSONDecodeError:
        json_data = {}
    if '"return_type": "' in action_input:
        if '"return_type": "give_answer"' in action_input:
            return_type = "give_answer"
        elif '"return_type": "give_up_and_restart"' in action_input:
            return_type = "give_up_and_restart"
        else:
            start = action_input.find('"return_type": "') + len('"return_type": "')
            return_type = action_input[start : action_input.find('",', start)]
        json_data["return_type"] = return_type
    if '"final_answer": "' in action_input:
        start = action_input.find('"final_answer": "') + len('"final_answer": "')
        json_data["final_answer"] = action_input[start:]
    if "return_type" not in json_data:
        return '{error:"must have "return_type""}', 2
    if json_data["return_type"] == "give_up_and_restart":
        return '{"response":"chose to give up and restart"}', 4
    if json_data["return_type"] == "give_answer":
        if "final_answer" not in json_data:
            return '{error:"must have "final_answer""}', 2
        return '{"response":"successfully giving the final answer."}', 3
    return '{error:""return_type" is not a valid choice"}', 2


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
    """run one tool call via local api.py; never hits the upstream rapidapi proxy."""
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


class CachedToolEnv:
    """cache-only toolbench env; mirrors upstream rapidapi_wrapper step/status codes."""

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
        from telos.evaluation.benchmarks.toolbench.subset import default_data_root

        self.data_root = data_root or default_data_root()
        self.tool_root = toolenv_path(self.data_root)
        self.observ_compress_method = observ_compress_method
        self.max_observation_length = max_observation_length
        self.rapidapi_key = rapidapi_key
        self.use_cache = use_cache

        data_dict = fetch_api_json(entry, self.tool_root)
        tool_descriptions = build_tool_descriptions(data_dict, self.tool_root)
        self.input_description = entry["query"]
        self.functions: list[dict[str, Any]] = []
        self.api_name_reflect: dict[str, str] = {}
        self.tool_names: list[str] = []
        self.cate_names: list[str] = []

        for k, api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[k][0]
            openai_fn, cate_name, pure_api_name = api_json_to_openai_json(api_json, standard_tool_name)
            self.functions.append(openai_fn)
            self.api_name_reflect[openai_fn["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)

        self.functions.append(_finish_function())
        self.success = 0
        self.task_description = self._build_task_description(tool_descriptions)

    def _build_task_description(self, tool_descriptions: list[list[str]]) -> str:
        text = (
            "You should use functions to help handle the real time user querys. Remember:\n"
            '1.ALWAYS call "Finish" function at the end of the task. And the final answer should '
            "contain enough information to show to the user,If you can't handle the task, or you "
            'find that function calls always fail(the function is not valid now), use function '
            "Finish->give_up_and_restart.\n"
            "2.Do not use origin tool names, use only subfunctions' names.\n"
            "You have access of the following tools:\n"
        )
        unduplicated: dict[str, str] = {name: desc for name, desc in tool_descriptions}
        for k, (standard_tool_name, tool_des) in enumerate(unduplicated.items()):
            striped = tool_des[:512].replace("\n", "").strip() or "None"
            text += f"{k + 1}.{standard_tool_name}: {striped}\n"
        return text

    def check_success(self) -> int:
        return self.success

    def step(self, action_name: str = "", action_input: str = "") -> tuple[str, int]:
        obs, code = self._step(action_name, action_input)
        if len(obs) > self.max_observation_length:
            obs = obs[: self.max_observation_length] + "..."
        return obs, code

    def _step(self, action_name: str = "", action_input: str = "") -> tuple[str, int]:
        if action_name == "Finish":
            obs, code = _handle_finish(action_input)
            if code == 3:
                self.success = 1
            return obs, code

        for k, function in enumerate(self.functions):
            if function["name"].endswith(action_name):
                pure_api_name = self.api_name_reflect[function["name"]]
                response = call_tool_local(
                    category=self.cate_names[k],
                    tool_name=self.tool_names[k],
                    api_name=pure_api_name,
                    tool_input=action_input,
                    data_root=self.data_root,
                    observ_compress_method=self.observ_compress_method,
                    rapidapi_key=self.rapidapi_key,
                    use_cache=self.use_cache,
                )
                return json.dumps(response), _status_code_from_response(response)

        return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1


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
    """one-shot step helper for tests and the toolbench driver."""
    env = CachedToolEnv(
        entry,
        data_root=data_root,
        observ_compress_method=observ_compress_method,
        max_observation_length=max_observation_length,
        rapidapi_key=rapidapi_key,
        use_cache=use_cache,
    )
    return env.step(action_name, action_input)
