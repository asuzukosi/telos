"""shared bfcl plumbing: schemas, results io, multi-turn loop, handler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

from telos.constants import TERMINAL_TOOLS
from telos.evaluation.benchmarks.bfcl.subset import ensure_bfcl_on_path
from telos.evaluation.benchmarks.common import model_dir_name

TState = TypeVar("TState")

RETRY_INJECT_MSG = "Error: simulated tool failure. Retry with a corrected call."


def execution_result_frame(execution_result: str) -> dict[str, Any]:
    """map bfcl mock-api json to a telos result frame (ok=0 when the payload has error)."""
    text = (execution_result or "").strip()
    if not text:
        return {"ok": 1, "value": text}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("error") is not None:
            return {"ok": 0, "value": text}
    except json.JSONDecodeError:
        pass
    return {"ok": 1, "value": text}


def functions_to_schemas(functions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not functions:
        return []
    out: list[dict[str, Any]] = []
    for fn in functions:
        name = fn.get("name")
        if not name:
            continue
        raw_params = fn.get("parameters") or {}
        if isinstance(raw_params, dict) and "properties" in raw_params:
            parameters = dict(raw_params)
        else:
            parameters = {"type": "object", "properties": {}, "required": []}
        if parameters.get("type") in ("dict", None):
            parameters["type"] = "object"
        out.append(
            {
                "name": str(name),
                "description": str(fn.get("description") or ""),
                "parameters": parameters,
            }
        )
    return out


def entry_tool_schemas(entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw = entry.get("function")
    if not isinstance(raw, list):
        return []
    return functions_to_schemas(raw)


def actions_to_result(actions: list[dict], entry_id: str) -> str:
    if "irrelevance" in entry_id:
        tool_actions = [
            a for a in actions if a.get("tool") and a.get("tool") not in TERMINAL_TOOLS
        ]
        if not tool_actions:
            return "[]"
    calls: list[dict] = []
    for action in actions:
        tool = action.get("tool")
        if not tool or tool in TERMINAL_TOOLS:
            continue
        params = {k: v for k, v in action.items() if k != "tool"}
        calls.append({"name": str(tool), "parameters": params})
    if not calls:
        return "[]"
    if len(calls) == 1:
        return json.dumps(calls[0])
    return json.dumps(calls)


def encode_result(
    entry_id: str,
    *,
    actions: Optional[list[dict]] = None,
    raw: str = "",
    call: Optional[dict] = None,
) -> str:
    if call is not None:
        name = call.get("name") or call.get("tool")
        if not name:
            return "[]"
        raw_args = call.get("arguments", call.get("parameters", {}))
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        actions = [{"tool": str(name), **args}]
    if actions is not None:
        return actions_to_result(actions, entry_id)
    if "irrelevance" in entry_id:
        return "[]"
    from telos.evaluation.harness.chatml_fc import strip_chat_generation_tokens

    text = strip_chat_generation_tokens(raw)
    return "[]" if not text else text


def result_row(
    entry_id: str,
    result: Any,
    *,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    latency: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "result": result,
        "input_token_count": prompt_tokens,
        "output_token_count": generated_tokens,
        "latency": latency,
    }


@dataclass
class BFCLStep:
    result: str
    prompt_tokens: int
    generated_tokens: int
    inference_sec: float
    stop_reason: str = ""


def _decode_name_parameters_result(result: str) -> list[dict]:
    from telos.evaluation.harness.chatml_fc import strip_chat_generation_tokens

    text = strip_chat_generation_tokens(result)
    if not text or text == "[]":
        return []
    if ";" in text:
        calls = [json.loads(part.strip()) for part in text.split(";") if part.strip()]
    elif text.startswith("["):
        parsed = json.loads(text)
        calls = parsed if isinstance(parsed, list) else [parsed]
    else:
        parsed = json.loads(text)
        calls = [parsed] if isinstance(parsed, dict) else parsed
    out: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("tool")
        if not name:
            continue
        params = call.get("parameters", call.get("arguments", {}))
        if isinstance(params, str):
            params = json.loads(params)
        out.append({str(name): params})
    return out


def _decode_bfcl_ast(result: Any, language: Any, has_tool_call_tag: bool) -> Any:
    from bfcl_eval.constants.enums import ReturnFormat
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting

    lang = language or ReturnFormat.PYTHON

    def decode_one(raw: Any) -> Any:
        if raw == "[]" or not raw:
            return []
        if isinstance(raw, str):
            try:
                return _decode_name_parameters_result(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return default_decode_ast_prompting(raw, lang, has_tool_call_tag=has_tool_call_tag)

    if isinstance(result, list):
        out = []
        for turn in result:
            if isinstance(turn, list):
                out.append([decode_one(step) for step in turn])
            else:
                out.append(decode_one(turn))
        return out
    return decode_one(result)


def count_retry_steps(result: Any, entry_id: str) -> int:
    ensure_bfcl_on_path()
    from bfcl_eval.utils import contain_multi_turn_interaction

    if not contain_multi_turn_interaction(entry_id):
        return 0
    if not isinstance(result, list):
        return 0
    return sum(
        max(0, len(turn) - 1)
        for turn in result
        if isinstance(turn, list)
    )


def retry_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = [count_retry_steps(r.get("result"), str(r.get("id", ""))) for r in rows]
    n = len(counts)
    if n == 0:
        return {"avg_retry_count": 0.0, "total_retry_count": 0.0}
    return {
        "avg_retry_count": sum(counts) / n,
        "total_retry_count": float(sum(counts)),
    }


def infer_multi_turn(
    entry: dict,
    model_slug: str,
    *,
    init_state: Callable[[dict], TState],
    begin_turn: Callable[[TState, dict, int], TState],
    step: Callable[[TState, dict], tuple[BFCLStep, TState]],
    feed_results: Callable[[TState, list[str], list[str]], TState],
    max_new_tokens: int = 512,
    max_steps_per_turn: int = 12,
    inject_retry_failure: bool = False,
) -> dict[str, Any]:
    ensure_bfcl_on_path()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )
    from bfcl_eval.model_handler.utils import decoded_output_to_execution_list

    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]
    test_id = entry["id"]
    test_category = test_id.rsplit("_", 1)[0]
    long_context = "long_context" in test_category
    handler = ResultHandler.from_model_id(model_slug)

    state = init_state(entry)
    all_turn_results: list[list[str]] = []
    total_prompt = 0
    total_gen = 0
    total_latency = 0.0

    for turn_idx in range(len(entry["question"])):
        state = begin_turn(state, entry, turn_idx)
        turn_responses: list[str] = []
        turn_injected = False
        for _ in range(max_steps_per_turn):
            st, state = step(state, entry)
            total_prompt += st.prompt_tokens
            total_gen += st.generated_tokens
            total_latency += st.inference_sec
            turn_responses.append(st.result)

            if st.result == "[]" or st.stop_reason.startswith("parse_error"):
                break
            try:
                call_strings = decoded_output_to_execution_list(
                    handler.decode_ast(st.result)
                )
            except Exception:
                break
            if is_empty_execute_response(call_strings):
                break
            execution_results, _ = execute_multi_turn_func_call(
                call_strings,
                initial_config,
                involved_classes,
                model_slug,
                test_id,
                long_context=long_context,
                is_evaL_run=False,
            )
            feedback = list(execution_results)
            if inject_retry_failure and not turn_injected and feedback:
                feedback[0] = RETRY_INJECT_MSG
                turn_injected = True
            state = feed_results(state, call_strings, feedback)

        all_turn_results.append(turn_responses)

    return result_row(
        test_id,
        all_turn_results,
        prompt_tokens=total_prompt,
        generated_tokens=total_gen,
        latency=total_latency,
    )


def route_infer_entry(
    backend: Any,
    entry: dict,
    *,
    model_id: str,
    infer_single: Callable[..., dict[str, Any]],
    infer_multi: Callable[..., dict[str, Any]],
    max_new_tokens: int = 512,
    inject_retry_failure: bool = False,
) -> dict[str, Any]:
    ensure_bfcl_on_path()
    from bfcl_eval.utils import contain_multi_turn_interaction, extract_test_category_from_id

    cat = extract_test_category_from_id(entry["id"])
    slug = model_dir_name(model_id)
    if contain_multi_turn_interaction(cat):
        return infer_multi(
            backend,
            entry,
            model_slug=slug,
            max_new_tokens=max_new_tokens,
            inject_retry_failure=inject_retry_failure,
        )
    return infer_single(backend, entry, max_new_tokens=max_new_tokens)


def write_results(
    result_dir: Path,
    model_id: str,
    rows: list[dict[str, Any]],
) -> None:
    ensure_bfcl_on_path()
    from bfcl_eval.constants.eval_config import VERSION_PREFIX
    from bfcl_eval.utils import (
        extract_test_category_from_id,
        get_directory_structure_by_id,
        load_file,
        make_json_serializable,
        sort_key,
    )

    model_root = result_dir / model_dir_name(model_id)
    file_entries: dict[Path, list[dict]] = {}

    for row in rows:
        serialized = make_json_serializable(row)
        if not isinstance(serialized, dict):
            continue
        item = cast(dict[str, Any], serialized)
        eid = str(item["id"])
        cat = extract_test_category_from_id(eid)
        group = get_directory_structure_by_id(eid)
        path = model_root / group / f"{VERSION_PREFIX}_{cat}_result.json"
        file_entries.setdefault(path, []).append(item)

    for path, new_rows in file_entries.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        by_id: dict[str, dict[str, Any]] = {}
        if path.exists():
            for e in load_file(path):
                by_id[str(e["id"])] = cast(dict[str, Any], e)
        for e in new_rows:
            by_id[str(e["id"])] = e
        merged = sorted(by_id.values(), key=sort_key)
        with path.open("w") as f:
            for e in merged:
                f.write(json.dumps(e) + "\n")


@dataclass
class ResultHandler:
    """minimal handler surface for bfcl evaluate."""

    model_id: str

    @property
    def registry_dir_name(self) -> str:
        return model_dir_name(self.model_id)

    @classmethod
    def from_model_id(cls, model_id: str) -> ResultHandler:
        return cls(model_id=model_id)

    def decode_ast(
        self,
        result: Any,
        language: Any = None,
        has_tool_call_tag: bool = False,
    ) -> Any:
        return _decode_bfcl_ast(result, language, has_tool_call_tag)

    def decode_execute(self, result: Any, has_tool_call_tag: bool = False) -> Any:
        from bfcl_eval.constants.enums import ReturnFormat
        from bfcl_eval.model_handler.utils import decoded_output_to_execution_list

        if isinstance(result, list):
            return [
                decoded_output_to_execution_list(
                    self.decode_ast(turn, ReturnFormat.PYTHON, has_tool_call_tag)
                )
                if turn != "[]"
                else []
                for turn in result
            ]
        decoded = self.decode_ast(result, ReturnFormat.PYTHON, has_tool_call_tag)
        return decoded_output_to_execution_list(decoded)

    def write(self, result, result_dir: Path, update_mode: bool = False) -> None:
        del update_mode
        rows = result if isinstance(result, list) else [result]
        write_results(result_dir, self.model_id, rows)
