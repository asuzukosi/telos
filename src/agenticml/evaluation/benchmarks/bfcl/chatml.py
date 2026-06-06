"""run bfcl subset through chatml backend."""

from __future__ import annotations

from typing import Any, Optional

from telos.evaluation.benchmarks.bfcl.common import (
    BFCLStep,
    encode_result,
    entry_tool_schemas,
    infer_multi_turn,
    result_row,
    route_infer_entry,
)
from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.harness.backends.chatml_backend import (
    ChatMLBackend,
    _parse,
    _with_tools,
)
from telos.evaluation.harness.backends.common import GenStats
from telos.evaluation.harness.chatml_fc import parse_chatml_fc_call


def entry_turn_messages(entry: dict, turn_idx: int) -> list[dict]:
    turn = [dict(m) for m in entry["question"][turn_idx]]
    if turn_idx == 0:
        from telos.evaluation.benchmarks.bfcl.subset import ensure_bfcl_on_path

        ensure_bfcl_on_path()
        from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

        turn = system_prompt_pre_processing_chat_model(
            turn, entry.get("function") or [], entry["id"]
        )
    return turn


def _chatml_step(
    messages: list[dict],
    entry: dict,
    backend: ChatMLBackend,
    *,
    max_new_tokens: int,
    tools_already_in_messages: bool = False,
) -> tuple[BFCLStep, list[dict]]:
    if tools_already_in_messages:
        prompt_messages = list(messages)
    else:
        prompt_messages = _with_tools(list(messages), entry_tool_schemas(entry))
    stats = GenStats()
    raw = backend._generate(prompt_messages, max_new_tokens, stats)
    call, _, stop = _parse(raw)
    if call is None:
        call = parse_chatml_fc_call(raw)
        if call is not None:
            stop = "tool_call"
    bfcl_result = (
        encode_result(entry["id"], call=call)
        if call is not None
        else encode_result(entry["id"], raw=raw)
    )
    updated = list(messages)
    updated.append({"role": "assistant", "content": raw.strip()})
    return (
        BFCLStep(
            result=bfcl_result,
            prompt_tokens=stats.prompt_tokens,
            generated_tokens=stats.generated_tokens,
            inference_sec=stats.inference_sec,
            stop_reason=stop,
        ),
        updated,
    )


def infer_single_turn(
    backend: ChatMLBackend,
    entry: dict,
    *,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    st, _ = _chatml_step(
        entry_turn_messages(entry, 0),
        entry,
        backend,
        max_new_tokens=max_new_tokens,
        tools_already_in_messages=True,
    )
    return result_row(
        entry["id"],
        st.result,
        prompt_tokens=st.prompt_tokens,
        generated_tokens=st.generated_tokens,
        latency=st.inference_sec,
    )


def infer_multi_turn_chatml(
    backend: ChatMLBackend,
    entry: dict,
    *,
    model_slug: str,
    max_new_tokens: int = 512,
    max_steps_per_turn: int = 12,
    inject_retry_failure: bool = False,
) -> dict[str, Any]:
    def init_state(_e: dict) -> list[dict]:
        return []

    def begin_turn(state: list[dict], e: dict, turn_idx: int) -> list[dict]:
        if turn_idx == 0:
            return entry_turn_messages(e, 0)
        return state + [dict(m) for m in e["question"][turn_idx]]

    def step(state: list[dict], e: dict) -> tuple[BFCLStep, list[dict]]:
        return _chatml_step(
            state,
            e,
            backend,
            max_new_tokens=max_new_tokens,
            tools_already_in_messages=True,
        )

    def feed_results(
        state: list[dict],
        call_strings: list[str],
        execution_results: list[str],
    ) -> list[dict]:
        for call_str, er in zip(call_strings, execution_results):
            state.append({"role": "tool", "name": call_str, "content": er})
        return state

    return infer_multi_turn(
        entry,
        model_slug,
        init_state=init_state,
        begin_turn=begin_turn,
        step=step,
        feed_results=feed_results,
        max_new_tokens=max_new_tokens,
        max_steps_per_turn=max_steps_per_turn,
        inject_retry_failure=inject_retry_failure,
    )


def run_one_task(
    backend: ChatMLBackend,
    entry: dict,
    ctx: RunContext,
) -> dict[str, Any]:
    return route_infer_entry(
        backend,
        entry,
        model_id=ctx.model_id,
        infer_single=infer_single_turn,
        infer_multi=infer_multi_turn_chatml,
        max_new_tokens=ctx.max_new_tokens,
        inject_retry_failure=ctx.inject_retry_failure,
    )
