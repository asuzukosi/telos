"""run bfcl subset through agenticml backend."""

from __future__ import annotations

import json
from typing import Any

from agenticml.evaluation.benchmarks.bfcl.common import (
    BFCLStep,
    encode_result,
    entry_tool_schemas,
    execution_result_frame,
    tool_name_from_call_string,
    infer_multi_turn,
    result_row,
    route_infer_entry,
)
from agenticml.evaluation.benchmarks.suite import RunContext
from agenticml.evaluation.harness.backends.agenticml_backend import AgenticMLBackend
from agenticml.sdk import StepResult, with_tool_obs
from agenticml.trajectory import Trajectory

DEFAULT_GOAL = "You are a helpful assistant that calls functions accurately."


def entry_to_prelude(entry: dict, *, user_content: str) -> list[dict]:
    return [
        {"type": "goal", "content": DEFAULT_GOAL},
        {"type": "mission", "content": user_content},
    ]


def _actions_from_step(step: StepResult | None) -> list[dict]:
    if step is None or not step.new_frames:
        return []
    out: list[dict] = []
    for f in step.new_frames.to_dict():
        if f.get("type") != "action":
            continue
        content = f.get("content") or {}
        if isinstance(content, str):
            content = json.loads(content)
        out.append(dict(content))
    return out


def _agenticml_step(
    traj: Trajectory,
    entry: dict,
    backend: AgenticMLBackend,
    *,
    max_new_tokens: int,
) -> tuple[BFCLStep, Trajectory]:
    out = backend.step(
        with_tool_obs(traj, entry_tool_schemas(entry)),
        max_new_tokens=max_new_tokens,
        strict=False,
    )
    actions = _actions_from_step(out.step)
    result = encode_result(entry["id"], actions=actions)
    new_traj = out.step.trajectory if out.step else traj
    return (
        BFCLStep(
            result=result,
            prompt_tokens=out.prompt_tokens,
            generated_tokens=out.generated_tokens,
            inference_sec=out.inference_sec,
        ),
        new_traj,
    )


def infer_single_turn(
    backend: AgenticMLBackend,
    entry: dict,
    *,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    user_turn = entry["question"][0]
    content = user_turn[0]["content"] if user_turn else ""
    st, _ = _agenticml_step(
        Trajectory(entry_to_prelude(entry, user_content=content)),
        entry,
        backend,
        max_new_tokens=max_new_tokens,
    )
    return result_row(
        entry["id"],
        st.result,
        prompt_tokens=st.prompt_tokens,
        generated_tokens=st.generated_tokens,
        latency=st.inference_sec,
    )


def infer_multi_turn_agenticml(
    backend: AgenticMLBackend,
    entry: dict,
    *,
    model_slug: str,
    max_new_tokens: int = 512,
    max_steps_per_turn: int = 12,
    inject_retry_failure: bool = False,
) -> dict[str, Any]:
    first_user = entry["question"][0][0]["content"]

    def init_state(e: dict) -> Trajectory:
        return Trajectory(entry_to_prelude(e, user_content=first_user))

    def begin_turn(state: Trajectory, e: dict, turn_idx: int) -> Trajectory:
        if turn_idx > 0:
            state.append(
                {"type": "feedback", "content": e["question"][turn_idx][0]["content"]}
            )
        return state

    def step(state: Trajectory, e: dict) -> tuple[BFCLStep, Trajectory]:
        return _agenticml_step(state, e, backend, max_new_tokens=max_new_tokens)

    def feed_results(
        state: Trajectory, call_strings: list[str], execution_results: list[str]
    ) -> Trajectory:
        for call_str, er in zip(call_strings, execution_results):
            state.append({
                "type": "result",
                "content": execution_result_frame(er, tool_name_from_call_string(call_str)),
            })
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
    backend: AgenticMLBackend,
    entry: dict,
    ctx: RunContext,
) -> dict[str, Any]:
    return route_infer_entry(
        backend,
        entry,
        model_id=ctx.model_id,
        infer_single=infer_single_turn,
        infer_multi=infer_multi_turn_agenticml,
        max_new_tokens=ctx.max_new_tokens,
        inject_retry_failure=ctx.inject_retry_failure,
    )
