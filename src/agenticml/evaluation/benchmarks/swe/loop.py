"""agent loops for swe-bench via mini-swe bash environment (agenticml + chatml)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from agenticml.constants import FrameType
from agenticml.evaluation.benchmarks.swe.env import REPEAT_COMMAND_LIMIT
from agenticml.evaluation.benchmarks.swe.prelude import instance_to_messages, instance_to_prelude
from agenticml.evaluation.benchmarks.swe.registry import SweEnvBridge, registry_from_bridge
from agenticml.bridge import bridge as format_bridge
from agenticml.evaluation.harness.backends.chatml_backend import ChatMLBackend
from agenticml.evaluation.harness.backends.agenticml_backend import AgenticMLBackend
from agenticml.frames import Frame
from agenticml.runtime.tools import ToolError, ToolRegistry
from agenticml.sdk import with_tool_obs
from agenticml.trajectory import Trajectory

_PREVIEW = 300


def _verbose() -> bool:
    return os.environ.get("SWE_VERBOSE", "").lower() in ("1", "true", "yes")


def _log(instance_id: str, iteration: int, msg: str) -> None:
    if _verbose():
        print(f"[swe {instance_id} #{iteration}] {msg}", flush=True)


def _preview(text: str, limit: int = _PREVIEW) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class _RepeatTracker:
    last_cmd: str | None = None
    streak: int = 0

    def would_repeat(self, cmd: str, *, limit: int = REPEAT_COMMAND_LIMIT) -> bool:
        if not cmd:
            return False
        if cmd == self.last_cmd:
            self.streak += 1
        else:
            self.last_cmd = cmd
            self.streak = 1
        return self.streak >= limit


@dataclass
class SweRunResult:
    instance_id: str
    stopped_on: str
    iterations: int
    trajectory: Trajectory | None = None
    messages: list[dict[str, Any]] | None = None
    model_patch: str | None = None
    exit_status: str | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    inference_sec: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "stopped_on": self.stopped_on,
            "iterations": self.iterations,
            "model_patch": self.model_patch or "",
            "exit_status": self.exit_status,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "inference_sec": self.inference_sec,
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else [],
            "messages": list(self.messages) if self.messages is not None else [],
        }


@dataclass
class _ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass
class _StepOutcome:
    stopped_on: str
    tool_calls: list[_ToolCall]
    prompt_tokens: int
    generated_tokens: int
    inference_sec: float


def _submitted(
    *,
    instance_id: str,
    iterations: int,
    bridge: SweEnvBridge,
    trajectory: Trajectory | None,
    messages: list[dict[str, Any]] | None,
    prompt_tokens: int,
    generated_tokens: int,
    inference_sec: float,
) -> SweRunResult:
    return SweRunResult(
        instance_id=instance_id,
        stopped_on="submitted",
        iterations=iterations,
        trajectory=trajectory,
        messages=messages,
        model_patch=bridge.submission,
        exit_status=bridge.exit_status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        inference_sec=inference_sec,
    )


def _run_tool_calls(
    *,
    instance_id: str,
    iteration: int,
    tool_calls: list[_ToolCall],
    registry: ToolRegistry,
    bridge: SweEnvBridge,
    repeat_tracker: _RepeatTracker,
    apply_result: Callable[[_ToolCall, dict[str, Any]], None],
) -> str | None:
    for tc in tool_calls:
        cmd = str(tc.args.get("command") or "")
        if repeat_tracker.would_repeat(cmd):
            _log(instance_id, iteration, f"repeated command ({REPEAT_COMMAND_LIMIT}x): {_preview(cmd)}")
            return "repeated_command"
        try:
            value = registry.call(tc.name, tc.args)
            payload = {"tool": tc.name, "value": value}
            _log(instance_id, iteration, f"bash: {_preview(cmd)} -> {_preview(str(value))}")
        except ToolError as exc:
            payload = {"tool": tc.name, "value": str(exc)}
            _log(instance_id, iteration, f"bash: {_preview(cmd)} -> error: {exc}")
        apply_result(tc, payload)
        if bridge.submission is not None:
            _log(instance_id, iteration, f"submitted patch ({len(bridge.submission)} chars)")
            return "submitted"
    return None


def _run_swe_loop(
    instance_id: str,
    bridge: SweEnvBridge,
    *,
    max_iterations: int,
    trajectory: Trajectory | None,
    messages: list[dict[str, Any]] | None,
    step_once: Callable[[int], _StepOutcome | SweRunResult],
    apply_result: Callable[[_ToolCall, dict[str, Any]], None],
) -> SweRunResult:
    registry = registry_from_bridge(bridge)
    repeat_tracker = _RepeatTracker()
    prompt_tokens = 0
    generated_tokens = 0
    inference_sec = 0.0

    for iteration in range(1, max_iterations + 1):
        if bridge.submission is not None:
            return _submitted(
                instance_id=instance_id,
                iterations=iteration - 1,
                bridge=bridge,
                trajectory=trajectory,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        outcome = step_once(iteration)
        if isinstance(outcome, SweRunResult):
            outcome.prompt_tokens += prompt_tokens
            outcome.generated_tokens += generated_tokens
            outcome.inference_sec += inference_sec
            outcome.trajectory = trajectory
            outcome.messages = messages
            return outcome

        prompt_tokens += outcome.prompt_tokens
        generated_tokens += outcome.generated_tokens
        inference_sec += outcome.inference_sec
        _log(
            instance_id,
            iteration,
            f"generate {outcome.generated_tokens} tok in {outcome.inference_sec:.1f}s",
        )

        if outcome.stopped_on not in ("ok", "max_tokens"):
            return SweRunResult(
                instance_id=instance_id,
                stopped_on=outcome.stopped_on,
                iterations=iteration,
                trajectory=trajectory,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        stop = _run_tool_calls(
            instance_id=instance_id,
            iteration=iteration,
            tool_calls=outcome.tool_calls,
            registry=registry,
            bridge=bridge,
            repeat_tracker=repeat_tracker,
            apply_result=apply_result,
        )
        if stop == "submitted":
            return _submitted(
                instance_id=instance_id,
                iterations=iteration,
                bridge=bridge,
                trajectory=trajectory,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )
        if stop == "repeated_command":
            return SweRunResult(
                instance_id=instance_id,
                stopped_on="repeated_command",
                iterations=iteration,
                trajectory=trajectory,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )
        if outcome.stopped_on == "max_tokens":
            return SweRunResult(
                instance_id=instance_id,
                stopped_on="max_tokens",
                iterations=iteration,
                trajectory=trajectory,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

    return SweRunResult(
        instance_id=instance_id,
        stopped_on="max_iterations",
        iterations=max_iterations,
        trajectory=trajectory,
        messages=messages,
        model_patch=bridge.submission,
        exit_status=bridge.exit_status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        inference_sec=inference_sec,
    )


def run_agenticml_swe(
    backend: AgenticMLBackend,
    bridge: SweEnvBridge,
    instance: dict[str, Any],
    *,
    max_iterations: int = 250,
    max_new_tokens: int = 512,
) -> SweRunResult:
    instance_id = str(instance.get("instance_id") or "")
    traj = with_tool_obs(
        Trajectory(instance_to_prelude(instance)),
        registry_from_bridge(bridge).schemas(),
    )

    def step_once(_iteration: int) -> _StepOutcome | SweRunResult:
        out = backend.step(traj, max_new_tokens=max_new_tokens, strict=False)
        if out.step is None:
            return SweRunResult(instance_id=instance_id, stopped_on="step_failed", iterations=_iteration)
        step_result = out.step
        if step_result.stopped_on.startswith("parse_error"):
            return SweRunResult(
                instance_id=instance_id,
                stopped_on=step_result.stopped_on,
                iterations=_iteration,
                trajectory=traj,
            )
        traj.extend(step_result.new_frames)
        actions = [f for f in step_result.new_frames.to_frames() if f.type == FrameType.ACTION]
        if not actions:
            return _StepOutcome("no_action", [], out.prompt_tokens, out.generated_tokens, out.inference_sec)
        tool_calls = [
            _ToolCall(
                str((a.content or {}).get("tool") or ""),
                {k: v for k, v in (a.content or {}).items() if k != "tool"},
            )
            for a in actions
        ]
        stop = "max_tokens" if step_result.stopped_on == "max_tokens" else "ok"
        return _StepOutcome(stop, tool_calls, out.prompt_tokens, out.generated_tokens, out.inference_sec)

    def apply_result(_tc: _ToolCall, payload: dict[str, Any]) -> None:
        traj.append(Frame(FrameType.RESULT, content=payload))

    return _run_swe_loop(
        instance_id,
        bridge,
        max_iterations=max_iterations,
        trajectory=traj,
        messages=None,
        step_once=step_once,
        apply_result=apply_result,
    )


def run_chatml_swe(
    backend: ChatMLBackend,
    bridge: SweEnvBridge,
    instance: dict[str, Any],
    *,
    max_iterations: int = 250,
    max_new_tokens: int = 512,
) -> SweRunResult:
    instance_id = str(instance.get("instance_id") or "")
    messages = instance_to_messages(instance)

    def step_once(_iteration: int) -> _StepOutcome | SweRunResult:
        out = backend.step(messages, registry_from_bridge(bridge).schemas(), max_new_tokens=max_new_tokens, strict=False)
        if out.stopped_on.startswith("parse_error"):
            messages[:] = out.messages or []
            return SweRunResult(
                instance_id=instance_id,
                stopped_on=out.stopped_on,
                iterations=_iteration,
                messages=messages,
            )
        messages[:] = out.messages or []
        assistant = out.new_messages[0] if out.new_messages else {}
        tool_calls_raw = assistant.get("tool_calls") or []
        if not tool_calls_raw:
            return _StepOutcome("no_action", [], out.prompt_tokens, out.generated_tokens, out.inference_sec)
        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            name, args = format_bridge.tool_name_args({"name": fn.get("name"), "arguments": fn.get("arguments", "{}")})
            tool_calls.append(_ToolCall(name, args, call_id=str(tc.get("id") or f"call_{_iteration}")))
        return _StepOutcome("ok", tool_calls, out.prompt_tokens, out.generated_tokens, out.inference_sec)

    def apply_result(tc: _ToolCall, payload: dict[str, Any]) -> None:
        messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(payload)})

    return _run_swe_loop(
        instance_id,
        bridge,
        max_iterations=max_iterations,
        trajectory=None,
        messages=messages,
        step_once=step_once,
        apply_result=apply_result,
    )
