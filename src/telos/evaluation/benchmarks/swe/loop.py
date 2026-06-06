"""agent loops for swe-bench via mini-swe bash environment (telos + chatml)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from telos.constants import FrameType
from telos.evaluation.benchmarks.swe.env import REPEAT_COMMAND_LIMIT
from telos.evaluation.benchmarks.swe.prelude import instance_to_messages, instance_to_prelude
from telos.evaluation.benchmarks.swe.registry import SweEnvBridge, registry_from_bridge
from telos.evaluation.harness.backends.chatml_backend import ChatMLBackend, _tool_name_args
from telos.evaluation.harness.backends.telos_backend import TelosBackend
from telos.frames import Frame
from telos.runtime.tools import ToolError
from telos.trajectory import Trajectory

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


def run_telos_swe(
    backend: TelosBackend,
    bridge: SweEnvBridge,
    instance: dict[str, Any],
    *,
    max_iterations: int = 250,
    max_new_tokens: int = 512,
) -> SweRunResult:
    """run telos steps against a mini-swe bash environment until submit or limit."""
    instance_id = str(instance.get("instance_id") or "")
    registry = registry_from_bridge(bridge)
    traj = Trajectory(instance_to_prelude(instance))

    prompt_tokens = 0
    generated_tokens = 0
    inference_sec = 0.0
    repeat_tracker = _RepeatTracker()

    for iteration in range(1, max_iterations + 1):
        if bridge.submission is not None:
            return SweRunResult(
                instance_id=instance_id,
                trajectory=traj,
                stopped_on="submitted",
                iterations=iteration - 1,
                model_patch=bridge.submission,
                exit_status=bridge.exit_status,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        out = backend.step(
            traj,
            registry.schemas(),
            max_new_tokens=max_new_tokens,
            strict=False,
        )
        prompt_tokens += out.prompt_tokens
        generated_tokens += out.generated_tokens
        inference_sec += out.inference_sec
        _log(
            instance_id,
            iteration,
            f"generate {out.generated_tokens} tok in {out.inference_sec:.1f}s",
        )

        if out.step is None:
            return SweRunResult(
                instance_id=instance_id,
                trajectory=traj,
                stopped_on="step_failed",
                iterations=iteration,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        step_result = out.step
        if step_result.stopped_on.startswith("parse_error"):
            return SweRunResult(
                instance_id=instance_id,
                trajectory=step_result.trajectory,
                stopped_on=step_result.stopped_on,
                iterations=iteration,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        traj = step_result.trajectory
        actions = [
            f for f in step_result.new_frames.to_frames() if f.type == FrameType.ACTION
        ]
        if not actions:
            return SweRunResult(
                instance_id=instance_id,
                trajectory=traj,
                stopped_on="no_action",
                iterations=iteration,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        for action_frame in actions:
            content = action_frame.content or {}
            tool_name = content.get("tool")
            args = {k: v for k, v in content.items() if k != "tool"}
            cmd = str(args.get("command") or "")
            if repeat_tracker.would_repeat(cmd):
                _log(instance_id, iteration, f"repeated command ({REPEAT_COMMAND_LIMIT}x): {_preview(cmd)}")
                return SweRunResult(
                    instance_id=instance_id,
                    trajectory=traj,
                    stopped_on="repeated_command",
                    iterations=iteration,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    inference_sec=inference_sec,
                )
            try:
                value = registry.call(str(tool_name), args)
                traj.append(Frame(FrameType.RESULT, content={"ok": 1, "value": value}))
                _log(instance_id, iteration, f"bash: {_preview(cmd)} -> {_preview(str(value))}")
            except ToolError as exc:
                traj.append(Frame(FrameType.RESULT, content={"ok": 0, "value": str(exc)}))
                _log(instance_id, iteration, f"bash: {_preview(cmd)} -> error: {exc}")

            if bridge.submission is not None:
                _log(instance_id, iteration, f"submitted patch ({len(bridge.submission)} chars)")
                return SweRunResult(
                    instance_id=instance_id,
                    trajectory=traj,
                    stopped_on="submitted",
                    iterations=iteration,
                    model_patch=bridge.submission,
                    exit_status=bridge.exit_status,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    inference_sec=inference_sec,
                )

        if step_result.stopped_on == "max_tokens":
            return SweRunResult(
                instance_id=instance_id,
                trajectory=traj,
                stopped_on="max_tokens",
                iterations=iteration,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

    return SweRunResult(
        instance_id=instance_id,
        trajectory=traj,
        stopped_on="max_iterations",
        iterations=max_iterations,
        model_patch=bridge.submission,
        exit_status=bridge.exit_status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        inference_sec=inference_sec,
    )


def _submitted_result(
    *,
    instance_id: str,
    stopped_on: str,
    iterations: int,
    bridge: SweEnvBridge,
    prompt_tokens: int,
    generated_tokens: int,
    inference_sec: float,
    trajectory: Trajectory | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> SweRunResult:
    return SweRunResult(
        instance_id=instance_id,
        stopped_on=stopped_on,
        iterations=iterations,
        trajectory=trajectory,
        messages=messages,
        model_patch=bridge.submission,
        exit_status=bridge.exit_status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        inference_sec=inference_sec,
    )


def run_chatml_swe(
    backend: ChatMLBackend,
    bridge: SweEnvBridge,
    instance: dict[str, Any],
    *,
    max_iterations: int = 250,
    max_new_tokens: int = 512,
) -> SweRunResult:
    """run chatml steps against a mini-swe bash environment until submit or limit."""
    instance_id = str(instance.get("instance_id") or "")
    registry = registry_from_bridge(bridge)
    messages = instance_to_messages(instance)

    prompt_tokens = 0
    generated_tokens = 0
    inference_sec = 0.0
    repeat_tracker = _RepeatTracker()

    for iteration in range(1, max_iterations + 1):
        if bridge.submission is not None:
            return _submitted_result(
                instance_id=instance_id,
                stopped_on="submitted",
                iterations=iteration - 1,
                bridge=bridge,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
                messages=messages,
            )

        out = backend.step(
            messages,
            registry.schemas(),
            max_new_tokens=max_new_tokens,
            strict=False,
        )
        prompt_tokens += out.prompt_tokens
        generated_tokens += out.generated_tokens
        inference_sec += out.inference_sec
        _log(
            instance_id,
            iteration,
            f"generate {out.generated_tokens} tok in {out.inference_sec:.1f}s",
        )

        if out.stopped_on.startswith("parse_error"):
            return SweRunResult(
                instance_id=instance_id,
                stopped_on=out.stopped_on,
                iterations=iteration,
                messages=list(out.messages),
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        messages = list(out.messages)
        assistant = out.new_messages[0] if out.new_messages else {}
        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            stop = "no_action"
            return SweRunResult(
                instance_id=instance_id,
                stopped_on=stop,
                iterations=iteration,
                messages=messages,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                inference_sec=inference_sec,
            )

        for tc in tool_calls:
            fn = tc.get("function") or {}
            call_id = str(tc.get("id") or f"call_{iteration}")
            name, args = _tool_name_args(
                {"name": fn.get("name"), "arguments": fn.get("arguments", "{}")}
            )
            cmd = str(args.get("command") or "")
            if repeat_tracker.would_repeat(cmd):
                _log(instance_id, iteration, f"repeated command ({REPEAT_COMMAND_LIMIT}x): {_preview(cmd)}")
                return SweRunResult(
                    instance_id=instance_id,
                    stopped_on="repeated_command",
                    iterations=iteration,
                    messages=messages,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    inference_sec=inference_sec,
                )
            try:
                value = registry.call(name, args)
                payload = {"ok": 1, "value": value}
                _log(instance_id, iteration, f"bash: {_preview(cmd)} -> {_preview(str(value))}")
            except ToolError as exc:
                payload = {"ok": 0, "value": str(exc)}
                _log(instance_id, iteration, f"bash: {_preview(cmd)} -> error: {exc}")
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)}
            )

            if bridge.submission is not None:
                _log(instance_id, iteration, f"submitted patch ({len(bridge.submission)} chars)")
                return _submitted_result(
                    instance_id=instance_id,
                    stopped_on="submitted",
                    iterations=iteration,
                    bridge=bridge,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    inference_sec=inference_sec,
                    messages=messages,
                )

    return SweRunResult(
        instance_id=instance_id,
        stopped_on="max_iterations",
        iterations=max_iterations,
        messages=messages,
        model_patch=bridge.submission,
        exit_status=bridge.exit_status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        inference_sec=inference_sec,
    )
