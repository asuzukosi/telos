"""
agenticml sdk for stateless trajectory advancement.

the public entry point is `step()`. callers build the trajectory (including tool
definitions in an <|obs|> frame via `with_tool_obs`) before calling `step()`.

sdk does not execute tools. when the model emits an action, the caller runs the
tool and appends the result frame before calling `step()` again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union
from transformers import PreTrainedTokenizerBase
from agenticml.agentic_template import parse_reserved_wire
from agenticml.tokenizer_helpers import chat_template_ids
from agenticml.constants import END_MARKER_TOKEN_ID, FrameType, WIRE_END_MARKER
from agenticml.frames import obs
from agenticml.trajectory import FrameLike, Trajectory

_TOOL_OBS_MARKER = "namespace tools {"


def _ts_type(spec: dict) -> str:
    """translate a JSON Schema fragment to a rough TypeScript type."""
    t = spec.get("type")
    if t == "string":
        if "enum" in spec:
            return " | ".join(repr(v) for v in spec["enum"])
        return "string"
    if t in ("integer", "number"):
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        item_type = _ts_type(spec.get("items", {}))
        return f"{item_type}[]"
    if t == "object":
        return "object"
    return "any"


def render_tool_schema(tools: list[dict[str, Any]]) -> str:
    """render an openai-style tool schema list to a typescript namespace."""
    if not tools:
        return ""
    lines = ["tools:", "namespace tools {"]
    for t in tools:
        name = t.get("name", "<unnamed>")
        description = t.get("description", "")
        parameters: dict[str, Any] = t.get("parameters", {})
        properties: dict[str, Any] = parameters.get("properties", {})
        required: list[str] = parameters.get("required", [])

        if description:
            lines.append(f" // {description}")
        if properties:
            lines.append(f" type {name} = (_: {{)")
            for pname, pspec in properties.items():
                ptype = _ts_type(pspec)
                optional = "" if pname in required else "?"
                pdesc = pspec.get("description")
                if pdesc:
                    lines.append(f"   // {pdesc}")
                lines.append(f"   {pname}{optional}: {ptype},")
            lines.append("  }) => any;")
        else:
            lines.append(f" type {name} = () => any;")
    lines.append("}")
    return "\n".join(lines)


def has_tool_obs(trajectory: Trajectory) -> bool:
    """true when the trajectory already has an obs frame with tool definitions."""
    for frame in trajectory:
        if frame.type is FrameType.OBS and _TOOL_OBS_MARKER in str(frame.content or ""):
            return True
    return False


def with_tool_obs(
    trajectory: Union[Trajectory, Iterable[FrameLike]],
    tools: Optional[list[dict]] = None,
) -> Trajectory:
    """return trajectory with a prelude obs frame for tools when missing."""
    traj = trajectory if isinstance(trajectory, Trajectory) else Trajectory(trajectory)
    if not tools or has_tool_obs(traj):
        return traj

    frames = traj.to_frames()
    insert_at = 0
    for i, frame in enumerate(frames):
        if frame.type is FrameType.GOAL:
            insert_at = i + 1
            break
        insert_at = i + 1
    frames.insert(insert_at, obs(render_tool_schema(tools)))
    return Trajectory(frames)


GenerateFn = Callable[[list[int], int, int], list[int]]


@dataclass
class StepResult:
    """return value of `step()`."""

    trajectory: Trajectory
    new_frames: Trajectory
    stopped_on: str
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory.to_dict(),
            "new_frames": self.new_frames.to_dict(),
            "stopped_on": self.stopped_on,
            "raw_text": self.raw_text,
        }


TrajectoryInput = Union[Trajectory, Iterable[FrameLike]]


def step(
    trajectory: TrajectoryInput,
    *,
    tokenizer: PreTrainedTokenizerBase,
    generate: GenerateFn,
    max_new_tokens: int = 512,
    strict: bool = True,
) -> StepResult:
    """advance the trajectory by one generation cycle."""
    input_traj = trajectory if isinstance(trajectory, Trajectory) else Trajectory(trajectory)

    prompt_ids = chat_template_ids(
        tokenizer,
        input_traj.to_dict(),
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    new_ids = generate(prompt_ids, END_MARKER_TOKEN_ID, max_new_tokens)

    if new_ids and new_ids[-1] == END_MARKER_TOKEN_ID:
        stopped_on = WIRE_END_MARKER
    elif len(new_ids) >= max_new_tokens:
        stopped_on = "max_tokens"
    else:
        stopped_on = "other"

    raw_text = tokenizer.decode(new_ids)
    try:
        new_frame_objs = parse_reserved_wire(raw_text, strict=strict)
    except Exception as e:
        return StepResult(
            trajectory=input_traj,
            new_frames=Trajectory(),
            stopped_on=f"parse_error: {e}",
            raw_text=raw_text,
        )

    new_traj = Trajectory(new_frame_objs)
    return StepResult(
        trajectory=input_traj + new_traj,
        new_frames=new_traj,
        stopped_on=stopped_on,
        raw_text=raw_text,
    )
