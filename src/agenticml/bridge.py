"""single source of truth for agenticml <-> chatml format conversion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from agenticml.constants import TERMINAL_TOOLS
from agenticml.evaluation.harness.chatml_fc import parse_chatml_fc_call
from agenticml.agentic_template import parse_reserved_wire, render_trajectory
from agenticml.frames import Frame, parse as parse_marker_wire
from agenticml.trajectory import Trajectory

FrameDict = dict[str, Any]
MessageDict = dict[str, Any]
FrameLike = Union[Trajectory, list[FrameDict], list[Frame], list[MessageDict]]

_TOOL_RE = re.compile(r"<\|python_tag\|>(.+?)<\|(?:eom_id|eot_id)\|>", re.DOTALL)
_TEXT_RE = re.compile(
    r"(?:<\|start_header_id\|>assistant<\|end_header_id\|>)?\s*(.*?)<\|(?:eot_id|eom_id)\|>",
    re.DOTALL,
)
_TOOLS_MARKERS = ("namespace tools {", "tools:", "available tools:")


@dataclass(frozen=True)
class ParsedChatMLGeneration:
    tool_call: Optional[dict[str, Any]]
    text: str
    stop_reason: str


class FormatBridge:
    """canonical conversions between agenticml frames/wire and chatml messages/wire."""

    def is_chatml_messages(self, data: Any) -> bool:
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and "role" in first

    def is_agenticml_frames(self, data: Any) -> bool:
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and "type" in first and "role" not in first

    def coerce_frames(self, data: FrameLike) -> list[FrameDict]:
        if isinstance(data, Trajectory):
            return data.to_dict()
        items = list(data)
        if not items:
            return []
        if self.is_chatml_messages(items):
            messages: list[MessageDict] = [
                m for m in items if isinstance(m, dict) and "role" in m
            ]
            return self.messages_to_frames(messages)
        return Trajectory(items).to_dict()

    def coerce_messages(self, data: FrameLike) -> list[MessageDict]:
        if isinstance(data, Trajectory):
            return self.frames_to_messages(data.to_dict())
        items = list(data)
        if not items:
            return []
        if self.is_chatml_messages(items):
            return [
                dict(m) for m in items if isinstance(m, dict) and "role" in m
            ]
        return self.frames_to_messages(Trajectory(items).to_dict())

    def frames_to_messages(self, frames: list[FrameDict]) -> list[MessageDict]:
        messages: list[MessageDict] = []
        system_parts: list[str] = []
        pending_reasoning: list[str] = []

        for frame in frames:
            t = frame["type"]
            c = frame["content"]

            if t == "goal":
                system_parts.append(str(c))
            elif t == "obs":
                system_parts.append(str(c))
            elif t == "mission":
                if system_parts:
                    messages.append({"role": "system", "content": "\n\n".join(system_parts)})
                    system_parts = []
                messages.append({"role": "user", "content": str(c)})
            elif t == "end":
                continue
            elif t in ("belief", "plan", "think"):
                pending_reasoning.append(str(c))
            elif t == "action":
                if not isinstance(c, dict):
                    raise ValueError(f"action frame content must be dict, got {type(c)!r}")
                tool = c.get("tool")
                args = {k: v for k, v in c.items() if k != "tool"}
                assistant_content = "\n".join(pending_reasoning) if pending_reasoning else ""
                pending_reasoning = []

                if tool in TERMINAL_TOOLS:
                    text = args.get("text") or args.get("reason") or ""
                    final_content = (
                        (assistant_content + "\n\n" + text).strip() if assistant_content else text
                    )
                    messages.append({"role": "assistant", "content": final_content})
                else:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [{
                            "id": f"call_{len(messages)}",
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(args),
                            },
                        }],
                    })
            elif t == "result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{len(messages) - 1}",
                    "content": json.dumps(c),
                })
            elif t == "feedback":
                messages.append({"role": "user", "content": str(c)})

        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

        return messages

    def messages_to_frames(self, messages: list[MessageDict]) -> list[FrameDict]:
        frames: list[FrameDict] = []
        pending_tool_names: list[str] = []

        for message in messages:
            role = message.get("role")
            if role == "system":
                frames.extend(self._system_content_to_frames(str(message.get("content") or "")))
            elif role == "user":
                content = str(message.get("content") or "")
                if any(f["type"] == "mission" for f in frames):
                    frames.append({"type": "feedback", "content": content})
                else:
                    frames.append({"type": "mission", "content": content})
            elif role == "assistant":
                new_frames = self._assistant_message_to_frames(message)
                pending_tool_names = [
                    str((f.get("content") or {}).get("tool"))
                    for f in new_frames
                    if f.get("type") == "action" and (f.get("content") or {}).get("tool")
                ]
                frames.extend(new_frames)
            elif role == "tool":
                raw = message.get("content") or "{}"
                tool = pending_tool_names.pop(0) if pending_tool_names else "unknown"
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {"tool": tool, "value": str(raw)}
                else:
                    if "tool" not in payload:
                        payload = {"tool": tool, "value": payload}
                frames.append({"type": "result", "content": payload})

        return frames

    def trajectory_to_messages(self, trajectory: FrameLike) -> list[MessageDict]:
        return self.frames_to_messages(self.coerce_frames(trajectory))

    def frames_to_agenticml_wire(self, frames: FrameLike, *, tokenizer: Any) -> str:
        trajectory = (
            frames if isinstance(frames, Trajectory)
            else Trajectory(frames if frames and isinstance(frames[0], Frame) else self.coerce_frames(frames))
        )
        return render_trajectory(tokenizer, trajectory)

    def agenticml_wire_to_frames(self, wire: str, *, strict: bool = False) -> list[FrameDict]:
        if "<|reserved_special_token_" in wire:
            parsed = parse_reserved_wire(wire, strict=strict)
        else:
            parsed = parse_marker_wire(wire, strict=strict)
        return Trajectory(parsed).to_dict()

    def agenticml_wire_to_messages(self, wire: str, *, strict: bool = False) -> list[MessageDict]:
        return self.frames_to_messages(self.agenticml_wire_to_frames(wire, strict=strict))

    def parse_chatml_generation(self, text: str) -> ParsedChatMLGeneration:
        m = _TOOL_RE.search(text)
        if m:
            try:
                call = json.loads(m.group(1))
                if not isinstance(call, dict):
                    return ParsedChatMLGeneration(None, "", "parse_error: not object")
                return ParsedChatMLGeneration(call, "", "tool_call")
            except json.JSONDecodeError as e:
                return ParsedChatMLGeneration(None, "", f"parse_error: {e.msg}")
        bare = parse_chatml_fc_call(text)
        if bare is not None:
            return ParsedChatMLGeneration(bare, "", "tool_call")
        m = _TEXT_RE.search(text)
        content = (m.group(1) if m else text).strip()
        if content:
            return ParsedChatMLGeneration(None, content, "assistant_text")
        return ParsedChatMLGeneration(None, "", "parse_error: empty")

    def inject_tool_schemas(self, messages: list[MessageDict], tools: list[dict]) -> list[MessageDict]:
        if not tools:
            return list(messages)
        block = "available tools:\n" + json.dumps(tools)
        out = list(messages)
        for i, message in enumerate(out):
            if message.get("role") == "system":
                out[i] = {**message, "content": f"{message.get('content', '')}\n\n{block}".strip()}
                return out
        return [{"role": "system", "content": block}, *out]

    def tool_name_args(self, call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = call.get("name") or call.get("tool")
        if not name:
            raise ValueError("tool call missing name")
        raw = call.get("arguments", call.get("parameters", {}))
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        return str(name), args

    def _system_content_to_frames(self, content: str) -> list[FrameDict]:
        for marker in _TOOLS_MARKERS:
            if marker in content:
                idx = content.index(marker)
                prefix = content[:idx].strip()
                tools_part = content[idx:].strip()
                frames: list[FrameDict] = []
                if prefix:
                    frames.append({"type": "goal", "content": prefix})
                frames.append({"type": "obs", "content": tools_part})
                return frames

        parts = content.split("\n\n")
        if len(parts) >= 2 and any("tools" in part.lower() for part in parts[1:]):
            return [
                {"type": "goal", "content": parts[0]},
                {"type": "obs", "content": "\n\n".join(parts[1:])},
            ]
        return [{"type": "goal", "content": content}]

    def _assistant_message_to_frames(self, message: MessageDict) -> list[FrameDict]:
        frames: list[FrameDict] = []
        tool_calls = message.get("tool_calls") or []
        content = str(message.get("content") or "")
        if tool_calls:
            if content.strip():
                frames.append({"type": "think", "content": content.strip()})
            for tool_call in tool_calls:
                fn = tool_call.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                frames.append({"type": "action", "content": {"tool": name, **args}})
            return frames
        if content.strip():
            frames.append({"type": "action", "content": {"tool": "answer", "text": content.strip()}})
        return frames


bridge = FormatBridge()
