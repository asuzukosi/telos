"""
frame definitions and marker-wire parser for agenticml.

rendering uses the hub tokenizer chat template; decoding model output uses
agenticml.agentic_template.parse_reserved_wire.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from agenticml.constants import AGENTICML_OWNERS, FrameType, FrameOwner

_JSON_TYPES = frozenset({FrameType.ACTION, FrameType.RESULT})
_NUMBER_TYPES = frozenset({FrameType.REWARD})
_EMPTY_TYPES = frozenset({FrameType.END})
_MARKERS: tuple[str, ...] = tuple(ft.value for ft in FrameType)
_MARKER_TO_TYPE: dict[str, FrameType] = {ft.value: ft for ft in FrameType}


@dataclass
class Frame:
    type: FrameType
    content: Any = None
    raw: str = ""
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def owner(self) -> FrameOwner:
        return AGENTICML_OWNERS[self.type]

def goal(text: str) -> Frame:        return Frame(FrameType.GOAL, text)
def mission(text: str) -> Frame:     return Frame(FrameType.MISSION, text)
def obs(text: str) -> Frame:         return Frame(FrameType.OBS, text)
def belief(text: str) -> Frame:      return Frame(FrameType.BELIEF, text)
def plan(text: str) -> Frame:        return Frame(FrameType.PLAN, text)
def think(text: str) -> Frame:       return Frame(FrameType.THINK, text)
def action(payload: dict) -> Frame:  return Frame(FrameType.ACTION, payload)
def end() -> Frame:                  return Frame(FrameType.END)
def result(payload: dict) -> Frame:  return Frame(FrameType.RESULT, payload)
def feedback(text: str) -> Frame:    return Frame(FrameType.FEEDBACK, text)
def reward(value: float) -> Frame:   return Frame(FrameType.REWARD, float(value))


class AgenticMLParseError(ValueError):
    """raised when a trajectory string cannot be parsed."""
 
 
class AgenticMLOwnershipError(AgenticMLParseError):
    """raised in strict mode when a runtime-owned frame is found in
    model-only output."""


def _find_next_marker(text: str, start: int) -> tuple[int, Optional[str]]:
    best_idx = len(text)
    best_marker: Optional[str] = None
    for marker in _MARKERS:
        idx = text.find(marker, start)
        if idx != -1 and idx < best_idx:
            best_idx = idx
            best_marker = marker
    return best_idx, best_marker

def _parse_payload(ft: FrameType, body: str) -> tuple[Any, Optional[str]]:
    if ft in _EMPTY_TYPES:
        return None, None
    if ft in _JSON_TYPES:
        stripped = body.strip()
        if not stripped:
            return None, "empty json payload"
        try:
            return json.loads(stripped), None
        except json.JSONDecodeError as e:
            return None, f"invalid json: {e.msg} at pos {e.pos}"
    if ft in _NUMBER_TYPES:
        stripped = body.strip()
        if not stripped:
            return None, "empty reward payload"
        try:
            return float(stripped), None
        except ValueError as e:
            return None, f"invalid number: {e}"
    return body.rstrip(), None

def parse(text: str, *, strict: bool = False) -> list[Frame]:
    """parse human-readable marker wire into frames."""
    frames: list[Frame] = []
    n = len(text)
    idx = 0
    while idx < n and text[idx].isspace():
        idx += 1
    if idx >= n:
        return frames
 
    cur_marker_idx, cur_marker = _find_next_marker(text, idx)
    if cur_marker is None or cur_marker_idx != idx:
        preview = text[idx : min(idx + 40, n)].replace("\n", "\\n")
        raise AgenticMLParseError(
            f"expected frame marker at offset {idx}, got: {preview!r}"
        )
    while cur_marker is not None:
        cur_type = _MARKER_TO_TYPE[cur_marker]
        if strict and AGENTICML_OWNERS[cur_type] == FrameOwner.RUNTIME:
            raise AgenticMLOwnershipError(
                f"runtime-owned marker {cur_marker!r} found in model output"
            )
        body_start = cur_marker_idx + len(cur_marker)
        next_marker_idx, next_marker = _find_next_marker(text, body_start)
        raw_body = text[body_start:next_marker_idx]
        content, error = _parse_payload(cur_type, raw_body)
        frames.append(Frame(type=cur_type, content=content, raw=raw_body, error=error))
        cur_marker_idx, cur_marker = next_marker_idx, next_marker
    return frames
