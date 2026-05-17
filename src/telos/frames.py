"""
frame definitions, parser and render for the telos wire format.

a trajectory is a sequence of frames. each frame opens with one of the 10 telos
marker tokens and extends until the next marker or end-of-string.

serialization for the model: render() inserts the string END_MARKER ("<|end|>")
after the last model-owned frame before a runtime-owned frame (and after the
final frame if it is model-owned). that marker is not a separate Frame — parse()
skips it and never returns an end frame.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from telos.constants import TELOS_OWNERS, TELOS_TOKEN_MAP, FrameType, FrameOwner, END_MARKER
from telos.utils import format_number

class PayloadKind(Enum):
    PROSE = "prose"
    JSON = "json"
    NUMBER = "number"

# per-type payload-kind classification.
TELOS_PAYLOAD_KIND: dict[str, PayloadKind] = {
    "<|goal|>":     PayloadKind.PROSE,
    "<|mission|>":  PayloadKind.PROSE,
    "<|obs|>":      PayloadKind.PROSE,
    "<|belief|>":   PayloadKind.PROSE,
    "<|plan|>":     PayloadKind.PROSE,
    "<|think|>":    PayloadKind.PROSE,
    "<|action|>":   PayloadKind.JSON,
    "<|result|>":   PayloadKind.JSON,
    "<|feedback|>": PayloadKind.PROSE,
    "<|reward|>":   PayloadKind.NUMBER,
}
 
# set of all known markers, for fast membership tests in the parser.
TELOS_MARKERS: tuple[str, ...] = tuple(name for name, _ in TELOS_TOKEN_MAP) + (END_MARKER,)
_MARKER_SET: frozenset[str] = frozenset(TELOS_MARKERS)

@dataclass
class Frame:
    """
    a single frame in a telos trajectory.
    
    The frame type is the canonical wire-format marker string for that frame type.
    The content is the payload of the frame.
    The raw is the raw wire-format string for the frame.
    The error is an optional error message for the frame.
    The meta is a dictionary of metadata for the frame.
    """

    type: FrameType
    content: Any = None
    raw: str = ""
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def owner(self) -> FrameOwner:
        return TELOS_OWNERS[self.type]

    @property
    def kind(self) -> PayloadKind:
        return TELOS_PAYLOAD_KIND[self.type]

def goal(text: str) -> Frame:        return Frame(FrameType.GOAL, text)
def mission(text: str) -> Frame:     return Frame(FrameType.MISSION, text)
def obs(text: str) -> Frame:         return Frame(FrameType.OBS, text)
def belief(text: str) -> Frame:      return Frame(FrameType.BELIEF, text)
def plan(text: str) -> Frame:        return Frame(FrameType.PLAN, text)
def think(text: str) -> Frame:       return Frame(FrameType.THINK, text)
def action(payload: dict) -> Frame:  return Frame(FrameType.ACTION, payload)
def result(payload: dict) -> Frame:  return Frame(FrameType.RESULT, payload)
def feedback(text: str) -> Frame:    return Frame(FrameType.FEEDBACK, text)
def reward(value: float) -> Frame:   return Frame(FrameType.REWARD, float(value))


class TelosParseError(ValueError):
    """raised when a trajectory string cannot be parsed."""
 
 
class TelosOwnershipError(TelosParseError):
    """raised in strict mode when a runtime-owned frame is found in
    model-only output."""


def _render_payload(frame: Frame) -> str:
    kind = frame.kind
    if kind == PayloadKind.PROSE:
        return "" if frame.content is None else str(frame.content)
    if kind == PayloadKind.JSON:
        if frame.content is None:
            return frame.raw  # fall back to raw if no parsed content
        return json.dumps(frame.content, separators=(",", ":"), ensure_ascii=False)
    if kind == PayloadKind.NUMBER:
        return "" if frame.content is None else format_number(frame.content)
    raise ValueError(f"unknown payload kind: {kind}")

def render_frame(frame: Frame) -> str:
    """render a single frame to its wire-format substring."""
    return f"{frame.type.value}{_render_payload(frame)}"

def render(frames: list[Frame], *, separator: str = "\n") -> str:
    """render frames to wire text for the model. appends END_MARKER between model
    and runtime segments; END_MARKER is not a Frame and is ignored by parse().
    """
    parts: list[str] = []
    n = len(frames)
    for i, f in enumerate(frames):
        if i > 0 and separator and not parts[-1].endswith("\n"):
            parts.append(separator)
        parts.append(render_frame(f))
        # emit END_MARKER at any model->runtime boundary, or at the
        # final frame if it is model-owned.
        is_last = (i == n - 1)
        next_is_runtime = (not is_last) and (TELOS_OWNERS[frames[i + 1].type] == FrameOwner.RUNTIME)
        if f.owner == FrameOwner.MODEL and (is_last or next_is_runtime):
            if separator and not parts[-1].endswith("\n"):
                parts.append(separator)
            parts.append(END_MARKER)
    return "".join(parts)

_MARKER_TO_TYPE: dict[str, FrameType] = {
    marker: FrameType(marker) for marker, _ in TELOS_TOKEN_MAP
}


def _find_next_marker(text: str, start: int) -> tuple[int, Optional[str]]:
    """return (index, marker string) for the next marker at or after start.
    if no marker is found, return (len(text), None).
    """
    best_idx = len(text)
    best_marker: Optional[str] = None
    for marker in TELOS_MARKERS:
        idx = text.find(marker, start)
        if idx == -1:
            continue
        if idx < best_idx:
            best_idx = idx
            best_marker = marker
    return best_idx, best_marker

def _parse_payload(ft: FrameType, body: str) -> tuple[Any, Optional[str]]:
    """parse the payload of a frame. return (content, error_or_None)."""
    kind = TELOS_PAYLOAD_KIND[ft]

    if kind == PayloadKind.PROSE:
        return body.rstrip(), None
    
    if kind == PayloadKind.JSON:
        stripped = body.strip()
        if not stripped:
            return None, "empty json payload"
        try:
            return json.loads(stripped), None
        except json.JSONDecodeError as e:
            return None, f"invalid json: {e.msg} at pos {e.pos}"
    
    if kind == PayloadKind.NUMBER:
        stripped = body.strip()
        if not stripped:
            return None, "empty reward payload"
        try:
            return float(stripped), None
        except ValueError as e:
            return None, f"invalid number: {e}"
    
    return None, f"unknown payload kind: {kind}"

def parse(text: str, *, strict: bool = False) -> list[Frame]:
    """parse wire text into Frames. END_MARKER is only a boundary token: it ends
    the current payload search and does not produce a Frame.

    frames are identified by their opening markers. Each frame extends from its
    marker until the next marker or end-of-string. whitespace between frames is
    discarded; whitespace inside prose frames is preserved (except trailing
    whitespace before the next marker).

    args:
      text:   the wire-format string.
      strict: if True, raise TelosOwnershipError when a runtime-owned marker is encountered. use this when parsing live model output to catch hallucinated frames.

    returns:
      a list of frame objects. frames with malformed payloads are included with content=None and error set.
    """
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
        raise TelosParseError(
            f"expected frame marker at offset {idx}, got: {preview!r}"
        )
    while cur_marker is not None:
        # END_MARKER is a block boundary - skip it, advance to next
        if cur_marker == END_MARKER:
            body_start = cur_marker_idx + len(END_MARKER)
            cur_marker_idx, cur_marker = _find_next_marker(text, body_start)
            continue
        cur_type = _MARKER_TO_TYPE[cur_marker]
        if strict and TELOS_OWNERS[cur_type] == FrameOwner.RUNTIME:
            raise TelosOwnershipError(
                f"runtime-owned marker {cur_marker!r} found in model output"
            )
        body_start = cur_marker_idx + len(cur_marker)
        next_marker_idx, next_marker = _find_next_marker(text, body_start)
        raw_body = text[body_start:next_marker_idx]
        content, error = _parse_payload(cur_type, raw_body)
        frames.append(Frame(type=cur_type, content=content, raw=raw_body, error=error))
        cur_marker_idx, cur_marker = next_marker_idx, next_marker
    return frames


def sanitize(text: str) -> str:
    """strip telos marker strings from external text.
    """
    out = text
    for marker in TELOS_MARKERS:
        if marker in out:
            out = out.replace(marker, "")
    return out