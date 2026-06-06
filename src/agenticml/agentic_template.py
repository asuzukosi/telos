"""agenticml chat template, trajectory rendering, and reserved-wire helpers."""

from __future__ import annotations
from typing import Any, Iterable, Union
from transformers import PreTrainedTokenizerBase
from agenticml.constants import AGENTICML_TOKEN_MAP, FRAME_WIRE_MARKERS
from agenticml.frames import parse
from agenticml.trajectory import FrameLike, Trajectory

_MARKER_SUBS: tuple[tuple[str, str], ...] = tuple(
    (frame_type.value, FRAME_WIRE_MARKERS[frame_type]) for frame_type, _ in AGENTICML_TOKEN_MAP
)
_JSON_FRAME_TYPES = frozenset({"action", "result"})


def reserved_to_markers(text: str) -> str:
    """translate reserved-token model wire to human-readable agenticml markers."""
    out = text
    for marker, reserved in reversed(_MARKER_SUBS):
        if reserved in out:
            out = out.replace(reserved, marker)
    return out


def parse_reserved_wire(text: str, *, strict: bool = False):
    return parse(reserved_to_markers(text), strict=strict)


def _template_content_expr(short: str) -> str:
    if short == "end":
        return ""
    if short in _JSON_FRAME_TYPES:
        return "{{ frame['content'] | tojson }}"
    return "{{ frame['content'] or '' }}"


def _build_agentic_template() -> str:
    branches: list[str] = []
    for i, (frame_type, slot) in enumerate(AGENTICML_TOKEN_MAP):
        short = frame_type.value[2:-2]
        keyword = "if" if i == 0 else "elif"
        branches.append(
            f"{{%- {keyword} frame['type'] == '{short}' -%}}"
            f"<|reserved_special_token_{slot}|>{_template_content_expr(short)}"
        )
    return "{%- for frame in messages -%}" + "".join(branches) + "{%- endif -%}{%- endfor -%}"


AGENTIC_CHAT_TEMPLATE = _build_agentic_template()


def bake_agentic_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """attach the agenticml jinja chat template (init-embeddings --format agenticml only)."""
    tokenizer.chat_template = AGENTIC_CHAT_TEMPLATE
    return tokenizer


def _trajectory_messages(
    trajectory: Union[Trajectory, Iterable[FrameLike]],
) -> list[dict[str, Any]]:
    if isinstance(trajectory, Trajectory):
        return trajectory.to_dict()
    return Trajectory(trajectory).to_dict()


def render_trajectory(
    tokenizer: PreTrainedTokenizerBase,
    trajectory: Union[Trajectory, Iterable[FrameLike]],
) -> str:
    """render trajectory to reserved-token wire."""
    out = tokenizer.apply_chat_template(
        _trajectory_messages(trajectory),
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    if not isinstance(out, str):
        raise TypeError("apply_chat_template with tokenize=False must return str")
    return out
