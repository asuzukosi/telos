from __future__ import annotations

from typing import Callable, Optional
from agenticml.constants import MODEL_TURN_CLOSERS, TERMINAL_TOOLS, FrameType
from agenticml.frames import Frame


class Violation:
    def __init__(self, rule: str, frame_index: int, message: str):
        self.rule = rule
        self.frame_index = frame_index
        self.message = message

    def __str__(self) -> str:
        return f"[{self.rule}] frame {self.frame_index}: {self.message}"


def _action_tool(f: Frame) -> Optional[str]:
    if f.type is not FrameType.ACTION:
        return None
    return (f.content or {}).get("tool")


def _action_is_terminal(f: Frame) -> bool:
    t = _action_tool(f)
    return t is not None and t in TERMINAL_TOOLS

def _prev_non_end(frames: list[Frame], index: int) -> Optional[Frame]:
    j = index - 1
    while j >= 0 and frames[j].type is FrameType.END:
        j -= 1
    return frames[j] if j >= 0 else None


def _violations_goal_must_be_first(frames: list[Frame]) -> list[Violation]:
    if frames[0].type is not FrameType.GOAL:
        return [Violation(
            "missing_goal",
            0,
            f"trajectory must begin with <|goal|>, got {frames[0].type.value}",
        )]
    return []


def _violations_opening_prelude(frames: list[Frame]) -> list[Violation]:
    allowed = {FrameType.GOAL, FrameType.MISSION, FrameType.OBS}
    out: list[Violation] = []
    for i, f in enumerate(frames):
        if f.type in MODEL_TURN_CLOSERS:
            break
        if f.type is FrameType.END:
            out.append(Violation("opening_prelude", i, "end frame before any model frame"))
        elif f.type not in allowed:
            out.append(Violation(
                "opening_prelude",
                i,
                f"unexpected {f.type.value} before any model frame (expected goal/mission/obs)",
            ))
    return out


def _violations_end_placement(frames: list[Frame]) -> list[Violation]:
    out: list[Violation] = []
    for i, f in enumerate(frames):
        if f.type is not FrameType.END:
            continue
        if i == 0:
            out.append(Violation("misplaced_end", i, "trajectory cannot begin with end"))
        elif frames[i - 1].type not in MODEL_TURN_CLOSERS:
            out.append(Violation(
                "misplaced_end",
                i,
                f"end must follow a model turn frame, got {frames[i - 1].type.value}",
            ))
    return out


def _violations_action_result_walk(
    frames: list[Frame],
    *,
    allow_unresolved_actions_at_end: bool,
) -> list[Violation]:
    violations: list[Violation] = []
    pending_non_terminal = 0
    first_pending_action_idx: Optional[int] = None
    saw_any_action = False
    in_model_turn = False

    for i, f in enumerate(frames):
        if f.type is FrameType.END:
            in_model_turn = False
            continue

        if f.type is FrameType.ACTION:
            if pending_non_terminal > 0 and not in_model_turn:
                violations.append(Violation(
                    "orphan_action",
                    first_pending_action_idx or i,
                    "non-terminal action emitted while previous batch unresolved",
                ))
            in_model_turn = True
            saw_any_action = True
            if not _action_is_terminal(f):
                if pending_non_terminal == 0:
                    first_pending_action_idx = i
                pending_non_terminal += 1
            continue

        if f.type in (FrameType.BELIEF, FrameType.PLAN, FrameType.THINK):
            in_model_turn = True
            if pending_non_terminal > 0:
                violations.append(Violation(
                    "unresolved_action",
                    first_pending_action_idx or i,
                    (
                        "non-terminal action(s) not followed by <|result|> before "
                        f"next {f.type.value}"
                    ),
                ))
            continue

        if f.type is FrameType.RESULT:
            in_model_turn = False
            if pending_non_terminal > 0:
                pending_non_terminal -= 1
                if pending_non_terminal == 0:
                    first_pending_action_idx = None
                continue
            prev = _prev_non_end(frames, i)
            if prev is not None and prev.type is FrameType.ACTION and _action_is_terminal(prev):
                continue
            violations.append(Violation(
                "orphan_result",
                i,
                "result with no preceding unresolved non-terminal action",
            ))
            continue

        if f.type in (FrameType.FEEDBACK, FrameType.REWARD):
            in_model_turn = False
            if not saw_any_action:
                violations.append(Violation(
                    "premature_runtime_frame",
                    i,
                    f"{f.type.value} appears before any action",
                ))

    if pending_non_terminal > 0 and not allow_unresolved_actions_at_end:
        violations.append(Violation(
            "unresolved_action",
            first_pending_action_idx or (len(frames) - 1),
            (
                f"{pending_non_terminal} non-terminal action(s) without "
                f"matching result(s) by end of trajectory"
            ),
        ))
    return violations


StructuralRule = Callable[[list[Frame]], list[Violation]]

STRUCTURAL_RULES: tuple[StructuralRule, ...] = (
    _violations_goal_must_be_first,
    _violations_opening_prelude,
    _violations_end_placement,
)


def validate(
    frames: list[Frame],
    *,
    allow_unresolved_actions_at_end: bool = False,
) -> list[Violation]:
    if not frames:
        return []
    violations: list[Violation] = []
    for rule in STRUCTURAL_RULES:
        violations.extend(rule(frames))
    violations.extend(
        _violations_action_result_walk(
            frames,
            allow_unresolved_actions_at_end=allow_unresolved_actions_at_end,
        )
    )
    return violations


def is_valid(
    frames: list[Frame],
    *,
    allow_unresolved_actions_at_end: bool = False,
) -> bool:
    return not validate(
        frames, allow_unresolved_actions_at_end=allow_unresolved_actions_at_end
    )
