from __future__ import annotations
 
from dataclasses import dataclass
from typing import Optional
 
from telos.constants import FrameType, FrameOwner
from telos.frames import Frame
 
 
@dataclass
class Violation:
    """a single trajectory-level rule violation."""
    rule: str
    frame_index: int
    message: str
 
    def __str__(self) -> str:
        return f"[{self.rule}] frame {self.frame_index}: {self.message}"
 
 
def validate(frames: list[Frame]) -> list[Violation]:
    """check a trajectory against the telos wire format sequence rules.
    returns a list of violations. empty list means the trajectory is structurally valid. the caller decides how to react.
    """
    violations: list[Violation] = []
 
    if not frames:
        return violations

    # rule 0: <|goal|> must be the very first frame.
    if frames[0].type is not FrameType.GOAL:
        violations.append(Violation(
            rule="missing_goal",
            frame_index=0,
            message=f"trajectory must begin with <|goal|>, got {frames[0].type.value}",
        ))
 
 
    # rule 1: opening prelude
    # find the first model-owned frame. every frame before it must be one of goal/mission/obs.
    allowed_prelude = {FrameType.GOAL, FrameType.MISSION, FrameType.OBS}
    for i, f in enumerate(frames):
        if f.owner == FrameOwner.MODEL:
            break
        if f.type not in allowed_prelude:
            violations.append(Violation(
                rule="opening_prelude",
                frame_index=i,
                message=f"unexpected {f.type.value} before any model frame "
                        f"(expected goal/mission/obs)",
            ))
    
    # walk the trajectory checking action/result balance.
    pending_actions = 0                # actions emitted but not yet resulted
    first_pending_action_idx: Optional[int] = None
    saw_any_action = False            # whether any action has been seen
    in_model_block = False            # whether we are in a model block
    block_has_non_end_frame = False   # whether the current block has a model frame other than <|end|>
    model_block_ended_with_end = False # whether the current model block ended with <|end|>
 
    for i, f in enumerate(frames):
        if f.owner == FrameOwner.MODEL:
            if not in_model_block:
                in_model_block = True
                block_has_non_end_frame = False
                model_block_ended_with_end = False
        else:
            if in_model_block:
                if not model_block_ended_with_end:
                    violations.append(Violation(
                        rule="block_termination",
                        frame_index=i - 1,
                        message="model generation block did not end with <|end|>",
                    ))
                in_model_block = False
                block_has_non_end_frame = False
                model_block_ended_with_end = False
 
        # rule 3: the model may emit multiple actions in one block; the following runtime block must emit a result for each. a new action emitted while a previous action is still unresolved is only valid if we are still in the same model block.
        if f.type is FrameType.ACTION:
            if pending_actions > 0 and not in_model_block:
                # an action arrived while previous batch is unresolved and we are not in a model block - structurally impossible (action is model-owned) but kept for safety.
                violations.append(Violation(
                    rule="orphan_action",
                    frame_index=first_pending_action_idx or i,
                    message="action emitted while previous batch unresolved",
                ))
            if pending_actions == 0:
                first_pending_action_idx = i
            pending_actions += 1
            saw_any_action = True
 
        if f.type is FrameType.RESULT:
            if pending_actions == 0:
                violations.append(Violation(
                    rule="orphan_result",
                    frame_index=i,
                    message="result with no preceding unresolved action",
                ))
            else:
                pending_actions -= 1
                if pending_actions == 0:
                    first_pending_action_idx = None
 
        # rule 5: feedback/reward only after at least one action exists.
        if f.type in (FrameType.FEEDBACK, FrameType.REWARD):
            if not saw_any_action:
                violations.append(Violation(
                    rule="premature_runtime_frame",
                    frame_index=i,
                    message=f"{f.type.value} appears before any action",
                ))
 
        # track whether the current model block contains any non-end frame.
        if f.owner == FrameOwner.MODEL and f.type is not FrameType.END:
            block_has_non_end_frame = True
 
        # track <|end|> as the close of the current model block.
        if f.type is FrameType.END:
            if not block_has_non_end_frame:
                violations.append(Violation(
                    rule="stray_end",
                    frame_index=i,
                    message="<|end|> with no preceding model frame in its block",
                ))
            else:
                model_block_ended_with_end = True
 
    # handle the case where the trajectory ends inside an open model block.
    if in_model_block and not model_block_ended_with_end:
        violations.append(Violation(
            rule="block_termination",
            frame_index=len(frames) - 1,
            message="trajectory ends inside an unterminated model block",
        ))
 
    # trailing unresolved actions.
    if pending_actions > 0:
        violations.append(Violation(
            rule="unresolved_action",
            frame_index=first_pending_action_idx or (len(frames) - 1),
            message=(
                f"{pending_actions} action(s) without corresponding result(s) "
                f"by end of trajectory"
            ),
        ))
 
    return violations
 
def is_valid(frames: list[Frame]) -> bool:
    """convenience: true iff validate returns no violations. used for testing."""
    return not validate(frames)
 