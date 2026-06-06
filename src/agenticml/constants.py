"""
constants for the agenticml v1 format

pure python, no external dependencies
"""
from enum import Enum

class FrameType(str, Enum):
    """The 11 AgenticML frame types. The value of each member is the
    canonical wire-format marker string for that frame type.
 
    Inheriting from ``str`` means FrameType.GOAL == "<|goal|>" is True,
    which makes the enum interchangeable with the marker strings in
    places like dict keys and ``in`` checks.
    """
    GOAL     = "<|goal|>"
    MISSION  = "<|mission|>"
    OBS      = "<|obs|>"
    BELIEF   = "<|belief|>"
    PLAN     = "<|plan|>"
    THINK    = "<|think|>"
    ACTION   = "<|action|>"
    END      = "<|end|>"
    RESULT   = "<|result|>"
    FEEDBACK = "<|feedback|>"
    REWARD   = "<|reward|>"


class FrameOwner(str, Enum):
    RUNTIME = "runtime"
    MODEL = "model"

# token mapping: agenticml v1 (llama-3.1 reserved slots)
AGENTICML_TOKEN_MAP: tuple[tuple[FrameType, int], ...] = (
    (FrameType.GOAL,     0),
    (FrameType.MISSION,  1),
    (FrameType.OBS,      2),
    (FrameType.BELIEF,   3),
    (FrameType.PLAN,     4),
    (FrameType.THINK,    5),
    (FrameType.ACTION,   6),
    (FrameType.END,      7),
    (FrameType.RESULT,   8),
    (FrameType.FEEDBACK, 9),
    (FrameType.REWARD,   10),
)

AGENTICML_OWNERS: dict[FrameType, FrameOwner] = {
    FrameType.GOAL:     FrameOwner.RUNTIME,
    FrameType.MISSION:  FrameOwner.RUNTIME,
    FrameType.OBS:      FrameOwner.RUNTIME,
    FrameType.BELIEF:   FrameOwner.MODEL,
    FrameType.PLAN:     FrameOwner.MODEL,
    FrameType.THINK:    FrameOwner.MODEL,
    FrameType.ACTION:   FrameOwner.MODEL,
    FrameType.END:      FrameOwner.MODEL,
    FrameType.RESULT:   FrameOwner.RUNTIME,
    FrameType.FEEDBACK: FrameOwner.RUNTIME,
    FrameType.REWARD:   FrameOwner.RUNTIME,
}

MODEL_TURN_CLOSERS: frozenset[FrameType] = frozenset({
    FrameType.BELIEF,
    FrameType.PLAN,
    FrameType.THINK,
    FrameType.ACTION,
})


def reserved_wire_token(slot: int) -> str:
    return f"<|reserved_special_token_{slot}|>"


FRAME_WIRE_MARKERS: dict[FrameType, str] = {
    frame_type: reserved_wire_token(slot) for frame_type, slot in AGENTICML_TOKEN_MAP
}
WIRE_END_MARKER: str = FRAME_WIRE_MARKERS[FrameType.END]
MODEL_FRAMES: frozenset[FrameType] = frozenset(
    ft for ft, owner in AGENTICML_OWNERS.items() if owner == FrameOwner.MODEL
)
RUNTIME_FRAMES: frozenset[FrameType] = frozenset(
    ft for ft, owner in AGENTICML_OWNERS.items() if owner == FrameOwner.RUNTIME
)

RESERVED_SLOT_TOKEN_IDS: dict[int, int] = {
    0: 128002,
    1: 128003,
    2: 128005,
    3: 128011,
    4: 128012,
    5: 128013,
    6: 128014,
    7: 128015,
    8: 128016,
    9: 128017,
    10: 128018,
}
AGENTICML_MARKER_TOKEN_IDS: dict[FrameType, int] = {
    frame_type: RESERVED_SLOT_TOKEN_IDS[slot] for frame_type, slot in AGENTICML_TOKEN_MAP
}
END_MARKER_TOKEN_ID: int = AGENTICML_MARKER_TOKEN_IDS[FrameType.END]
MODEL_MARKER_TOKEN_IDS: frozenset[int] = frozenset(
    AGENTICML_MARKER_TOKEN_IDS[frame_type] for frame_type in MODEL_FRAMES
)
RUNTIME_MARKER_TOKEN_IDS: frozenset[int] = frozenset(
    AGENTICML_MARKER_TOKEN_IDS[frame_type] for frame_type in RUNTIME_FRAMES
)
TERMINAL_TOOLS: frozenset[str] = frozenset({"answer", "fail"})
DEFAULT_BASE_MODEL: str = "kosiasuzu/agenticml-agent-llama-3.1-8b-init"
DEFAULT_CHATML_BASE_MODEL: str = "kosiasuzu/chatml-agent-llama-3.1-8b-init"
DEFAULT_TRAJECTORY_DATASET: str = "kosiasuzu/agenticml-agent-trajectory-dataset"
DEFAULT_AGENTICML_MERGED_MODEL: str = "kosiasuzu/agenticml-llama3.1-8b-lora-merged"
DEFAULT_CHATML_MERGED_MODEL: str = "kosiasuzu/chatml-llama3.1-8b-lora-merged"
DEFAULT_CHATML_INSTRUCT_TOKENIZER: str = "meta-llama/Llama-3.1-8B-Instruct"
