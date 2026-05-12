"""
contstant for the telos v1 format

pure python, no external dependencies
"""
from enum import Enum

class FrameType(str, Enum):
    """The 11 Telos frame types. The value of each member is the
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

# token mapping: telos v1
TELOS_TOKEN_MAP: tuple[tuple[FrameType, int], ...] = (
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
 
# ownership: which side of the loop is allowed to emit each marker.
TELOS_OWNERS: dict[FrameType, FrameOwner] = {
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
 
DEFAULT_BASE_MODEL: str = "meta-llama/Llama-3.1-8B"