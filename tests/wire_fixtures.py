"""reserved wire marker strings for unit test generators."""

from agenticml.constants import FRAME_WIRE_MARKERS, FrameType, WIRE_END_MARKER

W_GOAL = FRAME_WIRE_MARKERS[FrameType.GOAL]
W_MISSION = FRAME_WIRE_MARKERS[FrameType.MISSION]
W_ACTION = FRAME_WIRE_MARKERS[FrameType.ACTION]
W_BELIEF = FRAME_WIRE_MARKERS[FrameType.BELIEF]
W_RESULT = FRAME_WIRE_MARKERS[FrameType.RESULT]
W_END = WIRE_END_MARKER
