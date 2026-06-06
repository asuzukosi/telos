"""
trajectory:: an ordered, mutable container of agenticml frames
"""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Iterable, Union, overload
from agenticml.constants import FrameType
from agenticml.frames import Frame

FrameLike = Union[Frame, dict[str, Any]]

def _frame_from_dict(d: dict[str, Any]) -> Frame:
    if "type" not in d:
        raise ValueError(f"frame dict missing 'type' key: {d}")
    type_str = d["type"]
    if not type_str.startswith("<|"):
        type_str = "<|" + type_str + "|>"
    try:
        ft = FrameType(type_str)
    except ValueError as e:
        raise ValueError(f"unknown frame type: {type_str}") from e
    return Frame(type=ft, content=d.get("content"))

def _frame_to_dict(f: Frame) -> dict:
    return {"type": f.type.value[2:-2], "content": f.content}

def _coerce_frame(item: FrameLike) -> Frame:
    if isinstance(item, Frame):
        return item
    return _frame_from_dict(item)


def _coerce_frames(items: Iterable[FrameLike]) -> list[Frame]:
    return [_coerce_frame(f) for f in items]


class Trajectory(Sequence[Frame]):
    """an ordered, mutable sequence of AgenticML frames."""
 
    __slots__ = ("_frames",)
 
    def __init__(
        self,
        frames: Union["Trajectory", Iterable[FrameLike], None] = None,
    ):
        if frames is None:
            self._frames: list[Frame] = []
        elif isinstance(frames, Trajectory):
            self._frames = list(frames._frames)
        else:
            self._frames = _coerce_frames(frames)

    @classmethod
    def _from_frames(cls, frames: list[Frame]) -> "Trajectory":
        out = cls.__new__(cls)
        out._frames = frames
        return out

    def to_dict(self) -> list[dict]:
        return [_frame_to_dict(f) for f in self._frames]
 
    def to_frames(self) -> list[Frame]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    @overload
    def __getitem__(self, index: int) -> Frame: ...

    @overload
    def __getitem__(self, index: slice) -> "Trajectory": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, "Trajectory"]:
        if isinstance(index, slice):
            return Trajectory._from_frames(self._frames[index])
        return self._frames[index]

    def __contains__(self, item) -> bool:
        return item in self._frames
 
    def __bool__(self) -> bool:
        return bool(self._frames)

    def append(self, item: FrameLike) -> None:
        self._frames.append(_coerce_frame(item))
 
    def extend(self, items: Iterable[FrameLike]) -> None:
        self._frames.extend(_coerce_frames(items))

    def __add__(self, other) -> "Trajectory":
        frames = list(self._frames)
        if isinstance(other, Trajectory):
            frames.extend(other._frames)
        elif isinstance(other, Frame) or isinstance(other, dict):
            frames.append(_coerce_frame(other))
        elif isinstance(other, list):
            frames.extend(_coerce_frames(other))
        else:
            return NotImplemented
        return Trajectory._from_frames(frames)
 
    def __radd__(self, other) -> "Trajectory":
        if isinstance(other, Frame) or isinstance(other, dict):
            return Trajectory([other]) + self
        if isinstance(other, list):
            return Trajectory(other) + self
        return NotImplemented
 
    def __iadd__(self, other) -> "Trajectory":
        if isinstance(other, Trajectory):
            self._frames.extend(other._frames)
        elif isinstance(other, Frame) or isinstance(other, dict):
            self._frames.append(_coerce_frame(other))
        elif isinstance(other, list):
            self._frames.extend(_coerce_frames(other))
        else:
            return NotImplemented
        return self
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Trajectory):
            return self._frames == other._frames
        return NotImplemented
 
    def __repr__(self) -> str:
        return f"Trajectory({self._frames!r})"
