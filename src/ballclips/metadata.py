"""Metadata schema definitions for ballclips video files."""

from __future__ import annotations

from implicitdict import ImplicitDict


class SquareCroppingPoint(ImplicitDict):
    """A square cropping sample taken at a specific frame."""

    frame: int
    u: int
    v: int
    size: int


class SquareCroppingData(ImplicitDict):
    """Cropping samples describing the in/out regions for a clip."""

    in_point: SquareCroppingPoint
    out_point: SquareCroppingPoint


class VideoMetadata(ImplicitDict):
    """Top-level metadata stored alongside a captured video."""

    square_cropping: SquareCroppingData


__all__ = [
    "SquareCroppingPoint",
    "SquareCroppingData",
    "VideoMetadata",
]

