"""Metadata schema definitions for ballclips video files."""

from __future__ import annotations

from implicitdict import ImplicitDict


class SquareCroppingPoint(ImplicitDict):
    """A square cropping sample taken at a specific frame.

    The coordinate system is defined on the encoded video frame (after
    applying any pixel-aspect ratio from the container but before
    considering display matrix rotations).  ``u`` and ``v`` identify the
    horizontal and vertical centre of the crop, in **pixels**, measured
    from the top-left corner of that encoded frame.  ``size`` is the edge
    length of the square crop, also in pixels, and is measured along the
    shorter dimension of the encoded frame so that a value of ``size`` =
    ``height`` indicates a crop that spans the full vertical extent of a
    landscape frame.  All values must be finite and ``size`` must be
    positive.
    """

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

