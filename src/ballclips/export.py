"""Utilities for exporting cropped ballclips videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class CropMetadata:
    """Cropping data parsed from a metadata JSON file."""

    center_x: float
    center_y: float
    size: float

    @classmethod
    def from_points(
        cls,
        width: int,
        height: int,
        points: Iterable[dict[str, object]],
    ) -> "CropMetadata" | None:
        valid: list[tuple[float, float, float]] = []
        for point in points:
            try:
                u = float(point["u"])  # type: ignore[index]
                v = float(point["v"])  # type: ignore[index]
                size = float(point["size"])  # type: ignore[index]
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(u) and math.isfinite(v) and math.isfinite(size):
                valid.append((u, v, size))
        if not valid:
            return None
        avg_u = sum(p[0] for p in valid) / len(valid)
        avg_v = sum(p[1] for p in valid) / len(valid)
        avg_size = sum(p[2] for p in valid) / len(valid)
        min_dimension = min(width, height)
        size_px = max(1.0, min(avg_size, float(min_dimension)))
        half = size_px / 2.0
        center_x = max(half, min(avg_u, width - half))
        center_y = max(half, min(avg_v, height - half))
        return cls(center_x=center_x, center_y=center_y, size=size_px)


@dataclass
class CropRegion:
    """Pixel-aligned crop region."""

    x: int
    y: int
    size: int

    @property
    def filter_expression(self) -> str:
        return f"crop={self.size}:{self.size}:{self.x}:{self.y}"


def _load_video_dimensions(path: Path) -> tuple[int, int]:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        os.fspath(path),
    ]
    process = subprocess.run(
        probe_cmd,
        capture_output=True,
        check=False,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Unable to probe '{path.name}': {process.stderr.strip() or process.stdout.strip()}"
        )
    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for '{path.name}': {exc}") from exc
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"No video stream information found for '{path.name}'.")
    stream = streams[0]
    try:
        width = int(stream["width"])  # type: ignore[index]
        height = int(stream["height"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(f"Invalid width/height metadata for '{path.name}'.")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Non-positive dimensions reported for '{path.name}'.")
    return width, height


def _load_crop_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _determine_crop_region(video_path: Path) -> CropRegion:
    width, height = _load_video_dimensions(video_path)
    metadata = _load_crop_metadata(video_path.with_suffix(".json"))
    crop_metadata: CropMetadata | None = None
    if isinstance(metadata, dict):
        square = metadata.get("square_cropping")
        if isinstance(square, dict):
            points: list[dict[str, object]] = []
            for key in ("in_point", "out_point"):
                value = square.get(key)
                if isinstance(value, dict):
                    points.append(value)
            crop_metadata = CropMetadata.from_points(width, height, points)
    if crop_metadata is None:
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        return CropRegion(x=x, y=y, size=size)
    half = crop_metadata.size / 2.0
    x = int(round(crop_metadata.center_x - half))
    y = int(round(crop_metadata.center_y - half))
    size = int(round(crop_metadata.size))
    max_x = width - size
    max_y = height - size
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    size = max(1, min(size, min(width, height)))
    return CropRegion(x=x, y=y, size=size)


def _build_encoding_command(
    source: Path,
    destination: Path,
    crop_region: CropRegion,
) -> list[str]:
    filter_chain = ",".join(
        [crop_region.filter_expression, "scale=480:480:flags=lanczos", "setsar=1"]
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        os.fspath(source),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=cl=stereo:r=44100",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        filter_chain,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-level:v",
        "4.2",
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-refs",
        "1",
        "-g",
        "60",
        "-b:v",
        "1100k",
        "-maxrate",
        "1100k",
        "-bufsize",
        "2200k",
        "-c:a",
        "aac",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-shortest",
        os.fspath(destination),
    ]


def _encode_segments(video_files: Sequence[Path], tmpdir: Path) -> list[Path]:
    segments: list[Path] = []
    for index, video_path in enumerate(video_files, start=1):
        crop_region = _determine_crop_region(video_path)
        segment_path = tmpdir / f"segment_{index:03d}.mp4"
        cmd = _build_encoding_command(video_path, segment_path, crop_region)
        process = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    [
                        f"Failed to encode '{video_path.name}'.",
                        process.stderr.strip() or process.stdout.strip(),
                    ]
                )
            )
        segments.append(segment_path)
    return segments


def _concat_segments(segments: Sequence[Path], output_path: Path) -> None:
    if not segments:
        raise ValueError("At least one segment is required to concatenate.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        list_path = Path(handle.name)
        for segment in segments:
            handle.write(f"file '{segment.as_posix()}'\n")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            os.fspath(list_path),
            "-c",
            "copy",
            os.fspath(output_path),
        ]
        process = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    [
                        "Failed to concatenate segments.",
                        process.stderr.strip() or process.stdout.strip(),
                    ]
                )
            )
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass


def _list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder '{folder}' does not exist or is not a directory.")
    video_files = [path for path in folder.iterdir() if path.suffix.lower() == ".mp4"]
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in '{folder}'.")
    return video_files


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export cropped videos from a folder into a single compilation.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the folder containing input MP4 files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ballclips_compilation.mp4"),
        help="Output video file (default: ballclips_compilation.mp4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used to shuffle the video order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    folder: Path = args.folder
    output_path: Path = args.output
    seed = args.seed

    video_files = _list_videos(folder)
    if seed is not None:
        random.Random(seed).shuffle(video_files)
    else:
        random.shuffle(video_files)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        segments = _encode_segments(video_files, tmpdir)
        _concat_segments(segments, output_path)

    print(f"Created {output_path} from {len(video_files)} videos.")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    sys.exit(main())
