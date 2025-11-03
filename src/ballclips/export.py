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
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from implicitdict import ImplicitDict

from .metadata import SquareCroppingPoint, VideoMetadata


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
        points: Iterable[SquareCroppingPoint],
    ) -> "CropMetadata" | None:
        valid: list[tuple[float, float, float]] = []
        for point in points:
            try:
                u = float(point.u)
                v = float(point.v)
                size = float(point.size)
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(u) and math.isfinite(v) and math.isfinite(size):
                valid.append((u, v, size))
        if not valid:
            return None
        avg_u = sum(p[0] for p in valid) / len(valid)
        avg_v = sum(p[1] for p in valid) / len(valid)
        avg_size = sum(p[2] for p in valid) / len(valid)
        size_px = avg_size
        center_x = avg_u
        center_y = avg_v
        if not (
            math.isfinite(center_x)
            and math.isfinite(center_y)
            and math.isfinite(size_px)
        ):
            return None
        return cls(center_x=center_x, center_y=center_y, size=size_px)

    @classmethod
    def from_point(
        cls,
        width: int,
        height: int,
        point: SquareCroppingPoint,
    ) -> "CropMetadata" | None:
        return cls.from_points(width, height, [point])


@dataclass
class CropRegion:
    """Pixel-aligned crop region."""

    x: int
    y: int
    size: int

    @property
    def filter_expression(self) -> str:
        return f"crop={self.size}:{self.size}:{self.x}:{self.y}"


@dataclass
class CropFilterSpec:
    """Specification describing how to crop a video frame."""

    filters: list[str]
    time_variant: bool = False
    produces_scaled_output: bool = False


@dataclass
class CropSummary:
    """Human-readable summary of the crop that will be applied."""

    width: int
    height: int
    start_rect: tuple[int, int, int, int]
    end_rect: tuple[int, int, int, int]
    start_frame: int | None = None
    end_frame: int | None = None

    def format_summary(self) -> str:
        dims = f"({self.width}x{self.height})"
        start_str = (
            f"({self.start_rect[0]}, {self.start_rect[1]})-({self.start_rect[2]}, {self.start_rect[3]})"
        )
        end_str = (
            f"({self.end_rect[0]}, {self.end_rect[1]})-({self.end_rect[2]}, {self.end_rect[3]})"
        )

        parts: list[str] = [dims, "from", start_str]
        if self.start_frame is not None:
            parts.append(f"at frame {self.start_frame}")

        if self.start_rect != self.end_rect or self.start_frame != self.end_frame:
            parts.extend(["to", end_str])
            if self.end_frame is not None:
                parts.append(f"at frame {self.end_frame}")
        else:
            parts.append("(static crop)")

        return " ".join(parts)


@dataclass
class VideoProbeInfo:
    """Information gathered from probing a video file."""

    width: int
    height: int
    duration: float | None
    frame_rate: float | None


def _parse_ratio(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
        except ValueError:
            return None
        if den == 0:
            return None
        result = num / den
    else:
        try:
            result = float(value)
        except ValueError:
            return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _probe_video_info(path: Path) -> VideoProbeInfo:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration",
        "-show_entries",
        "format=duration",
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
    frame_rate = None
    try:
        frame_rate = _parse_ratio(stream.get("avg_frame_rate"))  # type: ignore[arg-type]
    except AttributeError:
        frame_rate = None
    duration: float | None = None
    raw_stream_duration = stream.get("duration") if isinstance(stream, dict) else None
    raw_format = data.get("format")
    raw_format_duration = None
    if isinstance(raw_format, dict):
        raw_format_duration = raw_format.get("duration")
    for candidate in (raw_stream_duration, raw_format_duration):
        if candidate is None:
            continue
        try:
            duration_val = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration_val) and duration_val > 0:
            duration = duration_val
            break
    return VideoProbeInfo(width=width, height=height, duration=duration, frame_rate=frame_rate)


def _load_video_metadata(path: Path) -> VideoMetadata | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return ImplicitDict.parse(raw_metadata, VideoMetadata)
    except (TypeError, ValueError):
        return None


def _determine_crop_region(video_path: Path, info: VideoProbeInfo) -> CropRegion:
    width, height = info.width, info.height
    metadata = _load_video_metadata(video_path.with_suffix(".json"))
    crop_metadata: CropMetadata | None = None
    if metadata is not None:
        crop_metadata = CropMetadata.from_points(
            width,
            height,
            [
                metadata.square_cropping.in_point,
                metadata.square_cropping.out_point,
            ],
        )
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


def _format_ffmpeg_float(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-9):
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-"}:
        return "0"
    if text == "-0":
        return "0"
    return text


def _escape_ffmpeg_expr(expression: str) -> str:
    """Escape characters in an ffmpeg filter expression."""

    # Commas separate filters/arguments, so escape them to treat them as literals.
    return expression.replace(",", r"\,")


def _validate_crop_metadata(
    metadata: CropMetadata,
    width: int,
    height: int,
    label: str,
    fail: Callable[[str], RuntimeError],
) -> tuple[float, float, float]:
    if not math.isfinite(metadata.size):
        raise fail(f"its '{label}' crop size is not finite.")
    if metadata.size <= 0:
        raise fail(f"its '{label}' crop size is not positive.")
    if not math.isfinite(metadata.center_x) or not math.isfinite(metadata.center_y):
        raise fail(f"its '{label}' crop position is not finite.")

    half = metadata.size / 2.0
    left = metadata.center_x - half
    top = metadata.center_y - half
    right = metadata.center_x + half
    bottom = metadata.center_y + half

    if left < 0 or top < 0 or right > width or bottom > height:
        raise fail(
            f"its '{label}' crop window extends outside the video frame ({width}x{height})."
        )

    return left, top, metadata.size


def _extract_point_time(
    point: SquareCroppingPoint, frame_rate: float | None
) -> float | None:
    frame_value = point.frame
    try:
        frame_float = float(frame_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(frame_float):
        return None
    return _frame_to_seconds(frame_float, frame_rate)


def _determine_crop_filter(
    video_path: Path,
    info: VideoProbeInfo,
    trim_range: tuple[float, float] | None,
) -> tuple[CropFilterSpec, CropSummary]:
    fallback_region = _determine_crop_region(video_path, info)
    fallback_spec = CropFilterSpec(
        filters=[fallback_region.filter_expression],
        time_variant=False,
    )
    fallback_rect = (
        fallback_region.x,
        fallback_region.y,
        fallback_region.x + fallback_region.size,
        fallback_region.y + fallback_region.size,
    )
    fallback_summary = CropSummary(
        width=info.width,
        height=info.height,
        start_rect=fallback_rect,
        end_rect=fallback_rect,
    )

    metadata = _load_video_metadata(video_path.with_suffix(".json"))
    if metadata is None:
        return fallback_spec, fallback_summary

    def _fail(reason: str) -> RuntimeError:
        return RuntimeError(
            f"{video_path.name} has square_cropping metadata but {reason}"
        )

    in_point_obj = metadata.square_cropping.in_point
    out_point_obj = metadata.square_cropping.out_point

    in_metadata = CropMetadata.from_point(info.width, info.height, in_point_obj)
    out_metadata = CropMetadata.from_point(info.width, info.height, out_point_obj)
    if in_metadata is None:
        raise _fail("its 'in_point' metadata is incomplete or invalid.")
    if out_metadata is None:
        raise _fail("its 'out_point' metadata is incomplete or invalid.")

    start_time = _extract_point_time(in_point_obj, info.frame_rate)
    end_time = _extract_point_time(out_point_obj, info.frame_rate)
    if start_time is None:
        raise _fail("its 'in_point' does not define a usable frame/time.")
    if end_time is None:
        raise _fail("its 'out_point' does not define a usable frame/time.")
    if end_time <= start_time:
        raise _fail("its 'out_point' occurs on or before its 'in_point'.")

    start_x, start_y, start_size = _validate_crop_metadata(
        in_metadata, info.width, info.height, "in_point", _fail
    )
    end_x, end_y, end_size = _validate_crop_metadata(
        out_metadata, info.width, info.height, "out_point", _fail
    )

    trim_start = trim_range[0] if trim_range is not None else 0.0
    offset = start_time - trim_start
    duration = max(end_time - start_time, 1e-6)

    offset_str = _format_ffmpeg_float(offset)
    duration_str = _format_ffmpeg_float(duration)

    # Guard against ffmpeg's filter graph initialization evaluating expressions before
    # presentation timestamps are defined by treating NaN timestamps as time zero and
    # constraining the normalized progress to the [0, 1] range.
    progress_expr = (
        f"clip(if(isnan(t),0,((t)-({offset_str}))/{duration_str}),0,1)"
    )

    def _interpolated_expr(start: float, end: float) -> str:
        start_str = _format_ffmpeg_float(start)
        delta = end - start
        if math.isclose(delta, 0.0, abs_tol=1e-9):
            return start_str
        delta_str = _format_ffmpeg_float(delta)
        return f"({start_str})+({delta_str})*{progress_expr}"

    size_expr_raw = _interpolated_expr(start_size, end_size)
    x_expr_raw = _interpolated_expr(start_x, end_x)
    y_expr_raw = _interpolated_expr(start_y, end_y)

    crop_filter = (
        "crop="
        f"w='{_escape_ffmpeg_expr(size_expr_raw)}':"
        f"h='{_escape_ffmpeg_expr(size_expr_raw)}':"
        f"x='{_escape_ffmpeg_expr(x_expr_raw)}':"
        f"y='{_escape_ffmpeg_expr(y_expr_raw)}'"
    )

    def _round_rect(x: float, y: float, size: float) -> tuple[int, int, int, int]:
        left = int(round(x))
        top = int(round(y))
        right = int(round(x + size))
        bottom = int(round(y + size))
        return left, top, right, bottom

    def _extract_frame_value(point: SquareCroppingPoint) -> int | None:
        try:
            value = int(point.frame)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            return None
        return value

    spec = CropFilterSpec(
        filters=[crop_filter, "fps=30", "scale=480:480:flags=lanczos"],
        time_variant=True,
        produces_scaled_output=True,
    )
    summary = CropSummary(
        width=info.width,
        height=info.height,
        start_rect=_round_rect(start_x, start_y, start_size),
        end_rect=_round_rect(end_x, end_y, end_size),
        start_frame=_extract_frame_value(in_point_obj),
        end_frame=_extract_frame_value(out_point_obj),
    )
    return spec, summary


def _frame_to_seconds(frame_value: float, frame_rate: float | None) -> float:
    if frame_rate is None or frame_rate <= 0:
        return max(0.0, frame_value) / 1000.0
    return max(0.0, frame_value) / frame_rate


def _determine_trim_range(video_path: Path, info: VideoProbeInfo) -> tuple[float, float] | None:
    metadata = _load_video_metadata(video_path.with_suffix(".json"))
    if metadata is None:
        return None

    start = _extract_point_time(metadata.square_cropping.in_point, info.frame_rate)
    end = _extract_point_time(metadata.square_cropping.out_point, info.frame_rate)
    if start is None or end is None:
        return None
    if info.duration is not None:
        start = max(0.0, min(start, info.duration))
        end = max(start, min(end, info.duration))
    if end <= start:
        return None
    return start, end


def _build_encoding_command(
    source: Path,
    destination: Path,
    crop_filter: CropFilterSpec,
    trim_range: tuple[float, float] | None,
) -> list[str]:
    filter_parts = []
    if crop_filter.time_variant:
        filter_parts.append("setpts=PTS-STARTPTS")
    filter_parts.extend(crop_filter.filters)
    if not crop_filter.produces_scaled_output:
        filter_parts.append("scale=480:480:flags=lanczos")
    filter_parts.append("setsar=1")
    filter_chain = ",".join(filter_parts)
    command = [
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
    if trim_range is not None:
        start, end = trim_range
        duration = max(0.0, end - start)
        if duration > 0:
            insertion_index = command.index("-vf")
            command[insertion_index:insertion_index] = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
    return command


def _encode_segments(
    video_files: Sequence[Path], tmpdir: Path, show_ffmpeg: bool = False
) -> list[Path]:
    segments: list[Path] = []
    total = len(video_files)
    for index, video_path in enumerate(video_files, start=1):
        info = _probe_video_info(video_path)
        trim_range = _determine_trim_range(video_path, info)
        crop_filter, crop_summary = _determine_crop_filter(
            video_path, info, trim_range
        )
        segment_path = tmpdir / f"segment_{index:03d}.mp4"
        summary_text = crop_summary.format_summary()
        print(
            f"[{index}/{total}] Encoding {video_path.name} {summary_text}...",
            flush=True,
        )
        cmd = _build_encoding_command(video_path, segment_path, crop_filter, trim_range)
        if show_ffmpeg:
            print(shlex.join(cmd), flush=True)
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
        print("Concatenating segments...", flush=True)
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
    parser.add_argument(
        "--show-ffmpeg",
        action="store_true",
        help="Print the full ffmpeg command for each segment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    folder: Path = args.folder
    output_path: Path = args.output
    seed = args.seed
    show_ffmpeg: bool = args.show_ffmpeg

    video_files = _list_videos(folder)
    if seed is not None:
        random.Random(seed).shuffle(video_files)
    else:
        random.shuffle(video_files)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        segments = _encode_segments(video_files, tmpdir, show_ffmpeg=show_ffmpeg)
        _concat_segments(segments, output_path)

    print(f"Created {output_path} from {len(video_files)} videos.")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    sys.exit(main())
