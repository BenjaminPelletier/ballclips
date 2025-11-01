"""Application entry point for the ballclips prototype UI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Sequence, TYPE_CHECKING

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Gst", "1.0")

try:
    gi.require_version("GstPbutils", "1.0")
except (ValueError, ImportError):
    _GST_PBUTILS_AVAILABLE = False
else:
    _GST_PBUTILS_AVAILABLE = True

from gi.repository import Gdk, GLib, Gst, Gtk, Pango, cairo

from implicitdict import ImplicitDict

if _GST_PBUTILS_AVAILABLE:
    try:
        from gi.repository import GstPbutils  # type: ignore
    except ImportError:  # pragma: no cover - depends on system packages
        GstPbutils = None  # type: ignore[assignment]
        _GST_PBUTILS_AVAILABLE = False
else:
    GstPbutils = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from gi.repository import GstPbutils as GstPbutilsTypes  # pragma: no cover
else:
    GstPbutilsTypes = object

_PBUTILS_WARNING_EMITTED = False


@dataclass
class CropRegion:
    center_x: float
    center_y: float
    size: float

    def clamped(self) -> "CropRegion":
        return CropRegion(
            center_x=min(max(self.center_x, 0.0), 1.0),
            center_y=min(max(self.center_y, 0.0), 1.0),
            size=max(0.0, min(self.size, 1.0)),
        )

    def to_rectangle(
        self,
        width: float,
        height: float,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> tuple[float, float, float]:
        min_dimension = max(1.0, min(width, height))
        size_px = max(0.0, self.size) * min_dimension
        center_x_px = offset_x + self.center_x * width
        center_y_px = offset_y + self.center_y * height
        half = size_px / 2.0
        left = center_x_px - half
        top = center_y_px - half
        left = max(offset_x, min(left, offset_x + width - size_px))
        top = max(offset_y, min(top, offset_y + height - size_px))
        return left, top, size_px


class SquareCroppingPoint(ImplicitDict):
    frame: int
    u: int
    v: int
    size: int


class SquareCroppingData(ImplicitDict):
    in_point: SquareCroppingPoint
    out_point: SquareCroppingPoint


class VideoMetadata(ImplicitDict):
    square_cropping: SquareCroppingData


def _coerce_fraction(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return int(value.numerator), int(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        try:
            return int(num), int(den)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return int(value.numerator), int(value.denominator)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if hasattr(value, "num") and hasattr(value, "den"):
        try:
            return int(value.num), int(value.den)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if hasattr(value, "n") and hasattr(value, "d"):
        try:
            return int(value.n), int(value.d)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def _structure_fraction(structure: Gst.Structure, field: str) -> tuple[int, int] | None:
    if not structure.has_field(field):
        return None
    try:
        success, numerator, denominator = structure.get_fraction(field)
    except TypeError:
        value = structure.get_value(field)
        return _coerce_fraction(value)
    else:
        if success:
            return int(numerator), int(denominator)
        return None


def _list_mp4_files(folder: Path) -> list[Path]:
    files = [path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() == ".mp4"]
    if not files:
        raise FileNotFoundError("No MP4 files found in the provided folder.")
    return files


class PlayerWindow(Gtk.ApplicationWindow):
    """Main window that embeds a GStreamer-backed video player."""

    def __init__(
        self,
        app: "BallclipsApplication",
        video_files: Sequence[Path],
        start_index: int,
    ) -> None:
        super().__init__(application=app)
        self.set_title("ballclips preview")
        self.set_default_size(960, 540)

        self._app = app
        self._video_files = list(video_files)
        if not self._video_files:
            raise ValueError("PlayerWindow requires at least one video file.")
        self._current_index = start_index % len(self._video_files)

        self._player = Gst.ElementFactory.make("playbin", "player")
        if self._player is None:
            raise RuntimeError("Unable to create GStreamer playbin element.")

        sink, video_widget = self._create_video_sink()

        self._player.set_property("video-sink", sink)
        video_widget.set_hexpand(True)
        video_widget.set_vexpand(True)

        video_overlay = Gtk.Overlay()
        video_overlay.set_hexpand(True)
        video_overlay.set_vexpand(True)
        video_overlay.add(video_widget)

        crop_overlay = Gtk.DrawingArea()
        crop_overlay.set_hexpand(True)
        crop_overlay.set_vexpand(True)
        crop_overlay.set_halign(Gtk.Align.FILL)
        crop_overlay.set_valign(Gtk.Align.FILL)
        crop_overlay.set_app_paintable(True)
        crop_overlay.connect("draw", self._on_crop_overlay_draw)
        crop_overlay.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
        )
        crop_overlay.connect("button-press-event", self._on_crop_button_press)
        crop_overlay.connect("button-release-event", self._on_crop_button_release)
        crop_overlay.connect("motion-notify-event", self._on_crop_motion)
        video_overlay.add_overlay(crop_overlay)
        video_overlay.set_overlay_pass_through(crop_overlay, False)

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        container.set_border_width(12)
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        prev_button = Gtk.Button.new_from_icon_name("go-previous", Gtk.IconSize.BUTTON)
        prev_button.connect("clicked", self._on_prev_clicked)
        header.pack_start(prev_button, False, False, 0)

        title_label = Gtk.Label()
        title_label.set_hexpand(True)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.set_max_width_chars(60)
        title_label.set_halign(Gtk.Align.CENTER)
        title_label.set_xalign(0.5)
        header.pack_start(title_label, True, True, 0)

        next_button = Gtk.Button.new_from_icon_name("go-next", Gtk.IconSize.BUTTON)
        next_button.connect("clicked", self._on_next_clicked)
        header.pack_start(next_button, False, False, 0)

        self._title_label = title_label

        container.pack_start(header, False, False, 0)
        container.pack_start(video_overlay, True, True, 0)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        play_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        controls.pack_start(play_column, False, False, 0)

        play_pause_button = Gtk.Button.new_from_icon_name(
            "media-playback-pause", Gtk.IconSize.BUTTON
        )
        play_pause_button.connect("clicked", self._on_play_pause_clicked)
        play_column.pack_start(play_pause_button, False, False, 0)

        maximize_button = Gtk.Button(label="✥")
        maximize_button.connect("clicked", self._on_maximize_crop_clicked)
        play_column.pack_start(maximize_button, False, False, 0)

        progress_adjustment = Gtk.Adjustment(0.0, 0.0, 1.0, 0.1, 1.0, 0.0)
        progress_scale = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, progress_adjustment)
        progress_scale.set_hexpand(True)
        progress_scale.set_draw_value(False)
        progress_scale.connect("value-changed", self._on_progress_changed)

        goto_in_button = Gtk.Button(label="→{")
        goto_in_button.connect("clicked", self._on_go_to_in_clicked)

        set_in_button = Gtk.Button(label="{")
        set_in_button.connect("clicked", self._on_set_in_clicked)

        set_out_button = Gtk.Button(label="}")
        set_out_button.connect("clicked", self._on_set_out_clicked)

        goto_out_button = Gtk.Button(label="→}")
        goto_out_button.connect("clicked", self._on_go_to_out_clicked)

        trim_area = Gtk.DrawingArea()
        trim_area.set_hexpand(True)
        trim_area.set_size_request(-1, 30)
        trim_area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
        )
        trim_area.connect("draw", self._on_trim_area_draw)
        trim_area.connect("button-press-event", self._on_trim_button_press)
        trim_area.connect("button-release-event", self._on_trim_button_release)
        trim_area.connect("motion-notify-event", self._on_trim_motion)

        timeline_grid = Gtk.Grid()
        timeline_grid.set_row_spacing(6)
        timeline_grid.set_column_spacing(6)
        timeline_grid.set_hexpand(True)

        timeline_grid.attach(goto_in_button, 0, 0, 1, 1)
        timeline_grid.attach(progress_scale, 1, 0, 1, 1)
        timeline_grid.attach(goto_out_button, 2, 0, 1, 1)

        timeline_grid.attach(set_in_button, 0, 1, 1, 1)
        timeline_grid.attach(trim_area, 1, 1, 1, 1)
        timeline_grid.attach(set_out_button, 2, 1, 1, 1)

        button_width_group = Gtk.SizeGroup(Gtk.SizeGroupMode.HORIZONTAL)
        button_width_group.add_widget(goto_in_button)
        button_width_group.add_widget(set_in_button)
        button_width_group.add_widget(goto_out_button)
        button_width_group.add_widget(set_out_button)

        controls.pack_start(timeline_grid, True, True, 0)

        container.pack_start(controls, False, False, 0)
        self.add(container)
        self.show_all()

        self.connect("destroy", self._on_destroy)

        bus = self._player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._play_pause_button = play_pause_button
        self._progress_adjustment = progress_adjustment
        self._progress_scale = progress_scale
        self._trim_area = trim_area
        self._duration_ns = 0
        self._updating_progress = False
        self._is_playing = True
        self._progress_update_id = GLib.timeout_add(200, self._update_progress)
        self._trim_in_seconds = 0.0
        self._trim_out_seconds = 0.0
        self._active_trim: Literal["in", "out"] | None = None
        self._pending_trim_reset = True

        self._crop_overlay = crop_overlay
        self._crop_dragging = False
        self._crop_drag_button: int | None = None
        self._crop_drag_start: tuple[float, float] | None = None
        self._crop_drag_current: tuple[float, float] | None = None
        self._crop_drag_key: Literal["in", "out"] | None = None
        self._crop_initial_region: CropRegion | None = None
        self._crop_edit_tolerance = 1e-3
        self._video_crop_regions: dict[Path, dict[str, CropRegion]] = {}
        self._current_video_file: Path | None = None
        self._discoverer: GstPbutilsTypes.Discoverer | None = None  # type: ignore[attr-defined]
        self._discoverer_failed = not _GST_PBUTILS_AVAILABLE
        self._current_video_pixel_size: tuple[float, float] | None = None
        self._current_video_frame_rate: tuple[int, int] | None = None

        self._load_video(self._current_index)

    def _create_video_sink(self) -> tuple[Gst.Element, Gtk.Widget]:
        """Choose a GTK sink that keeps frames on the GPU when possible."""

        # Prefer a GL-backed sink so the video can be rendered by the GPU when
        # available, but wrap it in glsinkbin so software decoders can still
        # negotiate with a conventional video/x-raw surface.  If the GL stack is
        # missing we fall back to gtksink so playback keeps working everywhere.

        failure_reasons: list[str] = []

        def _build_gtkglsink() -> tuple[Gst.Element, Gtk.Widget] | None:
            gtkgl = Gst.ElementFactory.make("gtkglsink", "gtkglsink")
            if gtkgl is None:
                failure_reasons.append(
                    "gtkglsink plugin is unavailable. Install the GStreamer GL/gtk plugins (e.g. gst-plugins-bad) and ensure the VM has 3D acceleration enabled."
                )
                return None
            widget = getattr(gtkgl.props, "widget", None)
            if not isinstance(widget, Gtk.Widget):
                failure_reasons.append(
                    "gtkglsink did not expose a Gtk.Widget. Verify gstreamer1.0-gtk3 is installed and up to date."
                )
                return None

            glbin = Gst.ElementFactory.make("glsinkbin", "gtkglsink_bin")
            if glbin is None:
                failure_reasons.append(
                    "glsinkbin plugin is unavailable. Install the GStreamer GL plugins to enable GPU-backed rendering."
                )
                return None
            glbin.set_property("sink", gtkgl)
            return glbin, widget

        builders = (_build_gtkglsink,)
        for build in builders:
            result = build()
            if result is not None:
                return result

        # Fallback: CPU-bound GTK sink.
        gtksink = Gst.ElementFactory.make("gtksink", "gtksink")
        if gtksink is not None:
            widget = getattr(gtksink.props, "widget", None)
            if isinstance(widget, Gtk.Widget):
                if not failure_reasons:
                    failure_reasons.append(
                        "gtkglsink could not be initialised for an unknown reason. Check that gstreamer1.0-plugins-bad and gstreamer1.0-gtk3 are installed."
                    )
                reason_text = "\n".join(f"- {reason}" for reason in failure_reasons)
                print(
                    "Using CPU-bound gtksink for video playback. Expect higher CPU usage and possible stuttering compared to VLC, which can keep frames on the GPU.\n"
                    "Diagnostic hints:\n"
                    f"{reason_text}\n"
                    "If you are running inside VirtualBox, double-check that Guest Additions and 3D acceleration are active.",
                    file=sys.stderr,
                )
                return gtksink, widget

        hint = "; ".join(failure_reasons) if failure_reasons else "unknown reason"
        raise RuntimeError(
            "No suitable GTK video sink is available. Install the GStreamer GTK plugins."
            f" ({hint})"
        )

    def _load_video(self, index: int) -> None:
        if not self._video_files:
            return

        self._current_index = index % len(self._video_files)
        video_file = self._video_files[self._current_index]
        self._current_video_file = video_file
        self._current_video_pixel_size = self._probe_video_size(video_file)
        self._ensure_crop_defaults(video_file)
        uri = Gst.filename_to_uri(str(video_file.resolve()))

        self._title_label.set_text(video_file.name)

        self._player.set_state(Gst.State.READY)
        self._player.set_property("uri", uri)
        self._player.set_state(Gst.State.PLAYING)

        self._is_playing = True
        self._update_play_pause_icon()
        self._duration_ns = 0
        self._progress_adjustment.set_upper(1.0)
        self._set_progress_value(0.0)
        self._trim_in_seconds = 0.0
        self._trim_out_seconds = 0.0
        self._pending_trim_reset = True
        self._trim_area.queue_draw()
        self._crop_overlay.queue_draw()

        self._app.set_current_index(self._current_index)

    def _ensure_discoverer(self) -> GstPbutilsTypes.Discoverer | None:  # type: ignore[attr-defined]
        if not _GST_PBUTILS_AVAILABLE:
            return None
        if self._discoverer is not None or self._discoverer_failed:
            return self._discoverer
        try:
            self._discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)  # type: ignore[call-arg]
        except GLib.Error as exc:
            print(
                f"Failed to initialise GStreamer discoverer: {exc.message}",
                file=sys.stderr,
            )
            self._discoverer_failed = True
            self._discoverer = None
        return self._discoverer

    def _probe_video_size(self, video_file: Path) -> tuple[float, float] | None:
        self._current_video_frame_rate = None
        discoverer = self._ensure_discoverer()
        if discoverer is not None and GstPbutils is not None:
            try:
                info = discoverer.discover_uri(
                    Gst.filename_to_uri(str(video_file.resolve()))
                )
            except GLib.Error as exc:
                print(
                    f"Unable to inspect video metadata for '{video_file.name}': {exc.message}",
                    file=sys.stderr,
                )
            else:
                for stream in info.get_video_streams():
                    if not isinstance(stream, GstPbutils.DiscovererVideoInfo):
                        continue
                    width = float(stream.get_width())
                    height = float(stream.get_height())
                    if width <= 0.0 or height <= 0.0:
                        continue
                    fr_num = stream.get_framerate_num()
                    fr_den = stream.get_framerate_den()
                    if fr_num > 0 and fr_den > 0:
                        self._current_video_frame_rate = (int(fr_num), int(fr_den))
                    par_num = stream.get_par_num()
                    par_den = stream.get_par_den()
                    if par_num > 0 and par_den > 0:
                        width *= par_num / par_den
                    return width, height

        fallback = self._probe_video_size_from_caps(video_file)
        if fallback is None and not _GST_PBUTILS_AVAILABLE:
            global _PBUTILS_WARNING_EMITTED
            if not _PBUTILS_WARNING_EMITTED:
                print(
                    "GStreamer GstPbutils introspection data is unavailable. Install "
                    "the gstreamer1.0-plugins-base (or equivalent) package to improve "
                    "letterbox detection for crop regions.",
                    file=sys.stderr,
                )
                _PBUTILS_WARNING_EMITTED = True
        return fallback

    def _probe_video_size_from_caps(self, video_file: Path) -> tuple[float, float] | None:
        pipeline = Gst.Pipeline.new("ballclips-metadata-probe")
        if pipeline is None:
            return None

        decodebin = Gst.ElementFactory.make("uridecodebin", "metadata_decodebin")
        sink = Gst.ElementFactory.make("fakesink", "metadata_sink")
        if decodebin is None or sink is None:
            pipeline.set_state(Gst.State.NULL)
            return None

        sink.set_property("sync", False)
        sink.set_property("enable-last-sample", False)
        # Limit the pads we receive to raw video to avoid not-linked errors from
        # audio streams when probing metadata.
        decode_caps = Gst.Caps.from_string("video/x-raw")
        decodebin.set_property("caps", decode_caps)

        pipeline.add(decodebin)
        pipeline.add(sink)

        result: dict[str, tuple[float, float] | None] = {"size": None}

        def _on_pad_added(_bin: Gst.Element, pad: Gst.Pad, _sink: Gst.Element) -> None:
            if result["size"] is not None:
                return
            caps = pad.get_current_caps()
            if caps is None:
                caps = pad.query_caps()
            if caps is None or caps.get_size() == 0:
                return
            structure = caps.get_structure(0)
            if structure is None:
                return
            name = structure.get_name()
            if not name.startswith("video/"):
                return
            width = structure.get_value("width")
            height = structure.get_value("height")
            if not isinstance(width, int) or not isinstance(height, int):
                return
            if width <= 0 or height <= 0:
                return
            par = _structure_fraction(structure, "pixel-aspect-ratio")
            fr = _structure_fraction(structure, "framerate")
            width_f = float(width)
            height_f = float(height)
            if par is not None:
                par_num, par_den = par
                if par_num > 0 and par_den > 0:
                    width_f *= par_num / par_den
            if fr is not None:
                fr_num, fr_den = fr
                if fr_num > 0 and fr_den > 0:
                    self._current_video_frame_rate = (fr_num, fr_den)
            result["size"] = (width_f, height_f)
            sink_pad = _sink.get_static_pad("sink")
            if sink_pad is not None and not sink_pad.is_linked():
                pad.link(sink_pad)

        decodebin.connect("pad-added", _on_pad_added, sink)
        decodebin.set_property("uri", Gst.filename_to_uri(str(video_file.resolve())))

        bus = pipeline.get_bus()
        if bus is None:
            pipeline.set_state(Gst.State.NULL)
            return None

        pipeline.set_state(Gst.State.PAUSED)
        try:
            message = bus.timed_pop_filtered(
                5 * Gst.SECOND,
                Gst.MessageType.ASYNC_DONE
                | Gst.MessageType.ERROR
                | Gst.MessageType.EOS,
            )
            if message is not None and message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                detail = f" ({debug})" if debug else ""
                print(
                    f"Unable to extract video metadata for '{video_file.name}': {err.message}{detail}",
                    file=sys.stderr,
                )
        finally:
            pipeline.set_state(Gst.State.NULL)

        return result["size"]

    def _on_destroy(self, *_args: object) -> None:
        self._player.set_state(Gst.State.NULL)
        if self._progress_update_id is not None:
            GLib.source_remove(self._progress_update_id)
            self._progress_update_id = None

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        message_type = message.type
        if message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer error: {err.message}")
            if debug:
                print(f"Debug details: {debug}")
            self._player.set_state(Gst.State.NULL)
        elif message_type == Gst.MessageType.EOS:
            # Restart playback when the video finishes to keep the preview running.
            self._player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)

    def _on_play_pause_clicked(self, _button: Gtk.Button) -> None:
        target_state = Gst.State.PAUSED if self._is_playing else Gst.State.PLAYING
        self._player.set_state(target_state)
        self._is_playing = target_state == Gst.State.PLAYING
        self._update_play_pause_icon()

    def _on_maximize_crop_clicked(self, _button: Gtk.Button) -> None:
        key = self._determine_active_crop_key()
        if key is None or self._current_video_file is None:
            return
        allocation = self._crop_overlay.get_allocation()
        width = allocation.width
        height = allocation.height
        if width <= 0 or height <= 0:
            return
        region = self._get_crop_region(self._current_video_file, key)
        expanded = self._maximize_region(region, width, height)
        self._set_crop_region(key, expanded, width=width, height=height)

    def _update_play_pause_icon(self) -> None:
        icon_name = "media-playback-pause" if self._is_playing else "media-playback-start"
        self._play_pause_button.set_image(Gtk.Image.new_from_icon_name(icon_name, Gtk.IconSize.BUTTON))

    def _on_progress_changed(self, _scale: Gtk.Scale) -> None:
        if self._updating_progress:
            return
        value_seconds = self._progress_adjustment.get_value()
        position_ns = int(value_seconds * Gst.SECOND)
        self._player.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
            position_ns,
        )

    def _update_progress(self) -> bool:
        success, duration_ns = self._player.query_duration(Gst.Format.TIME)
        if success and duration_ns > 0 and duration_ns != self._duration_ns:
            self._duration_ns = duration_ns
            self._progress_adjustment.set_upper(duration_ns / Gst.SECOND)
            self._reset_trims_if_needed()
            self._trim_area.queue_draw()

        success, position_ns = self._player.query_position(Gst.Format.TIME)
        if success and self._duration_ns > 0:
            self._set_progress_value(position_ns / Gst.SECOND)

        self._crop_overlay.queue_draw()
        return True

    def _set_progress_value(self, value_seconds: float) -> None:
        self._updating_progress = True
        self._progress_adjustment.set_value(value_seconds)
        self._updating_progress = False
        self._crop_overlay.queue_draw()

    def _seek_to_seconds(self, value_seconds: float) -> None:
        upper = self._progress_adjustment.get_upper()
        if upper <= 0.0:
            return
        clamped_value = max(0.0, min(upper, value_seconds))
        position_ns = int(clamped_value * Gst.SECOND)
        self._set_progress_value(clamped_value)
        self._player.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
            position_ns,
        )
        self._crop_overlay.queue_draw()

    def _on_prev_clicked(self, _button: Gtk.Button) -> None:
        self._load_video(self._current_index - 1)

    def _on_next_clicked(self, _button: Gtk.Button) -> None:
        self._load_video(self._current_index + 1)

    def _reset_trims_if_needed(self) -> None:
        duration_seconds = self._progress_adjustment.get_upper()
        if duration_seconds <= 0:
            return

        if self._pending_trim_reset or self._trim_out_seconds <= 0:
            self._trim_in_seconds = 0.0
            self._trim_out_seconds = duration_seconds
            self._pending_trim_reset = False
        else:
            self._trim_in_seconds = max(0.0, min(self._trim_in_seconds, duration_seconds))
            self._trim_out_seconds = max(
                self._trim_in_seconds, min(self._trim_out_seconds, duration_seconds)
            )

    def _ensure_crop_defaults(self, video_file: Path) -> None:
        if video_file in self._video_crop_regions:
            return
        default_in = CropRegion(0.5, 0.5, 1.0)
        default_out = CropRegion(0.5, 0.5, 1.0)
        in_region = default_in
        out_region = default_out
        metadata = self._load_video_metadata(video_file)
        if metadata is not None:
            square_data = getattr(metadata, "square_cropping", None)
            if isinstance(square_data, SquareCroppingData):
                converted_in = self._square_point_to_crop_region(square_data.in_point)
                if converted_in is not None:
                    in_region = converted_in
                converted_out = self._square_point_to_crop_region(square_data.out_point)
                if converted_out is not None:
                    out_region = converted_out
            elif isinstance(square_data, SquareCroppingPoint):
                converted = self._square_point_to_crop_region(square_data)
                if converted is not None:
                    in_region = converted
                    out_region = converted

        self._video_crop_regions[video_file] = {
            "in": in_region.clamped(),
            "out": out_region.clamped(),
        }

    def _get_crop_region(self, video_file: Path, key: Literal["in", "out"]) -> CropRegion:
        self._ensure_crop_defaults(video_file)
        stored = self._video_crop_regions[video_file][key]
        return stored.clamped()

    def _metadata_path(self, video_file: Path) -> Path:
        return video_file.with_suffix(".json")

    def _load_video_metadata(self, video_file: Path) -> VideoMetadata | None:
        metadata_path = self._metadata_path(video_file)
        if not metadata_path.exists():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                raw_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"Failed to read metadata for '{video_file.name}': {exc}",
                file=sys.stderr,
            )
            return None
        try:
            return ImplicitDict.parse(raw_metadata, VideoMetadata)
        except (TypeError, ValueError) as exc:
            print(
                f"Metadata for '{video_file.name}' is invalid: {exc}",
                file=sys.stderr,
            )
            return None

    def _save_video_metadata(self, video_file: Path) -> None:
        self._ensure_crop_defaults(video_file)
        metadata_path = self._metadata_path(video_file)
        in_region = self._video_crop_regions[video_file]["in"].clamped()
        out_region = self._video_crop_regions[video_file]["out"].clamped()
        metadata = VideoMetadata(
            square_cropping=SquareCroppingData(
                in_point=self._crop_region_to_square_point(in_region, self._trim_in_seconds),
                out_point=self._crop_region_to_square_point(out_region, self._trim_out_seconds),
            )
        )
        try:
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            print(
                f"Failed to write metadata for '{video_file.name}': {exc}",
                file=sys.stderr,
            )

    def _persist_metadata_for_current_video(self) -> None:
        if self._current_video_file is None:
            return
        self._ensure_crop_defaults(self._current_video_file)
        self._save_video_metadata(self._current_video_file)

    def _crop_region_to_square_point(self, region: CropRegion, seconds: float) -> SquareCroppingPoint:
        video_w, video_h = self._current_video_pixel_size or (0.0, 0.0)
        min_dimension = min(video_w, video_h) if video_w > 0 and video_h > 0 else 0.0
        u = int(round(region.center_x * video_w)) if video_w > 0 else 0
        v = int(round(region.center_y * video_h)) if video_h > 0 else 0
        size = int(round(region.size * min_dimension)) if min_dimension > 0 else 0
        return SquareCroppingPoint(
            frame=self._seconds_to_frame(seconds),
            u=u,
            v=v,
            size=size,
        )

    def _square_point_to_crop_region(
        self, point: SquareCroppingPoint
    ) -> CropRegion | None:
        video_size = self._current_video_pixel_size
        if not video_size:
            return None
        video_w, video_h = video_size
        if video_w <= 0 or video_h <= 0:
            return None
        u = point.u
        v = point.v
        size = point.size
        if not isinstance(u, (int, float)) or not isinstance(v, (int, float)):
            return None
        if not isinstance(size, (int, float)):
            return None
        min_dimension = min(video_w, video_h)
        if min_dimension <= 0:
            return None
        center_x = float(u) / video_w
        center_y = float(v) / video_h
        size_ratio = float(size) / min_dimension
        return CropRegion(center_x, center_y, size_ratio).clamped()

    def _seconds_to_frame(self, seconds: float) -> int:
        rate = self._current_video_frame_rate
        if rate is None:
            return int(round(seconds * 1000.0))
        num, den = rate
        if num <= 0 or den <= 0:
            return int(round(seconds * 1000.0))
        fps = num / den
        return int(round(seconds * fps))

    def _get_video_display_rect(
        self, width: int, height: int
    ) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            return 0.0, 0.0, 0.0, 0.0
        if not self._current_video_pixel_size:
            return 0.0, 0.0, float(width), float(height)
        video_w, video_h = self._current_video_pixel_size
        if video_w <= 0 or video_h <= 0:
            return 0.0, 0.0, float(width), float(height)
        scale = min(width / video_w, height / video_h)
        display_width = video_w * scale
        display_height = video_h * scale
        offset_x = (width - display_width) / 2.0
        offset_y = (height - display_height) / 2.0
        return offset_x, offset_y, display_width, display_height

    def _clamp_region_to_video_area(
        self, region: CropRegion, rect: tuple[float, float, float, float]
    ) -> CropRegion:
        offset_x, offset_y, video_w, video_h = rect
        if video_w <= 0 or video_h <= 0:
            return region.clamped()
        min_dimension = min(video_w, video_h)
        if min_dimension <= 0:
            return CropRegion(0.5, 0.5, 0.0)
        size_px = max(0.0, min(region.size, 1.0)) * min_dimension
        half = size_px / 2.0
        center_x_px = region.center_x * video_w
        center_y_px = region.center_y * video_h
        center_x_px = max(half, min(center_x_px, video_w - half))
        center_y_px = max(half, min(center_y_px, video_h - half))
        normalized_size = size_px / min_dimension if min_dimension > 0 else 0.0
        center_x = center_x_px / video_w if video_w > 0 else 0.5
        center_y = center_y_px / video_h if video_h > 0 else 0.5
        return CropRegion(center_x, center_y, normalized_size).clamped()

    def _point_in_video_area(self, x: float, y: float, width: int, height: int) -> bool:
        offset_x, offset_y, video_w, video_h = self._get_video_display_rect(width, height)
        if video_w <= 0 or video_h <= 0:
            return False
        return (
            offset_x <= x <= offset_x + video_w
            and offset_y <= y <= offset_y + video_h
        )

    def _region_to_rectangle(
        self, region: CropRegion, rect: tuple[float, float, float, float]
    ) -> tuple[float, float, float]:
        offset_x, offset_y, video_w, video_h = rect
        if video_w <= 0 or video_h <= 0:
            return 0.0, 0.0, 0.0
        return region.to_rectangle(video_w, video_h, offset_x=offset_x, offset_y=offset_y)

    def _set_crop_region(
        self,
        key: Literal["in", "out"],
        region: CropRegion,
        *,
        width: int | None = None,
        height: int | None = None,
        persist: bool = True,
    ) -> None:
        if self._current_video_file is None:
            return
        self._ensure_crop_defaults(self._current_video_file)
        if width is None or height is None:
            allocation = self._crop_overlay.get_allocation()
            width = allocation.width
            height = allocation.height
        fitted = self._fit_region_to_bounds(region, width or 0, height or 0)
        updated = fitted.clamped()
        current = self._video_crop_regions[self._current_video_file][key]
        if (
            abs(current.center_x - updated.center_x) <= self._crop_edit_tolerance
            and abs(current.center_y - updated.center_y) <= self._crop_edit_tolerance
            and abs(current.size - updated.size) <= self._crop_edit_tolerance
        ):
            return
        self._video_crop_regions[self._current_video_file][key] = updated
        self._crop_overlay.queue_draw()
        if persist:
            self._persist_metadata_for_current_video()

    def _interpolate_regions(self, start: CropRegion, end: CropRegion, t: float) -> CropRegion:
        t = max(0.0, min(1.0, t))
        return CropRegion(
            center_x=start.center_x + (end.center_x - start.center_x) * t,
            center_y=start.center_y + (end.center_y - start.center_y) * t,
            size=start.size + (end.size - start.size) * t,
        ).clamped()

    def _draw_crop_region(
        self,
        cr: cairo.Context,
        region: CropRegion,
        width: int,
        height: int,
        fill_rgba: tuple[float, float, float, float],
        stroke_rgba: tuple[float, float, float, float],
    ) -> bool:
        rect = self._get_video_display_rect(width, height)
        fitted = self._clamp_region_to_video_area(region, rect)
        left, top, size = self._region_to_rectangle(fitted, rect)
        if size <= 0:
            return False
        cr.set_source_rgba(*fill_rgba)
        cr.rectangle(left, top, size, size)
        cr.fill_preserve()
        cr.set_source_rgba(*stroke_rgba)
        cr.set_line_width(2.0)
        cr.stroke()
        return True

    def _determine_active_crop_key(self) -> Literal["in", "out"] | None:
        if self._current_video_file is None:
            return None
        position = self._progress_adjustment.get_value()
        if abs(position - self._trim_in_seconds) <= self._crop_edit_tolerance:
            return "in"
        if abs(position - self._trim_out_seconds) <= self._crop_edit_tolerance:
            return "out"
        return None

    def _fit_region_to_bounds(
        self, region: CropRegion, width: int, height: int
    ) -> CropRegion:
        rect = self._get_video_display_rect(width, height)
        return self._clamp_region_to_video_area(region, rect)

    def _translate_region_by_delta(
        self,
        region: CropRegion,
        dx: float,
        dy: float,
        width: int,
        height: int,
    ) -> CropRegion:
        rect = self._get_video_display_rect(width, height)
        _offset_x, _offset_y, video_w, video_h = rect
        if video_w <= 0 or video_h <= 0:
            return region.clamped()
        min_dimension = float(min(video_w, video_h))
        if min_dimension <= 0:
            return region.clamped()
        size_px = max(0.0, min(region.size, 1.0)) * min_dimension
        half = size_px / 2.0
        center_x_px = region.center_x * video_w + dx
        center_y_px = region.center_y * video_h + dy
        center_x_px = max(half, min(center_x_px, video_w - half))
        center_y_px = max(half, min(center_y_px, video_h - half))
        normalized_size = size_px / min_dimension if min_dimension > 0 else 0.0
        candidate = CropRegion(
            center_x_px / video_w if video_w > 0 else region.center_x,
            center_y_px / video_h if video_h > 0 else region.center_y,
            normalized_size,
        )
        return self._clamp_region_to_video_area(candidate, rect)

    def _maximize_region(
        self, region: CropRegion, width: int, height: int
    ) -> CropRegion:
        rect = self._get_video_display_rect(width, height)
        return self._clamp_region_to_video_area(
            CropRegion(region.center_x, region.center_y, 1.0), rect
        )

    def _build_region_from_drag(
        self,
        start: tuple[float, float],
        current: tuple[float, float],
        width: int,
        height: int,
    ) -> CropRegion | None:
        rect = self._get_video_display_rect(width, height)
        offset_x, offset_y, video_w, video_h = rect
        if video_w <= 0 or video_h <= 0:
            return None
        start_x = max(offset_x, min(start[0], offset_x + video_w))
        start_y = max(offset_y, min(start[1], offset_y + video_h))
        current_x = max(offset_x, min(current[0], offset_x + video_w))
        current_y = max(offset_y, min(current[1], offset_y + video_h))
        start_local_x = start_x - offset_x
        start_local_y = start_y - offset_y
        current_local_x = current_x - offset_x
        current_local_y = current_y - offset_y
        dx = current_local_x - start_local_x
        dy = current_local_y - start_local_y
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dx == 0 and abs_dy == 0:
            return None
        if abs_dx == 0 or abs_dy == 0:
            side = max(abs_dx, abs_dy)
        else:
            side = min(abs_dx, abs_dy)
        min_dimension = float(min(video_w, video_h))
        side = min(side, min_dimension)
        if side <= 0:
            return None
        left = start_local_x if dx >= 0 else start_local_x - side
        top = start_local_y if dy >= 0 else start_local_y - side
        left = max(0.0, min(left, video_w - side))
        top = max(0.0, min(top, video_h - side))
        center_x = (left + side / 2.0) / video_w if video_w > 0 else 0.5
        center_y = (top + side / 2.0) / video_h if video_h > 0 else 0.5
        size_ratio = side / min_dimension if min_dimension > 0 else 0.0
        region = CropRegion(center_x, center_y, size_ratio).clamped()
        return self._clamp_region_to_video_area(region, rect)

    def _on_crop_overlay_draw(self, widget: Gtk.DrawingArea, cr: cairo.Context) -> bool:
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        if width <= 0 or height <= 0:
            return False

        if (
            self._crop_dragging
            and self._crop_drag_button == 1
            and self._crop_drag_start
            and self._crop_drag_current
        ):
            preview = self._build_region_from_drag(
                self._crop_drag_start,
                self._crop_drag_current,
                width,
                height,
            )
            if preview is not None:
                self._draw_crop_region(
                    cr,
                    preview,
                    width,
                    height,
                    (0.0, 0.8, 0.0, 0.2),
                    (0.0, 0.8, 0.0, 0.8),
                )

        if self._current_video_file is None:
            return False

        trim_in = self._trim_in_seconds
        trim_out = self._trim_out_seconds
        if trim_out <= trim_in:
            active_region = self._get_crop_region(self._current_video_file, "in")
        else:
            position = self._progress_adjustment.get_value()
            if position < trim_in or position > trim_out:
                return False
            in_region = self._get_crop_region(self._current_video_file, "in")
            out_region = self._get_crop_region(self._current_video_file, "out")
            if position <= trim_in:
                active_region = in_region
            elif position >= trim_out:
                active_region = out_region
            else:
                t = (position - trim_in) / (trim_out - trim_in)
                active_region = self._interpolate_regions(in_region, out_region, t)

        self._draw_crop_region(
            cr,
            active_region,
            width,
            height,
            (0.0, 1.0, 1.0, 0.2),
            (0.0, 1.0, 1.0, 0.8),
        )
        return False

    def _on_crop_button_press(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        key = self._determine_active_crop_key()
        if key is None:
            return False

        if event.button == 1:
            allocation = self._crop_overlay.get_allocation()
            if not self._point_in_video_area(
                event.x, event.y, allocation.width, allocation.height
            ):
                return False
            self._crop_dragging = True
            self._crop_drag_button = 1
            self._crop_drag_key = key
            self._crop_drag_start = (event.x, event.y)
            self._crop_drag_current = (event.x, event.y)
            self._crop_initial_region = None
            self._crop_overlay.queue_draw()
            return True

        if event.button == 3:
            if self._current_video_file is None:
                return False
            allocation = self._crop_overlay.get_allocation()
            if not self._point_in_video_area(
                event.x, event.y, allocation.width, allocation.height
            ):
                return False
            self._crop_dragging = True
            self._crop_drag_button = 3
            self._crop_drag_key = key
            self._crop_drag_start = (event.x, event.y)
            self._crop_drag_current = (event.x, event.y)
            region = self._get_crop_region(self._current_video_file, key)
            allocation = self._crop_overlay.get_allocation()
            self._crop_initial_region = self._fit_region_to_bounds(
                region,
                allocation.width,
                allocation.height,
            )
            return True

        return False

    def _on_crop_motion(self, _widget: Gtk.Widget, event: Gdk.EventMotion) -> bool:
        if (
            not self._crop_dragging
            or self._crop_drag_start is None
            or self._crop_drag_key is None
        ):
            return False

        allocation = self._crop_overlay.get_allocation()
        width = allocation.width
        height = allocation.height

        if self._crop_drag_button == 1:
            self._crop_drag_current = (event.x, event.y)
            self._crop_overlay.queue_draw()
            return True

        if self._crop_drag_button == 3 and self._crop_initial_region is not None:
            dx = event.x - self._crop_drag_start[0]
            dy = event.y - self._crop_drag_start[1]
            region = self._translate_region_by_delta(
                self._crop_initial_region, dx, dy, width, height
            )
            self._set_crop_region(
                self._crop_drag_key,
                region,
                width=width,
                height=height,
                persist=False,
            )
            return True

        return False

    def _on_crop_button_release(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        if (
            not self._crop_dragging
            or self._crop_drag_start is None
            or self._crop_drag_key is None
            or event.button != self._crop_drag_button
        ):
            return False

        allocation = self._crop_overlay.get_allocation()
        width = allocation.width
        height = allocation.height

        if self._crop_drag_button == 1:
            region = self._build_region_from_drag(
                self._crop_drag_start,
                (event.x, event.y),
                width,
                height,
            )
            if region is not None:
                self._set_crop_region(
                    self._crop_drag_key,
                    region,
                    width=width,
                    height=height,
                )
        elif self._crop_drag_button == 3 and self._crop_initial_region is not None:
            dx = event.x - self._crop_drag_start[0]
            dy = event.y - self._crop_drag_start[1]
            region = self._translate_region_by_delta(
                self._crop_initial_region, dx, dy, width, height
            )
            self._set_crop_region(
                self._crop_drag_key,
                region,
                width=width,
                height=height,
            )

        self._crop_dragging = False
        self._crop_drag_button = None
        self._crop_drag_key = None
        self._crop_drag_start = None
        self._crop_drag_current = None
        self._crop_initial_region = None
        self._crop_overlay.queue_draw()
        return True

    def _on_trim_area_draw(self, _widget: Gtk.DrawingArea, cr: cairo.Context) -> bool:
        allocation = self._trim_area.get_allocation()
        width = allocation.width
        height = allocation.height

        cr.set_source_rgb(0.15, 0.15, 0.15)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        duration_seconds = max(self._progress_adjustment.get_upper(), 0.0)
        if duration_seconds <= 0 or width <= 0:
            return False

        in_x = self._seconds_to_x(self._trim_in_seconds, duration_seconds, width)
        out_x = self._seconds_to_x(self._trim_out_seconds, duration_seconds, width)

        cr.set_source_rgba(0.0, 1.0, 1.0, 0.3)
        cr.rectangle(in_x, 0, max(out_x - in_x, 0.0), height)
        cr.fill()

        self._draw_brace(cr, in_x, height, (0.0, 0.8, 0.0), 1)
        self._draw_brace(cr, out_x, height, (0.8, 0.0, 0.0), -1)

        return False

    def _draw_brace(
        self,
        cr: cairo.Context,
        x: float,
        height: int,
        color: tuple[float, float, float],
        direction: int,
    ) -> None:
        cr.set_source_rgb(*color)
        brace_width = 6.0 * direction
        cr.set_line_width(2.0)
        cr.move_to(x, 0)
        cr.line_to(x, height)
        cr.move_to(x, 0)
        cr.line_to(x + brace_width, 0)
        cr.move_to(x, height)
        cr.line_to(x + brace_width, height)
        cr.stroke()

    def _seconds_to_x(self, value: float, duration: float, width: int) -> float:
        return max(0.0, min(width, (value / duration) * width))

    def _x_to_seconds(self, x: float, duration: float, width: int) -> float:
        if width <= 0:
            return 0.0
        return max(0.0, min(duration, (x / width) * duration))

    def _on_trim_button_press(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        if event.button != 1:
            return False

        duration_seconds = self._progress_adjustment.get_upper()
        allocation = self._trim_area.get_allocation()
        width = allocation.width
        in_x = self._seconds_to_x(self._trim_in_seconds, duration_seconds, width)
        out_x = self._seconds_to_x(self._trim_out_seconds, duration_seconds, width)
        tolerance = 8.0

        if abs(event.x - in_x) <= tolerance:
            self._active_trim = "in"
            return True
        if abs(event.x - out_x) <= tolerance:
            self._active_trim = "out"
            return True

        return False

    def _on_trim_button_release(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        if event.button != 1:
            return False
        self._active_trim = None
        self._persist_metadata_for_current_video()
        return True

    def _on_trim_motion(self, _widget: Gtk.Widget, event: Gdk.EventMotion) -> bool:
        if self._active_trim is None:
            return False

        duration_seconds = self._progress_adjustment.get_upper()
        allocation = self._trim_area.get_allocation()
        width = allocation.width
        value = self._x_to_seconds(event.x, duration_seconds, width)

        if self._active_trim == "in":
            self._trim_in_seconds = min(value, self._trim_out_seconds)
        else:
            self._trim_out_seconds = max(value, self._trim_in_seconds)

        self._pending_trim_reset = False
        self._trim_area.queue_draw()
        self._crop_overlay.queue_draw()
        return True

    def _on_set_in_clicked(self, _button: Gtk.Button) -> None:
        position = self._progress_adjustment.get_value()
        self._trim_in_seconds = min(position, self._trim_out_seconds)
        self._pending_trim_reset = False
        self._trim_area.queue_draw()
        self._crop_overlay.queue_draw()
        self._persist_metadata_for_current_video()

    def _on_set_out_clicked(self, _button: Gtk.Button) -> None:
        position = self._progress_adjustment.get_value()
        self._trim_out_seconds = max(position, self._trim_in_seconds)
        self._pending_trim_reset = False
        self._trim_area.queue_draw()
        self._crop_overlay.queue_draw()
        self._persist_metadata_for_current_video()

    def _on_go_to_in_clicked(self, _button: Gtk.Button) -> None:
        self._seek_to_seconds(self._trim_in_seconds)

    def _on_go_to_out_clicked(self, _button: Gtk.Button) -> None:
        self._seek_to_seconds(self._trim_out_seconds)


class BallclipsApplication(Gtk.Application):
    """Gtk.Application wrapper for the prototype UI."""

    def __init__(self, video_files: Sequence[Path], start_index: int = 0) -> None:
        super().__init__(application_id="dev.ballclips.prototype")
        Gst.init(None)
        self._video_files = list(video_files)
        if not self._video_files:
            raise ValueError("BallclipsApplication requires at least one video file.")
        self._current_index = start_index % len(self._video_files)
        self._window: PlayerWindow | None = None

    def do_activate(self) -> None:  # type: ignore[override]
        if self._window is None:
            self._window = PlayerWindow(self, self._video_files, self._current_index)
        self._window.present()

    def do_shutdown(self) -> None:  # type: ignore[override]
        if self._window is not None:
            self._window.destroy()
            self._window = None
        Gtk.Application.do_shutdown(self)

    def set_current_index(self, index: int) -> None:
        self._current_index = index % len(self._video_files)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and browse MP4 files in a folder.")
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Folder to scan for MP4 files (defaults to the current working directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder '{folder}' does not exist or is not a directory.")

    try:
        video_files = _list_mp4_files(folder)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    app = BallclipsApplication(video_files)
    return app.run(sys.argv[:1])


if __name__ == "__main__":
    raise SystemExit(main())
