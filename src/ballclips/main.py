"""Application entry point for the ballclips prototype UI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Gst", "1.0")

from gi.repository import Gdk, GLib, Gst, Gtk, Pango, cairo


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

    def to_rectangle(self, width: int, height: int) -> tuple[float, float, float]:
        min_dimension = max(1, min(width, height))
        size_px = max(0.0, self.size) * min_dimension
        center_x_px = self.center_x * width
        center_y_px = self.center_y * height
        half = size_px / 2.0
        left = center_x_px - half
        top = center_y_px - half
        left = max(0.0, min(left, width - size_px))
        top = max(0.0, min(top, height - size_px))
        return left, top, size_px


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

        crop_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        crop_label = Gtk.Label(label="Crop region")
        crop_label.set_halign(Gtk.Align.START)
        crop_controls.pack_start(crop_label, False, False, 0)

        crop_in_button = Gtk.ToggleButton(label="in {")
        crop_in_button.connect("toggled", self._on_crop_edit_toggled, "in")
        crop_controls.pack_start(crop_in_button, False, False, 0)

        crop_out_button = Gtk.ToggleButton(label="out }")
        crop_out_button.connect("toggled", self._on_crop_edit_toggled, "out")
        crop_controls.pack_start(crop_out_button, False, False, 0)

        controls.pack_start(crop_controls, False, False, 0)

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
        self._crop_in_button = crop_in_button
        self._crop_out_button = crop_out_button
        self._crop_edit_mode: Literal["in", "out"] | None = None
        self._crop_dragging = False
        self._crop_drag_start: tuple[float, float] | None = None
        self._crop_drag_current: tuple[float, float] | None = None
        self._video_crop_regions: dict[Path, dict[str, CropRegion]] = {}
        self._current_video_file: Path | None = None

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
        self._video_crop_regions[video_file] = {"in": default_in, "out": default_out}

    def _get_crop_region(self, video_file: Path, key: Literal["in", "out"]) -> CropRegion:
        self._ensure_crop_defaults(video_file)
        stored = self._video_crop_regions[video_file][key]
        return stored.clamped()

    def _set_crop_region(self, key: Literal["in", "out"], region: CropRegion) -> None:
        if self._current_video_file is None:
            return
        self._ensure_crop_defaults(self._current_video_file)
        self._video_crop_regions[self._current_video_file][key] = region.clamped()
        self._crop_overlay.queue_draw()

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
        left, top, size = region.to_rectangle(width, height)
        if size <= 0:
            return False
        cr.set_source_rgba(*fill_rgba)
        cr.rectangle(left, top, size, size)
        cr.fill_preserve()
        cr.set_source_rgba(*stroke_rgba)
        cr.set_line_width(2.0)
        cr.stroke()
        return True

    def _build_region_from_drag(
        self,
        start: tuple[float, float],
        current: tuple[float, float],
        width: int,
        height: int,
    ) -> CropRegion | None:
        if width <= 0 or height <= 0:
            return None
        start_x = max(0.0, min(start[0], float(width)))
        start_y = max(0.0, min(start[1], float(height)))
        current_x = max(0.0, min(current[0], float(width)))
        current_y = max(0.0, min(current[1], float(height)))
        dx = current_x - start_x
        dy = current_y - start_y
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dx == 0 and abs_dy == 0:
            return None
        if abs_dx == 0 or abs_dy == 0:
            side = max(abs_dx, abs_dy)
        else:
            side = min(abs_dx, abs_dy)
        min_dimension = float(min(width, height))
        side = min(side, min_dimension)
        if side <= 0:
            return None
        left = start_x if dx >= 0 else start_x - side
        top = start_y if dy >= 0 else start_y - side
        left = max(0.0, min(left, width - side))
        top = max(0.0, min(top, height - side))
        center_x = (left + side / 2.0) / width if width > 0 else 0.5
        center_y = (top + side / 2.0) / height if height > 0 else 0.5
        size_ratio = side / min_dimension if min_dimension > 0 else 0.0
        return CropRegion(center_x, center_y, size_ratio).clamped()

    def _on_crop_overlay_draw(self, widget: Gtk.DrawingArea, cr: cairo.Context) -> bool:
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        if width <= 0 or height <= 0:
            return False

        if self._crop_dragging and self._crop_drag_start and self._crop_drag_current:
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

    def _on_crop_edit_toggled(
        self, button: Gtk.ToggleButton, key: Literal["in", "out"]
    ) -> None:
        if button.get_active():
            if key == "in" and self._crop_out_button.get_active():
                self._crop_out_button.set_active(False)
            elif key == "out" and self._crop_in_button.get_active():
                self._crop_in_button.set_active(False)
            self._crop_edit_mode = key
        else:
            if self._crop_edit_mode == key:
                self._crop_edit_mode = None
        if not button.get_active():
            self._crop_dragging = False
        self._crop_overlay.queue_draw()

    def _on_crop_button_press(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        if event.button != 1 or self._crop_edit_mode is None:
            return False
        self._crop_dragging = True
        self._crop_drag_start = (event.x, event.y)
        self._crop_drag_current = (event.x, event.y)
        self._crop_overlay.queue_draw()
        return True

    def _on_crop_motion(self, _widget: Gtk.Widget, event: Gdk.EventMotion) -> bool:
        if not self._crop_dragging or self._crop_drag_start is None:
            return False
        self._crop_drag_current = (event.x, event.y)
        self._crop_overlay.queue_draw()
        return True

    def _on_crop_button_release(self, _widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        if event.button != 1 or not self._crop_dragging or self._crop_drag_start is None:
            return False
        allocation = self._crop_overlay.get_allocation()
        width = allocation.width
        height = allocation.height
        if self._crop_edit_mode is not None:
            region = self._build_region_from_drag(
                self._crop_drag_start,
                (event.x, event.y),
                width,
                height,
            )
            if region is not None:
                self._set_crop_region(self._crop_edit_mode, region)
        self._crop_dragging = False
        self._crop_drag_start = None
        self._crop_drag_current = None
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

    def _on_set_out_clicked(self, _button: Gtk.Button) -> None:
        position = self._progress_adjustment.get_value()
        self._trim_out_seconds = max(position, self._trim_in_seconds)
        self._pending_trim_reset = False
        self._trim_area.queue_draw()
        self._crop_overlay.queue_draw()

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
