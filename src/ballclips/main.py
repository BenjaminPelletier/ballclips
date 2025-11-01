"""Application entry point for the ballclips prototype UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Sequence

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Gst", "1.0")

from gi.repository import Gdk, GLib, Gst, Gtk, Pango, cairo


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

        sink = Gst.ElementFactory.make("gtksink", "video_sink")
        if sink is None:
            raise RuntimeError(
                "The 'gtksink' plugin is required. Install the GStreamer GTK plugin package."
            )

        self._player.set_property("video-sink", sink)
        video_widget = sink.props.widget
        video_widget.set_hexpand(True)
        video_widget.set_vexpand(True)

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
        container.pack_start(video_widget, True, True, 0)

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

        set_in_button = Gtk.Button(label="{")
        set_in_button.connect("clicked", self._on_set_in_clicked)

        goto_in_button = Gtk.Button(label="→{")
        goto_in_button.connect("clicked", self._on_go_to_in_clicked)

        set_in_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        set_in_column.pack_start(goto_in_button, False, False, 0)
        set_in_column.pack_start(set_in_button, False, False, 0)

        set_out_button = Gtk.Button(label="}")
        set_out_button.connect("clicked", self._on_set_out_clicked)

        goto_out_button = Gtk.Button(label="→}")
        goto_out_button.connect("clicked", self._on_go_to_out_clicked)

        set_out_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        set_out_column.pack_start(goto_out_button, False, False, 0)
        set_out_column.pack_start(set_out_button, False, False, 0)

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

        left_spacer = Gtk.Box()
        right_spacer = Gtk.Box()

        timeline_grid.attach(left_spacer, 0, 0, 1, 1)
        timeline_grid.attach(progress_scale, 1, 0, 1, 1)
        timeline_grid.attach(right_spacer, 2, 0, 1, 1)

        timeline_grid.attach(set_in_column, 0, 1, 1, 1)
        timeline_grid.attach(trim_area, 1, 1, 1, 1)
        timeline_grid.attach(set_out_column, 2, 1, 1, 1)

        left_group = Gtk.SizeGroup(Gtk.SizeGroupMode.HORIZONTAL)
        left_group.add_widget(left_spacer)
        left_group.add_widget(set_in_button)

        right_group = Gtk.SizeGroup(Gtk.SizeGroupMode.HORIZONTAL)
        right_group.add_widget(right_spacer)
        right_group.add_widget(set_out_button)

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

        self._load_video(self._current_index)

    def _load_video(self, index: int) -> None:
        if not self._video_files:
            return

        self._current_index = index % len(self._video_files)
        video_file = self._video_files[self._current_index]
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

        return True

    def _set_progress_value(self, value_seconds: float) -> None:
        self._updating_progress = True
        self._progress_adjustment.set_value(value_seconds)
        self._updating_progress = False

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
        return True

    def _on_set_in_clicked(self, _button: Gtk.Button) -> None:
        position = self._progress_adjustment.get_value()
        self._trim_in_seconds = min(position, self._trim_out_seconds)
        self._pending_trim_reset = False
        self._trim_area.queue_draw()

    def _on_set_out_clicked(self, _button: Gtk.Button) -> None:
        position = self._progress_adjustment.get_value()
        self._trim_out_seconds = max(position, self._trim_in_seconds)
        self._pending_trim_reset = False
        self._trim_area.queue_draw()

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
