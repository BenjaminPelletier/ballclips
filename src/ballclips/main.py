"""Application entry point for the ballclips prototype UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")

from gi.repository import Gst, Gtk, Pango


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
        self.add(container)
        self.show_all()

        self.connect("destroy", self._on_destroy)

        bus = self._player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

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

        self._app.set_current_index(self._current_index)

    def _on_destroy(self, *_args: object) -> None:
        self._player.set_state(Gst.State.NULL)

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

    def _on_prev_clicked(self, _button: Gtk.Button) -> None:
        self._load_video(self._current_index - 1)

    def _on_next_clicked(self, _button: Gtk.Button) -> None:
        self._load_video(self._current_index + 1)


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
