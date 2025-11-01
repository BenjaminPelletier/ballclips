"""Application entry point for the ballclips prototype UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")

from gi.repository import Gst, Gtk


def _find_first_mp4(paths: Iterable[Path]) -> Path:
    for path in sorted(paths):
        if path.is_file() and path.suffix.lower() == ".mp4":
            return path
    raise FileNotFoundError("No MP4 files found in the provided folder.")


class PlayerWindow(Gtk.ApplicationWindow):
    """Main window that embeds a GStreamer-backed video player."""

    def __init__(self, app: "BallclipsApplication", video_file: Path) -> None:
        super().__init__(application=app)
        self.set_title("ballclips preview")
        self.set_default_size(960, 540)

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

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        container.set_border_width(12)
        container.pack_start(video_widget, True, True, 0)
        self.add(container)
        self.show_all()

        self.connect("destroy", self._on_destroy)

        self._configure_pipeline(video_file)

    def _configure_pipeline(self, video_file: Path) -> None:
        uri = Gst.filename_to_uri(str(video_file.resolve()))
        self._player.set_property("uri", uri)

        bus = self._player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._player.set_state(Gst.State.PLAYING)

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


class BallclipsApplication(Gtk.Application):
    """Gtk.Application wrapper for the prototype UI."""

    def __init__(self, video_file: Path) -> None:
        super().__init__(application_id="dev.ballclips.prototype")
        Gst.init(None)
        self._video_file = video_file
        self._window: PlayerWindow | None = None

    def do_activate(self) -> None:  # type: ignore[override]
        if self._window is None:
            self._window = PlayerWindow(self, self._video_file)
        self._window.present()

    def do_shutdown(self) -> None:  # type: ignore[override]
        if self._window is not None:
            self._window.destroy()
            self._window = None
        super().do_shutdown()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the first MP4 file in a folder.")
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
        video_file = _find_first_mp4(folder.iterdir())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    app = BallclipsApplication(video_file)
    return app.run(sys.argv[:1])


if __name__ == "__main__":
    raise SystemExit(main())
