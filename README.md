# ballclips

Prototype UI scaffolding for a future batch video trimming and cropping tool.

## Requirements

* [uv](https://github.com/astral-sh/uv) for dependency management and running the app.
* GStreamer 1.0 with the `gtksink` plugin (commonly available via `gstreamer1.0-plugins-good`).
* GTK 3 and the PyGObject introspection bindings.

## Running the prototype

Place one or more `.mp4` files in a folder and run:

```bash
uv run ballclips <path-to-folder>
```

The application will open a GTK window that plays the first MP4 file found in the
specified folder. If no folder is given, the current working directory is used.
