# ballclips

Batch video trimming, cropping tool, and composing tool for crystal ball video players.

## Requirements

* [uv](https://github.com/astral-sh/uv) for dependency management and running the app.
* Prerequisite packages (see below)
* GStreamer 1.0 with the `gtksink` plugin (commonly available via `gstreamer1.0-plugins-good`).
* GTK 3 and the PyGObject introspection bindings.

```bash
sudo apt install -y \
  pkg-config \
  libcairo2-dev \
  build-essential \
  python3-dev \
  libgirepository-2.0-dev \
  libglib2.0-dev \
  gobject-introspection \
  libffi-dev \
  gstreamer1.0-gtk3 \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-vaapi \
  gstreamer1.0-libav
```

When using inside a VM:

```bash
sudo apt install -y \
  vainfo \
  mesa-va-drivers
```

## Running the tool

Place one or more `.mp4` files in a folder and run:

```bash
uv run ballclips <path-to-folder-with-videos>
```

To improve performance:

```bash
sudo nice -n -5 "$(command -v uv)" run ballclips ~/shared
```

To debug video behavior, `export GST_DEBUG=3`.
