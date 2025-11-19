# ballclips

Batch video trimming, cropping tool, and composing tool for crystal ball video players.

![Crystal ball video](./assets/crystal_ball.gif)

Many Chinese sellers<sup>[1](https://a.aliexpress.com/_mO5snMN), [2](https://a.aliexpress.com/_m0JcoI9)</sup> sell a product that is a half-sphere crystal ball attached to a video screen so that the viewer can see videos playing "inside" the crystal ball (kind of like Harry Potter wizard portraits).  The downsides of these products are that videos need to have specific characteristics, and there isn't a "shuffle play" option -- instead, a specific video must be selected by hand to play.

This project allows the owner of one (or more) of these crystal balls to assemble a large number of normal video clips into a single square-aspect-ratio video suitable for playing on the crystal ball.

## Requirements

* Linux (confirmed to work in an Ubuntu VM on a Windows [VirtualBox](https://www.virtualbox.org/) host)
* [uv](https://github.com/astral-sh/uv) for dependency management and running the app
* Prerequisite packages (see below)
* GStreamer 1.0 with the `gtksink` plugin (commonly available via `gstreamer1.0-plugins-good`)
* GTK 3 and the PyGObject introspection bindings

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

Place one or more `.mp4` source files in a folder and run:

```bash
uv run ballclips <path-to-folder-with-videos>
```

* Set the start and end frames with the `{` and `}` buttons
* Navigate to the start or end frames with the `→{` and `→}` buttons
* When paused on a start or end frame:
  * Drag the crop region by right-clicking and dragging
  * Define a new crop region by left-clicking and dragging
* Navigate between source videos with the `>` and `<` buttons
* Navigate between source videos without cropping information with the `>>` and `<<` buttons
* Once all videos have been assigned crop regions (if they shouldn't have the default centered crop region), proceed to export below

To improve performance:

```bash
sudo nice -n -5 "$(command -v uv)" run ballclips ~/shared
```

To debug video behavior, `export GST_DEBUG=3`.

## Exporting cropped clips

Once you have saved crop metadata alongside your source videos, you can batch
export them into a single compilation. The `ballclips-export` CLI is published
with this project, so you can invoke it directly with `uv run`:

```bash
uv run ballclips-export <path-to-video-or-folder>
```

You can pass either a single MP4 file or a folder containing multiple MP4s.
By default the compilation is saved as `ballclips_compilation.mp4` in the
current directory. Pass `--output` to choose a different destination or
filename and `--seed` to make the shuffling of clips deterministic:

```bash
uv run ballclips-export ~/shared/clips --output ~/exports/my-compilation.mp4 --seed 1234
```

The export process requires `ffmpeg` to be installed and available on your
`PATH`.
