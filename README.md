# Startrails 🌌

Generate startrail images and videos from a project's image sequence. The script guides you through selecting a project, choosing a frame range, optional compression, and output types (startrail image, startrail video, starmotion video), with optional Instagram-compatible outputs.

## Project Layout

```
/projects/<project>/
  images/
  compressed/         (auto-created if you choose to compress)
  result/             (auto-created for outputs)
```

The script requires `/projects` to exist.

## Create A Project

1. Create a new project folder under `/projects`.
2. Put your source images inside the `images/` folder.

Example:

```bash
mkdir -p /projects/my_night_shoot/images
cp /path/to/your/images/*.jpg /projects/my_night_shoot/images/
```

## Install

This repo uses `uv` and Python 3.11+.

```bash
git clone https://github.com/midlajc/startrail.git
cd startrail
```

```bash
uv sync
```

## Use

```bash
uv run cmd/startrails.py
```

## Prompts (Order)

1. Project selection
2. Start/end range of images
3. Compress images first?
4. Outputs to generate
5. If any video: desired duration (seconds)
6. Instagram-compatible versions?

## Outputs

All outputs are written under:

```
/projects/<project>/result
```

Instagram versions end with `_insta` and are resized to max 1080px on the long side to reduce Instagram recompression.
