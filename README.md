# Video Tool

Desktop video analysis tool written in Rust.

The default profile is now accuracy-first for wedding footage: 720px analysis,
18fps motion sampling, automatic per-clip motion thresholds, and temporal person
evidence when YOLO is enabled. It is deliberately more conservative about
operator shake and more tolerant of brief subject-detection dropouts.

## What the analysis produces

The analyser:

1. Decodes a lightweight analysis stream with FFmpeg (the source media is never
   re-encoded).
2. Estimates global camera motion from a grid of tracked textured patches using
   a robust affine model. Local subject movement, exposure changes, and flashes
   are rejected rather than treated as camera moves.
3. Measures temporal direction/speed consistency so a smooth gimbal move is
   separated from a short handheld jerk.
4. Samples YOLO several times per analysis window and combines the evidence,
   instead of deciding from one centre frame.
5. Merges overlapping windows, keeps usable clip-edge material, and writes
   Premiere-compatible XML with source-accurate in/out points.

Analysis results are cached beside the selected input folder. Changing the
analysis settings or model automatically invalidates stale results.

### Profiles

- **Bulk Fast**: 540px / 12fps, motion-only.
- **Movement**: 720px / 18fps, motion-only.
- **Best Motion**: 720px / 24fps, shortest temporal window.
- **People + Motion**: 720px / 18fps with the bundled YOLO model (the default
  accuracy profile).

The motion threshold should normally remain **Auto**. A fixed threshold is
available for matching a known camera or shooting style.

## Release workflow

GitHub Actions publishes release builds when you push a tag that starts with `v`.

Example:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds release binaries for:

- Windows
- macOS
- Linux

Release assets are uploaded to the matching GitHub Release automatically.

## Runtime dependency note

Release workflow builds use the default feature set: embedded assets plus CPU YOLO.
The YOLO model is committed as `assets/yolo.onnx`, so GitHub Actions does not need a
separate secret just to build releases.

Release assets include FFmpeg tools:

- Windows embeds `ffmpeg.exe` and `ffprobe.exe` into `video-tool.exe` as a compressed
  archive during CI by running `scripts/download_tools.ps1` before compiling. On first
  setup/run, the tools are extracted under the app config directory.
- macOS releases are published as `video-tool-macos-x64.dmg` and include
  `tools/ffmpeg` and `tools/ffprobe` inside the `.app` bundle.
- Linux releases are published as `video-tool-linux-x64.AppImage` and include
  `tools/ffmpeg` and `tools/ffprobe` inside the AppImage.
- YOLO is embedded into all release binaries from `assets/yolo.onnx`.

Linux packages still assume the normal desktop GUI libraries available on common
desktop distributions. FFmpeg itself is bundled.

Local developer builds still use default features: embedded assets plus CPU YOLO. GPU
providers such as DirectML are opt-in, for example:

```powershell
cargo build --release --features directml
```
