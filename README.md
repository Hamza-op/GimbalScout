# Video Tool

Desktop video analysis tool written in Rust.

The default profile is accuracy-first for wedding footage: 18fps motion
sampling, automatic per-clip motion thresholds, half-window overlap, and
temporal person evidence when YOLO is enabled. It is deliberately conservative
about operator shake and tolerant of brief subject-detection dropouts.

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

Analysis results are cached beside the selected input folder. Cache entries are
validated against the source metadata, analysis settings, and the full SHA-256
digest of the model. Moving an unchanged model does not discard useful cache
data, while replacing its content always invalidates affected entries.

### Profiles

- **Movement**: a 144px grayscale / 18fps stream optimized for camera motion.
- **People + Motion**: 720px / 18fps with the bundled YOLO model when subjects
  matter. Camera motion is still evaluated on a normalized 144px thumbnail.

Both profiles export one highest-scoring select from every analyzed video, so
no source clip disappears from the Premiere workflow. Advanced controls expose
the motion threshold, analysis-window duration, and worker count when tuning is
needed.

The motion threshold should normally remain **Auto**. A fixed threshold is
available for matching a known camera or shooting style.

### Premiere clip colors

Every exported select has an explicit Premiere label:

| Select type | Premiere label |
| --- | --- |
| Static fallback | Cerulean |
| Static subject | Caribbean |
| Slow motion | Iris |
| Pan/tilt | Forest |
| Zoom | Mango |
| Roll | Lavender |
| Complex move | Rose |

The exporter validates the complete XML in memory and replaces the previous
file atomically. Missing or invalid source selections cause an error and leave
the last known-good XML untouched.

## CPU compatibility

The normal release uses ONNX Runtime's CPU provider and Rust's standard x64
target. It does not enable `target-cpu=native`, AVX-only code, or vendor-specific
Intel/AMD paths. Supported AMD and Intel x64 processors therefore use the same
portable code path. GPU providers remain optional build features; a GPU is not
required. A build without YOLO is also supported with
`--no-default-features`.

## Release workflow

GitHub Actions publishes release builds when you push a tag that starts with `v`.

Example:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds release binaries for:

- Windows
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
