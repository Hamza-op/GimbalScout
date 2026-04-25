# Video Tool

Desktop video analysis tool written in Rust.

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
