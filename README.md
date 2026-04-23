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

Windows builds can use the embedded `assets/ffmpeg.exe` and `assets/ffprobe.exe`.

macOS and Linux builds are not self-contained for FFmpeg right now. Those users need
`ffmpeg` and `ffprobe` installed on `PATH`, because the current embedded asset setup is
Windows-named only.
