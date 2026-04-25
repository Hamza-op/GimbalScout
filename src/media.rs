use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use tracing::{debug, warn};

use crate::error::{AppError, AppResult};

const DISCOVERY_CACHE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeInfo {
    pub source_path: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Total duration in seconds. Retained for XML export and future CLI output.
    #[allow(dead_code)]
    pub duration_seconds: f64,
    pub duration_frames: u64,
    /// Numerator of the stream average frame rate fraction.
    #[allow(dead_code)]
    pub fps_num: u32,
    /// Denominator of the stream average frame rate fraction.
    #[allow(dead_code)]
    pub fps_den: u32,
    pub timebase: u32,
    pub ntsc: bool,
    #[serde(default)]
    pub slow_motion: bool,
    #[serde(default)]
    pub capture_fps: Option<u32>,
    #[serde(default)]
    pub format_fps: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScanProgress {
    pub entries_scanned: usize,
    pub matches_found: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiscoveryCacheFile {
    schema_version: u32,
    extension_fingerprint: String,
    root: PathBuf,
    directories: Vec<DirectoryCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DirectoryCacheEntry {
    path: PathBuf,
    mtime_nanos: u128,
    direct_matches: Vec<PathBuf>,
    child_dirs: Vec<PathBuf>,
    all_matches: Vec<PathBuf>,
}

#[derive(Debug, Default)]
struct DiscoveryCache {
    directories: HashMap<PathBuf, DirectoryCacheEntry>,
}

struct DiscoverContext<'a, F, G>
where
    F: FnMut(ScanProgress),
    G: FnMut(PathBuf) -> AppResult<()>,
{
    root: &'a Path,
    extensions: &'a [String],
    cached: Option<&'a DiscoveryCache>,
    next_cache: &'a mut DiscoveryCache,
    progress: &'a mut ScanProgress,
    on_progress: &'a mut F,
    on_match: &'a mut G,
}

pub fn discover_inputs_streaming<F, G>(
    root: &Path,
    cache_root: &Path,
    extensions: &[String],
    on_progress: &mut F,
    mut on_match: G,
) -> AppResult<usize>
where
    F: FnMut(ScanProgress),
    G: FnMut(PathBuf) -> AppResult<()>,
{
    let fingerprint = extensions_fingerprint(extensions);
    let cached = load_discovery_cache(root, cache_root, &fingerprint);
    let mut next_cache = DiscoveryCache::default();
    let mut progress = ScanProgress::default();
    let mut ctx = DiscoverContext {
        root,
        extensions,
        cached: cached.as_ref(),
        next_cache: &mut next_cache,
        progress: &mut progress,
        on_progress,
        on_match: &mut on_match,
    };
    let matched = discover_dir(root, &mut ctx)?;
    on_progress(progress);
    if let Err(e) = save_discovery_cache(root, cache_root, &fingerprint, next_cache) {
        warn!("discovery cache save failed for {}: {e}", root.display());
    }
    Ok(matched)
}

fn matches_extension(ext: &str, extensions: &[String]) -> bool {
    extensions
        .iter()
        .any(|allowed| ext.eq_ignore_ascii_case(allowed))
}

fn discover_dir<F, G>(dir: &Path, ctx: &mut DiscoverContext<'_, F, G>) -> AppResult<usize>
where
    F: FnMut(ScanProgress),
    G: FnMut(PathBuf) -> AppResult<()>,
{
    ctx.progress.entries_scanned += 1;
    if ctx.progress.entries_scanned.is_multiple_of(32) {
        (ctx.on_progress)(*ctx.progress);
    }

    let mtime_nanos = file_mtime_nanos(dir)?;
    if let Some(entry) = ctx
        .cached
        .and_then(|cache| cache.directories.get(dir))
        .filter(|entry| entry.mtime_nanos == mtime_nanos)
    {
        let mut total = entry.direct_matches.len();
        for path in &entry.direct_matches {
            (ctx.on_match)(path.clone())?;
        }
        ctx.progress.matches_found = ctx
            .progress
            .matches_found
            .saturating_add(entry.direct_matches.len());
        ctx.next_cache
            .directories
            .insert(dir.to_path_buf(), entry.clone());
        for child_dir in &entry.child_dirs {
            total += discover_dir(child_dir, ctx)?;
        }
        return Ok(total);
    }

    let read_dir = fs::read_dir(dir).map_err(|e| AppError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;

    let mut direct_matches = Vec::new();
    let mut child_dirs_to_scan = Vec::new();
    let mut child_dirs = Vec::new();
    let mut all_matches = Vec::new();

    for entry in read_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                warn!("read_dir error under {}: {e}", dir.display());
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                warn!("file_type error for {}: {e}", path.display());
                continue;
            }
        };

        if file_type.is_dir() {
            if should_descend(ctx.root, &path) {
                child_dirs.push(path.clone());
                let child_mtime_nanos = match file_mtime_nanos(&path) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("mtime read error for {}: {e}", path.display());
                        child_dirs_to_scan.push(path.clone());
                        continue;
                    }
                };

                if ctx
                    .cached
                    .and_then(|cache| cache.directories.get(&path))
                    .is_some_and(|entry| entry.mtime_nanos == child_mtime_nanos)
                {
                    if let Some(entry) = ctx.cached.and_then(|cache| cache.directories.get(&path)) {
                        for matched in &entry.all_matches {
                            (ctx.on_match)(matched.clone())?;
                        }
                        ctx.progress.matches_found = ctx
                            .progress
                            .matches_found
                            .saturating_add(entry.all_matches.len());
                        all_matches.extend(entry.all_matches.iter().cloned());
                        clone_cached_subtree(&path, ctx.cached, ctx.next_cache);
                    }
                } else {
                    child_dirs_to_scan.push(path.clone());
                }
            }
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        if matches_extension(ext, ctx.extensions) {
            direct_matches.push(path);
        }
    }

    child_dirs.sort_unstable();
    child_dirs_to_scan.sort_unstable();
    direct_matches.sort_unstable();
    ctx.progress.matches_found = ctx
        .progress
        .matches_found
        .saturating_add(direct_matches.len());
    for path in &direct_matches {
        (ctx.on_match)(path.clone())?;
    }

    let mut total = direct_matches.len();
    all_matches.extend(direct_matches.iter().cloned());

    for child_dir in &child_dirs_to_scan {
        total += discover_dir(child_dir, ctx)?;
        if let Some(entry) = ctx.next_cache.directories.get(child_dir) {
            all_matches.extend(entry.all_matches.iter().cloned());
        }
    }

    all_matches.sort_unstable();

    ctx.next_cache.directories.insert(
        dir.to_path_buf(),
        DirectoryCacheEntry {
            path: dir.to_path_buf(),
            mtime_nanos,
            direct_matches,
            child_dirs,
            all_matches,
        },
    );

    Ok(total)
}

fn clone_cached_subtree(
    dir: &Path,
    cached: Option<&DiscoveryCache>,
    next_cache: &mut DiscoveryCache,
) {
    let Some(cache) = cached else {
        return;
    };
    let Some(entry) = cache.directories.get(dir) else {
        return;
    };
    next_cache
        .directories
        .insert(dir.to_path_buf(), entry.clone());
    for child_dir in &entry.child_dirs {
        clone_cached_subtree(child_dir, cached, next_cache);
    }
}

fn should_descend(root: &Path, path: &Path) -> bool {
    if path == root {
        return true;
    }

    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return true;
    };

    !matches!(
        name,
        ".cache"
            | ".git"
            | "target"
            | "node_modules"
            | "$RECYCLE.BIN"
            | "System Volume Information"
    )
}

fn extensions_fingerprint(extensions: &[String]) -> String {
    let mut normalized = extensions
        .iter()
        .map(|ext| ext.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    normalized.sort_unstable();
    normalized.dedup();
    normalized.join(",")
}

fn discovery_cache_path(cache_root: &Path) -> PathBuf {
    cache_root.join("discovery-cache.json")
}

fn load_discovery_cache(
    root: &Path,
    cache_root: &Path,
    fingerprint: &str,
) -> Option<DiscoveryCache> {
    let path = discovery_cache_path(cache_root);
    let data = fs::read(&path).ok()?;
    let parsed: DiscoveryCacheFile = serde_json::from_slice(&data).ok()?;
    if parsed.schema_version != DISCOVERY_CACHE_SCHEMA_VERSION
        || parsed.root != root
        || parsed.extension_fingerprint != fingerprint
    {
        return None;
    }

    let directories = parsed
        .directories
        .into_iter()
        .map(|entry| (entry.path.clone(), entry))
        .collect();
    Some(DiscoveryCache { directories })
}

fn save_discovery_cache(
    root: &Path,
    cache_root: &Path,
    fingerprint: &str,
    cache: DiscoveryCache,
) -> AppResult<()> {
    let path = discovery_cache_path(cache_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AppError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    let mut directories = cache.directories.into_values().collect::<Vec<_>>();
    directories.sort_by(|a, b| a.path.cmp(&b.path));
    let file = DiscoveryCacheFile {
        schema_version: DISCOVERY_CACHE_SCHEMA_VERSION,
        extension_fingerprint: fingerprint.to_string(),
        root: root.to_path_buf(),
        directories,
    };
    let json = serde_json::to_vec(&file).map_err(|e| AppError::ParseFailed {
        what: "serialise discovery cache",
        source: Box::new(e),
    })?;

    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, &json).map_err(|e| AppError::Io {
        path: tmp_path.clone(),
        source: e,
    })?;
    if path.exists() {
        fs::remove_file(&path).map_err(|e| AppError::Io {
            path: path.clone(),
            source: e,
        })?;
    }
    fs::rename(&tmp_path, &path).map_err(|e| AppError::Io {
        path: path.clone(),
        source: e,
    })?;
    Ok(())
}

fn file_mtime_nanos(path: &Path) -> AppResult<u128> {
    let metadata = fs::metadata(path).map_err(|e| AppError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0))
}

pub fn probe_video(input: &Path, ffprobe_bin: &Path) -> AppResult<ProbeInfo> {
    let mut cmd = Command::new(ffprobe_bin);
    suppress_child_console(&mut cmd);
    cmd.args([
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
    ]);
    cmd.arg(input);

    let cmd_str = format!("{} ... {}", ffprobe_bin.display(), input.display());
    let out = cmd.output().map_err(|e| AppError::CommandFailed {
        cmd: cmd_str,
        source: e,
    })?;

    if !out.status.success() {
        return Err(AppError::CommandNonZero {
            cmd: format!("{} (ffprobe)", ffprobe_bin.display()),
            code: out.status.code().unwrap_or(-1),
        });
    }

    let parsed: FfprobeOutput =
        serde_json::from_slice(&out.stdout).map_err(|e| AppError::ParseFailed {
            what: "ffprobe JSON output",
            source: Box::new(e),
        })?;

    let (stream, vfr_warn) = select_video_stream(&parsed)?;
    if vfr_warn {
        warn!("possible VFR input: {}", input.display());
    }

    let width = stream
        .width
        .ok_or_else(|| AppError::Unsupported("ffprobe missing stream width".to_string()))?;
    let height = stream
        .height
        .ok_or_else(|| AppError::Unsupported("ffprobe missing stream height".to_string()))?;
    let avg_frame_rate = stream
        .avg_frame_rate
        .as_deref()
        .ok_or_else(|| AppError::Unsupported("ffprobe missing avg_frame_rate".to_string()))?;
    let (fps_num, fps_den) = parse_rational(avg_frame_rate).ok_or_else(|| {
        AppError::Unsupported(format!("unsupported avg_frame_rate: {avg_frame_rate}"))
    })?;

    let duration_seconds = parsed
        .format
        .as_ref()
        .and_then(|f| f.duration.as_deref())
        .and_then(|d| d.parse::<f64>().ok())
        .ok_or_else(|| AppError::Unsupported("ffprobe missing format.duration".to_string()))?;

    let fps = fps_num as f64 / fps_den as f64;
    let duration_frames = (duration_seconds * fps).round().max(0.0) as u64;
    let (timebase, ntsc) = timebase_and_ntsc(fps_num, fps_den);
    let slow = probe_slow_motion_metadata(input, fps_num, fps_den);

    debug!(
        "probe {}: {}x{}, fps={}/{} duration={:.3}s slow_motion={}",
        input.display(),
        width,
        height,
        fps_num,
        fps_den,
        duration_seconds,
        slow.slow_motion
    );

    Ok(ProbeInfo {
        source_path: input.to_path_buf(),
        width,
        height,
        duration_seconds,
        duration_frames,
        fps_num,
        fps_den,
        timebase,
        ntsc,
        slow_motion: slow.slow_motion,
        capture_fps: slow.capture_fps,
        format_fps: slow.format_fps,
    })
}

fn suppress_child_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SlowMotionProbe {
    slow_motion: bool,
    capture_fps: Option<u32>,
    format_fps: Option<u32>,
}

fn probe_slow_motion_metadata(path: &Path, fps_num: u32, fps_den: u32) -> SlowMotionProbe {
    let mut probe = scan_embedded_sony_rtmd(path).unwrap_or_default();
    let container_fps = if fps_den == 0 {
        0.0
    } else {
        fps_num as f64 / fps_den as f64
    };
    if container_fps >= 48.0 && probe.capture_fps.is_none() {
        probe.capture_fps = Some(container_fps.round() as u32);
    }
    if let (Some(capture), Some(format)) = (probe.capture_fps, probe.format_fps)
        && format > 0
        && capture >= format.saturating_mul(2)
    {
        probe.slow_motion = true;
    }
    probe
}

fn scan_embedded_sony_rtmd(path: &Path) -> std::io::Result<SlowMotionProbe> {
    const HEAD_BYTES: u64 = 2 * 1024 * 1024;
    const TAIL_BYTES: u64 = 6 * 1024 * 1024;

    let mut file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let mut bytes = Vec::new();
    if len <= HEAD_BYTES + TAIL_BYTES {
        file.read_to_end(&mut bytes)?;
    } else {
        bytes.resize(HEAD_BYTES as usize, 0);
        file.read_exact(&mut bytes)?;

        let tail_start = len.saturating_sub(TAIL_BYTES);
        file.seek(SeekFrom::Start(tail_start))?;
        let old_len = bytes.len();
        bytes.resize(old_len + TAIL_BYTES as usize, 0);
        file.read_exact(&mut bytes[old_len..])?;
    }

    let text = String::from_utf8_lossy(&bytes);
    let capture_fps = parse_xml_fps_attr(&text, "captureFps");
    let format_fps = parse_xml_fps_attr(&text, "formatFps");
    let slow_motion = text.contains("slowAndQuickMotion");
    Ok(SlowMotionProbe {
        slow_motion,
        capture_fps,
        format_fps,
    })
}

fn parse_xml_fps_attr(text: &str, attr: &str) -> Option<u32> {
    let needle = format!("{attr}=\"");
    let start = text.find(&needle)? + needle.len();
    let value = text.get(start..)?.split('"').next()?;
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
}

fn select_video_stream(parsed: &FfprobeOutput) -> AppResult<(&FfprobeStream, bool)> {
    let stream = parsed
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| AppError::Unsupported("no video stream found".to_string()))?;

    let mut vfr_warn = false;
    if let (Some(avg), Some(r)) = (
        stream.avg_frame_rate.as_deref(),
        stream.r_frame_rate.as_deref(),
    ) && let (Some((an, ad)), Some((rn, rd))) = (parse_rational(avg), parse_rational(r))
    {
        let a = an as f64 / ad as f64;
        let b = rn as f64 / rd as f64;
        if a > 0.0 && ((a - b).abs() / a) > 0.01 {
            vfr_warn = true;
        }
    }

    Ok((stream, vfr_warn))
}

fn parse_rational(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('/')?;
    let num = a.trim().parse::<u32>().ok()?;
    let den = b.trim().parse::<u32>().ok()?;
    if den == 0 {
        return None;
    }
    Some((num, den))
}

fn timebase_and_ntsc(fps_num: u32, fps_den: u32) -> (u32, bool) {
    // Common NTSC rates
    if fps_num == 24000 && fps_den == 1001 {
        return (24, true);
    }
    if fps_num == 30000 && fps_den == 1001 {
        return (30, true);
    }
    if fps_num == 60000 && fps_den == 1001 {
        return (60, true);
    }

    if fps_den == 1 {
        return (fps_num, false);
    }

    let fps = fps_num as f64 / fps_den as f64;
    let rounded = fps.round().clamp(1.0, 240.0) as u32;
    (rounded, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn timebase_ntsc_common() {
        assert_eq!(timebase_and_ntsc(30000, 1001), (30, true));
        assert_eq!(timebase_and_ntsc(24000, 1001), (24, true));
        assert_eq!(timebase_and_ntsc(60000, 1001), (60, true));
        assert_eq!(timebase_and_ntsc(25, 1), (25, false));
    }

    #[test]
    fn parse_rational_ok() {
        assert_eq!(parse_rational("30000/1001"), Some((30000, 1001)));
        assert_eq!(parse_rational(" 25 / 1 "), Some((25, 1)));
        assert_eq!(parse_rational("0/0"), None);
    }

    #[test]
    fn parses_sony_fps_attrs() {
        let text = r#"<VideoFrame captureFps="100p" formatFps="25p"/>"#;
        assert_eq!(parse_xml_fps_attr(text, "captureFps"), Some(100));
        assert_eq!(parse_xml_fps_attr(text, "formatFps"), Some(25));
    }

    fn tmp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join("video-tool-media-test")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn discover_inputs_matches_extensions_case_insensitively() {
        let root = tmp_root("case");
        fs::write(root.join("A.MP4"), b"").unwrap();
        fs::write(root.join("b.mov"), b"").unwrap();
        fs::write(root.join("c.txt"), b"").unwrap();

        let mut last = ScanProgress::default();
        let mut files = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into(), "mov".into()],
            &mut |p| last = p,
            |path| {
                files.push(path);
                Ok(())
            },
        )
        .unwrap();
        files.sort_unstable();

        assert_eq!(files.len(), 2);
        assert_eq!(last.matches_found, 2);
        assert!(files.iter().any(|p| p.ends_with("A.MP4")));
        assert!(files.iter().any(|p| p.ends_with("b.mov")));
    }

    #[test]
    fn discover_inputs_skips_cache_and_other_junk_dirs() {
        let root = tmp_root("skipdirs");
        let cache_dir = root.join(".cache");
        let nested = root.join("clips");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&nested).unwrap();
        fs::write(cache_dir.join("cached.mp4"), b"").unwrap();
        fs::write(nested.join("real.mp4"), b"").unwrap();

        let mut files = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into()],
            &mut |_| {},
            |path| {
                files.push(path);
                Ok(())
            },
        )
        .unwrap();
        files.sort_unstable();

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("real.mp4"));
    }

    #[test]
    fn discover_inputs_updates_cached_nested_subtree() {
        let root = tmp_root("nested-cache");
        let child = root.join("nested");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("one.mp4"), b"").unwrap();

        let mut first = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into()],
            &mut |_| {},
            |path| {
                first.push(path);
                Ok(())
            },
        )
        .unwrap();
        first.sort_unstable();
        assert_eq!(first.len(), 1);

        std::thread::sleep(std::time::Duration::from_millis(1200));
        fs::write(child.join("two.mp4"), b"").unwrap();

        let mut second = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into()],
            &mut |_| {},
            |path| {
                second.push(path);
                Ok(())
            },
        )
        .unwrap();
        second.sort_unstable();
        assert_eq!(second.len(), 2);
        assert!(second.iter().any(|p| p.ends_with("one.mp4")));
        assert!(second.iter().any(|p| p.ends_with("two.mp4")));
    }

    #[test]
    fn discover_inputs_reuses_child_subtree_when_root_changes() {
        let root = tmp_root("root-dirty");
        let child = root.join("nested");
        let grandchild = child.join("deep");
        fs::create_dir_all(&grandchild).unwrap();
        fs::write(grandchild.join("clip.mp4"), b"").unwrap();

        let mut first_progress = ScanProgress::default();
        let mut first_files = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into()],
            &mut |p| first_progress = p,
            |path| {
                first_files.push(path);
                Ok(())
            },
        )
        .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1200));
        fs::write(root.join("analysis.premiere.xml"), b"dummy").unwrap();

        let mut second_progress = ScanProgress::default();
        let mut second_files = Vec::new();
        discover_inputs_streaming(
            &root,
            &root.join(".cache"),
            &["mp4".into()],
            &mut |p| second_progress = p,
            |path| {
                second_files.push(path);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(first_files.len(), 1);
        assert_eq!(second_files.len(), 1);
        assert!(
            second_progress.entries_scanned < first_progress.entries_scanned,
            "expected subtree cache reuse after root-only change: first={}, second={}",
            first_progress.entries_scanned,
            second_progress.entries_scanned
        );
    }
}
