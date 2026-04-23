use std::path::{Path, PathBuf};
#[cfg(feature = "embedded-assets")]
use std::time::{Duration, SystemTime};

use tracing::{debug, info, warn};

use crate::engine::AnalyzeArgs;
use crate::error::{AppError, AppResult};
use crate::settings::PersistedSettings;

#[derive(Debug, Clone)]
pub struct AssetConfig {
    pub ffmpeg_override: Option<PathBuf>,
    pub ffprobe_override: Option<PathBuf>,
    pub yolo_override: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub ffmpeg_bin: PathBuf,
    pub ffprobe_bin: PathBuf,
    pub yolo_model: Option<PathBuf>,
    pub enable_yolo: bool,
    pub config_fingerprint: String,

    pub analysis_height: u32,
    pub analysis_fps: f32,
    pub window_seconds: f32,
    pub motion_threshold: f32,
    pub person_confidence: f32,

    /// ORT intra-op thread count for the YOLO session.
    /// Computed automatically as `(cpus / parallel_files).max(1).min(4)`,
    /// or overridden via the GUI / caller.
    #[cfg_attr(not(feature = "yolo"), allow(dead_code))]
    pub yolo_intra_threads: usize,

    /// Number of threads ffmpeg uses for decoding (`-threads N`).
    /// 0 = let ffmpeg auto-detect (default).
    pub ffmpeg_threads: usize,

    /// How many raw frames the BufReader capacity should span.
    /// Higher values trade RAM for fewer syscalls on high-bitrate streams.
    pub buf_frames: usize,
}

impl AnalysisConfig {
    /// Build an `AnalysisConfig` from the given arguments.
    ///
    /// When `persisted` is `Some`, cached tool paths are tried first and the
    /// expensive embedded-asset extraction is skipped entirely if they are
    /// still valid on disk.  After a successful resolution the paths are
    /// written back so the *next* launch is equally fast.
    pub fn from_args(
        args: &AnalyzeArgs,
        assets: AssetConfig,
        persisted: Option<&mut PersistedSettings>,
    ) -> AppResult<Self> {
        if args.analysis_height == 0 {
            return Err(AppError::Unsupported(
                "analysis_height must be > 0".to_string(),
            ));
        }
        if args.analysis_fps <= 0.0 {
            return Err(AppError::Unsupported(
                "analysis_fps must be > 0".to_string(),
            ));
        }
        if args.window_seconds <= 0.0 {
            return Err(AppError::Unsupported(
                "window_seconds must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&args.person_confidence) {
            return Err(AppError::Unsupported(
                "person_confidence must be in [0,1]".to_string(),
            ));
        }
        if args.max_files == Some(0) {
            return Err(AppError::Unsupported(
                "max_files must be greater than 0".to_string(),
            ));
        }

        // ── Fast path: try cached tool paths from settings ──────────
        let (ffmpeg_bin, ffprobe_bin, yolo_model) = if let Some(ref settings) = persisted {
            if has_user_overrides(&assets, args.enable_yolo) {
                // User explicitly set overrides — honour them, skip cache.
                debug!("User overrides present — bypassing cached paths");
                resolve_assets(assets, args.enable_yolo)?
            } else if let Some(cached) = try_cached_paths(settings, args.enable_yolo) {
                info!("Using cached tool paths from settings (skipping extraction)");
                cached
            } else {
                debug!("Cached paths invalid or missing — falling back to extraction");
                resolve_assets(assets, args.enable_yolo)?
            }
        } else {
            resolve_assets(assets, args.enable_yolo)?
        };

        // Existence checks only — no subprocess spawns.
        check_exists(&ffprobe_bin, "ffprobe", "set FFprobe path in the GUI")?;
        check_exists(&ffmpeg_bin, "ffmpeg", "set FFmpeg path in the GUI")?;
        if args.enable_yolo
            && let Some(ref model) = yolo_model
            && !model.exists()
        {
            return Err(AppError::MissingDependency {
                what: "YOLO model file",
                hint: "set YOLO Model path in the GUI (or embed assets)",
            });
        }

        // ── Persist resolved paths for next launch ──────────────────
        if let Some(settings) = persisted {
            if args.enable_yolo {
                settings.set_resolved_paths(&ffmpeg_bin, &ffprobe_bin, yolo_model.as_deref());
            } else {
                settings.resolved_paths.ffmpeg = Some(ffmpeg_bin.to_string_lossy().into_owned());
                settings.resolved_paths.ffprobe = Some(ffprobe_bin.to_string_lossy().into_owned());
            }
            if let Err(e) = settings.save() {
                warn!("Failed to persist settings: {e}");
            }
        }

        // Share the CPU budget across file workers, ffmpeg decode threads
        // and ONNX intra-op threads.
        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let parallel_files = args
            .max_files
            .unwrap_or_else(|| total_cpus.div_ceil(2).clamp(1, 8));
        let per_file_budget = (total_cpus / parallel_files.max(1)).max(1);
        let yolo_intra_threads = args
            .yolo_intra_threads
            .unwrap_or_else(|| per_file_budget.div_ceil(2).clamp(1, 4));
        let ffmpeg_threads = args
            .ffmpeg_threads
            .unwrap_or_else(|| per_file_budget.clamp(1, 4));
        let buf_frames = args.buf_frames.unwrap_or(8);

        let mut config = Self {
            ffmpeg_bin,
            ffprobe_bin,
            yolo_model,
            enable_yolo: args.enable_yolo,
            config_fingerprint: String::new(),
            analysis_height: args.analysis_height,
            analysis_fps: args.analysis_fps,
            window_seconds: args.window_seconds,
            motion_threshold: args.motion_threshold,
            person_confidence: args.person_confidence,
            yolo_intra_threads,
            ffmpeg_threads,
            buf_frames,
        };
        config.config_fingerprint = crate::cache::config_fingerprint(&config);
        Ok(config)
    }
}

/// Returns `true` when the user has set explicit path overrides in the GUI.
fn has_user_overrides(assets: &AssetConfig, enable_yolo: bool) -> bool {
    assets.ffmpeg_override.is_some()
        || assets.ffprobe_override.is_some()
        || (enable_yolo && assets.yolo_override.is_some())
}

/// Attempt to use the tool paths cached in `PersistedSettings`.
/// Returns `None` if any required path is missing or no longer exists.
fn try_cached_paths(
    settings: &PersistedSettings,
    enable_yolo: bool,
) -> Option<(PathBuf, PathBuf, Option<PathBuf>)> {
    let ffmpeg = settings.cached_ffmpeg()?;
    let ffprobe = settings.cached_ffprobe()?;
    let yolo = if enable_yolo {
        settings.cached_yolo()
    } else {
        None
    };

    // If YOLO is missing from the cache but we have embedded assets enabled,
    // we must reject the cache and fall back to extraction to ensure it's available.
    #[cfg(feature = "embedded-assets")]
    if enable_yolo {
        yolo.as_ref()?;
    }

    Some((ffmpeg, ffprobe, yolo))
}

/// Lightweight pre-flight check: verify the binary path exists on disk without
/// spawning a subprocess.  Skipped for bare names like "ffmpeg" (PATH lookup)
/// because `exists()` on a bare name checks the current directory, not PATH.
/// Those cases are caught quickly by the first real invocation in the pipeline.
fn check_exists(bin: &Path, what: &'static str, hint: &'static str) -> AppResult<()> {
    // A path with more than one component means the caller gave an explicit
    // path (absolute or relative) — we can meaningfully test existence.
    if bin.components().count() > 1 && !bin.exists() {
        return Err(AppError::MissingDependency { what, hint });
    }
    Ok(())
}

fn resolve_assets(
    assets: AssetConfig,
    enable_yolo: bool,
) -> AppResult<(PathBuf, PathBuf, Option<PathBuf>)> {
    let ffmpeg = if let Some(p) = assets.ffmpeg_override {
        p
    } else if let Some(p) = find_binary_on_path("ffmpeg") {
        p
    } else if let Some(p) = try_extract_embedded("ffmpeg.exe")? {
        p
    } else {
        PathBuf::from("ffmpeg")
    };

    let ffprobe = if let Some(p) = assets.ffprobe_override {
        p
    } else if let Some(p) = find_binary_on_path("ffprobe") {
        p
    } else if let Some(p) = try_extract_embedded("ffprobe.exe")? {
        p
    } else {
        PathBuf::from("ffprobe")
    };

    // YOLO model: extracted only when person detection is enabled by caller.
    let yolo = if !enable_yolo {
        None
    } else if assets.yolo_override.is_some() {
        assets.yolo_override
    } else {
        try_extract_embedded("yolo.onnx")?
    };

    Ok((ffmpeg, ffprobe, yolo))
}

/// Prefer already-installed binaries over extracting embedded copies.
/// This keeps the first analysis run much snappier on machines that already
/// have ffmpeg / ffprobe available in PATH.
fn find_binary_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let paths = std::env::split_paths(&path_var);

    #[cfg(target_os = "windows")]
    let candidates: Vec<String> = {
        let mut out = vec![name.to_string()];
        out.push(format!("{name}.exe"));
        out
    };

    #[cfg(not(target_os = "windows"))]
    let candidates: Vec<String> = vec![name.to_string()];

    for dir in paths {
        for candidate in &candidates {
            let full = dir.join(candidate);
            if full.is_file() {
                return Some(full);
            }
        }
    }

    None
}

fn try_extract_embedded(name: &str) -> AppResult<Option<PathBuf>> {
    #[cfg(feature = "embedded-assets")]
    {
        extract_embedded_asset(name)
    }

    #[cfg(not(feature = "embedded-assets"))]
    {
        let _ = name;
        Ok(None)
    }
}

#[cfg(feature = "embedded-assets")]
#[derive(rust_embed::RustEmbed)]
#[folder = "assets"]
struct EmbeddedAssets;

#[cfg(feature = "embedded-assets")]
fn extract_embedded_asset(name: &str) -> AppResult<Option<PathBuf>> {
    use sha2::{Digest, Sha256};

    let Some(asset) = EmbeddedAssets::get(name) else {
        debug!("Embedded asset not present: {name}");
        return Ok(None);
    };

    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(name.as_bytes());
    hasher.update((asset.data.len() as u64).to_le_bytes());
    let sample_len = asset.data.len().min(4096);
    hasher.update(&asset.data[..sample_len]);
    let hash = format!("{:x}", hasher.finalize());

    let base = std::env::temp_dir().join("video-tool-assets").join(hash);
    let out_path = base.join(name);
    if out_path.exists() {
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(&base).map_err(|e| AppError::Io {
        path: base.clone(),
        source: e,
    })?;

    // Simple cross-process lock to avoid partial writes on concurrent runs.
    let lock_path = base.join("extract.lock");
    let lock_acquired = acquire_lock(&lock_path, Duration::from_secs(30))?;
    if !lock_acquired {
        return Err(AppError::Message(format!(
            "timed out waiting for asset extraction lock: {}",
            lock_path.display()
        )));
    }

    // Re-check after lock.
    if out_path.exists() {
        release_lock(&lock_path);
        return Ok(Some(out_path));
    }

    let tmp = base.join(format!("{name}.tmp"));
    std::fs::write(&tmp, &asset.data).map_err(|e| AppError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, &out_path).map_err(|e| AppError::Io {
        path: out_path.clone(),
        source: e,
    })?;

    info!(
        "Extracted embedded asset: {} -> {}",
        name,
        out_path.display()
    );
    release_lock(&lock_path);
    Ok(Some(out_path))
}

/// Result of a successful tool setup.
#[derive(Debug, Clone)]
pub struct SetupToolsResult {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub yolo_model: Option<PathBuf>,
}

/// Pre-resolve and extract all external tools (ffmpeg, ffprobe, YOLO model)
/// so that the analysis `from_args` fast-path hits immediately.
///
/// `on_progress` receives human-readable phase labels like
/// "Searching PATH for ffmpeg…", "Extracting YOLO model…", etc.
///
/// On success the resolved paths are persisted to `settings` and returned.
pub fn setup_tools(
    ffmpeg_override: Option<PathBuf>,
    ffprobe_override: Option<PathBuf>,
    yolo_override: Option<PathBuf>,
    enable_yolo: bool,
    settings: &mut PersistedSettings,
    mut on_progress: impl FnMut(&str),
) -> AppResult<SetupToolsResult> {
    on_progress("Resolving ffmpeg…");
    let ffmpeg = resolve_single_tool("ffmpeg", ffmpeg_override)?;
    check_exists(&ffmpeg, "ffmpeg", "set FFmpeg path in the GUI")?;
    on_progress(&format!("ffmpeg → {}", ffmpeg.display()));

    on_progress("Resolving ffprobe…");
    let ffprobe = resolve_single_tool("ffprobe", ffprobe_override)?;
    check_exists(&ffprobe, "ffprobe", "set FFprobe path in the GUI")?;
    on_progress(&format!("ffprobe → {}", ffprobe.display()));

    let yolo_model = if enable_yolo {
        on_progress("Resolving YOLO model…");
        let model = if let Some(p) = yolo_override {
            p
        } else if let Some(p) = try_extract_embedded("yolo.onnx")? {
            p
        } else {
            return Err(AppError::MissingDependency {
                what: "YOLO model file",
                hint: "set YOLO Model path in the GUI (or embed assets)",
            });
        };
        if !model.exists() {
            return Err(AppError::MissingDependency {
                what: "YOLO model file",
                hint: "set YOLO Model path in the GUI (or embed assets)",
            });
        }
        on_progress(&format!("YOLO → {}", model.display()));
        Some(model)
    } else {
        on_progress("YOLO detection disabled — skipping model");
        None
    };

    // Persist so from_args hits the fast path next time.
    on_progress("Saving resolved paths…");
    if enable_yolo {
        settings.set_resolved_paths(&ffmpeg, &ffprobe, yolo_model.as_deref());
    } else {
        settings.resolved_paths.ffmpeg = Some(ffmpeg.to_string_lossy().into_owned());
        settings.resolved_paths.ffprobe = Some(ffprobe.to_string_lossy().into_owned());
    }
    if let Err(e) = settings.save() {
        warn!("Failed to persist settings after setup: {e}");
    }

    on_progress("All tools ready ✓");
    Ok(SetupToolsResult {
        ffmpeg,
        ffprobe,
        yolo_model,
    })
}

/// Resolve a single tool binary: user override → PATH → embedded extraction → bare name.
fn resolve_single_tool(name: &str, user_override: Option<PathBuf>) -> AppResult<PathBuf> {
    if let Some(p) = user_override {
        return Ok(p);
    }
    if let Some(p) = find_binary_on_path(name) {
        return Ok(p);
    }
    #[cfg(target_os = "windows")]
    {
        let exe_name = format!("{name}.exe");
        if let Some(p) = try_extract_embedded(&exe_name)? {
            return Ok(p);
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Some(p) = try_extract_embedded(name)? {
            return Ok(p);
        }
    }
    Ok(PathBuf::from(name))
}

#[cfg(feature = "embedded-assets")]
fn acquire_lock(lock_path: &Path, timeout: Duration) -> AppResult<bool> {
    let start = SystemTime::now();
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(mut f) => {
                let _ = std::io::Write::write_all(&mut f, b"lock");
                return Ok(true);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if start.elapsed().unwrap_or(Duration::from_secs(0)) > timeout {
                    return Ok(false);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Err(AppError::Io {
                    path: lock_path.to_path_buf(),
                    source: e,
                });
            }
        }
    }
}

#[cfg(feature = "embedded-assets")]
fn release_lock(lock_path: &Path) {
    if let Err(e) = std::fs::remove_file(lock_path) {
        warn!("Failed to remove lock {}: {}", lock_path.display(), e);
    }
}
