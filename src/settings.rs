use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::atomic_file;
use crate::error::{AppError, AppResult};
use crate::media;

/// Schema version — bump when the JSON shape changes incompatibly.
const SCHEMA_VERSION: u32 = 11;
const APP_DIR_NAME: &str = "video-tool";
const SETTINGS_FILE: &str = "settings.json";

// ──────────────────────────────────────────────
//  Persisted state
// ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSettings {
    /// Internal schema version for future migrations.
    pub version: u32,

    /// Resolved tool paths (absolute) from a previous extraction / config run.
    #[serde(default)]
    pub resolved_paths: ResolvedPaths,

    /// User preferences (GUI form values that should survive restarts).
    #[serde(default)]
    pub preferences: UserPreferences,

    /// Last successful export, retained so Results remains useful after an
    /// app restart and the XML can be reopened without another analysis.
    #[serde(default)]
    pub last_export: Option<PersistedExportSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedExportSummary {
    pub files_scanned: usize,
    pub files_analyzed: usize,
    pub cached_files: usize,
    pub exported_segments: usize,
    pub selected_duration_seconds: f64,
    pub movement_segments: usize,
    pub subject_segments: usize,
    pub slow_motion_segments: usize,
    pub static_segments: usize,
    pub audio_segments: usize,
    pub failed_paths: Vec<String>,
    pub output_path: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolvedPaths {
    pub ffmpeg: Option<String>,
    pub ffprobe: Option<String>,
    pub yolo_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Last-used input folder.
    #[serde(default)]
    pub last_input: String,

    /// Comma-separated extensions.
    #[serde(default = "default_extensions")]
    pub extensions: String,

    #[serde(default = "default_analysis_height")]
    pub analysis_height: u32,

    #[serde(default = "default_analysis_fps")]
    pub analysis_fps: f32,

    #[serde(default = "default_window_seconds")]
    pub window_seconds: f32,

    #[serde(default = "default_motion_threshold")]
    pub motion_threshold: f32,

    #[serde(default = "default_person_confidence")]
    pub person_confidence: f32,

    #[serde(default = "default_enable_yolo")]
    pub enable_yolo: bool,

    /// Optional linked source audio in the Premiere XML. Visual selects are
    /// the common workflow, so this deliberately defaults to off.
    #[serde(default)]
    pub include_audio: bool,

    /// Maximum duration of the peak-centered select exported per source.
    #[serde(default = "default_max_select_seconds")]
    pub max_select_seconds: f32,

    #[serde(default)]
    pub verbose: bool,

    /// User-specified overrides (empty string = auto-detect).
    #[serde(default)]
    pub ffmpeg_override: String,
    #[serde(default)]
    pub ffprobe_override: String,
    #[serde(default)]
    pub yolo_override: String,
}

fn default_extensions() -> String {
    media::DEFAULT_VIDEO_EXTENSIONS.to_string()
}
fn default_analysis_height() -> u32 {
    720
}
fn default_analysis_fps() -> f32 {
    18.0
}
fn default_window_seconds() -> f32 {
    1.0
}
fn default_motion_threshold() -> f32 {
    0.0 // 0.0 means Auto
}
fn default_person_confidence() -> f32 {
    0.42
}
fn default_enable_yolo() -> bool {
    cfg!(feature = "yolo")
}
fn default_max_select_seconds() -> f32 {
    8.0
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            last_input: String::new(),
            extensions: default_extensions(),
            analysis_height: default_analysis_height(),
            analysis_fps: default_analysis_fps(),
            window_seconds: default_window_seconds(),
            motion_threshold: default_motion_threshold(),
            person_confidence: default_person_confidence(),
            enable_yolo: default_enable_yolo(),
            include_audio: false,
            max_select_seconds: default_max_select_seconds(),
            verbose: false,
            ffmpeg_override: String::new(),
            ffprobe_override: String::new(),
            yolo_override: String::new(),
        }
    }
}

impl Default for PersistedSettings {
    fn default() -> Self {
        Self {
            version: SCHEMA_VERSION,
            resolved_paths: ResolvedPaths::default(),
            preferences: UserPreferences::default(),
            last_export: None,
        }
    }
}

// ──────────────────────────────────────────────
//  Path helpers
// ──────────────────────────────────────────────

/// Returns the OS-appropriate configuration directory for this app.
///
/// - Windows: `%LOCALAPPDATA%/video-tool/`
/// - macOS:   `~/Library/Application Support/video-tool/`
/// - Linux:   `$XDG_CONFIG_HOME/video-tool/` (defaults to `~/.config/video-tool/`)
pub fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("LOCALAPPDATA")
            .ok()
            .map(|base| PathBuf::from(base).join(APP_DIR_NAME))
    }

    #[cfg(target_os = "macos")]
    {
        dirs_fallback_home().map(|home| {
            home.join("Library")
                .join("Application Support")
                .join(APP_DIR_NAME)
        })
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let base = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs_fallback_home().map(|h| h.join(".config")));
        base.map(|b| b.join(APP_DIR_NAME))
    }
}

/// On non-Windows platforms, resolve `$HOME` for fallback paths.
#[cfg(not(target_os = "windows"))]
fn dirs_fallback_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

fn settings_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join(SETTINGS_FILE))
}

// ──────────────────────────────────────────────
//  Load / Save
// ──────────────────────────────────────────────

impl PersistedSettings {
    fn migrate(&mut self) -> bool {
        let mut changed = false;

        if self.version < 2 {
            if (self.preferences.motion_threshold - 12.0).abs() < f32::EPSILON {
                self.preferences.motion_threshold = default_motion_threshold();
            }
            self.version = 2;
            changed = true;
        }

        if self.version < 3 {
            if (self.preferences.analysis_fps - 4.0).abs() < f32::EPSILON {
                self.preferences.analysis_fps = default_analysis_fps();
            }
            self.version = 3;
            changed = true;
        }

        if self.version < 4 {
            if self.preferences.analysis_height == 480 {
                self.preferences.analysis_height = default_analysis_height();
            }
            if (self.preferences.window_seconds - 1.0).abs() < f32::EPSILON {
                self.preferences.window_seconds = default_window_seconds();
            }
            if (self.preferences.motion_threshold - 4.0).abs() < f32::EPSILON {
                self.preferences.motion_threshold = default_motion_threshold();
            }
            if (self.preferences.person_confidence - 0.50).abs() < f32::EPSILON {
                self.preferences.person_confidence = default_person_confidence();
            }
            self.version = 4;
            changed = true;
        }

        if self.version < 5 {
            if self.preferences.analysis_height == 540 {
                self.preferences.analysis_height = default_analysis_height();
            }
            if (self.preferences.analysis_fps - 6.0).abs() < f32::EPSILON {
                self.preferences.analysis_fps = default_analysis_fps();
            }
            if (self.preferences.window_seconds - 1.25).abs() < f32::EPSILON {
                self.preferences.window_seconds = default_window_seconds();
            }
            self.version = 5;
            changed = true;
        }

        if self.version < 6 {
            self.preferences.enable_yolo = default_enable_yolo();
            self.version = 6;
            changed = true;
        }

        if self.version < 7 {
            if (self.preferences.motion_threshold - 3.2).abs() < f32::EPSILON {
                self.preferences.motion_threshold = default_motion_threshold();
            }
            self.version = 7;
            changed = true;
        }

        if self.version < 8 {
            if self.preferences.analysis_height == 720 {
                self.preferences.analysis_height = default_analysis_height();
            }
            self.version = 8;
            changed = true;
        }

        if self.version < 9 {
            // Move the untouched legacy speed defaults to the accuracy
            // profile while preserving genuinely custom preferences.
            if self.preferences.analysis_height == 360
                && (self.preferences.analysis_fps - 12.0).abs() < f32::EPSILON
            {
                self.preferences.analysis_height = default_analysis_height();
                self.preferences.analysis_fps = default_analysis_fps();
            }
            if (self.preferences.motion_threshold - 1.8).abs() < f32::EPSILON {
                self.preferences.motion_threshold = default_motion_threshold();
            }
            self.version = 9;
            changed = true;
        }

        if self.version < 10 {
            if self
                .preferences
                .extensions
                .trim()
                .eq_ignore_ascii_case("mov,mp4,mxf")
            {
                self.preferences.extensions = default_extensions();
            }
            self.version = 10;
            changed = true;
        }

        if self.version < 11 {
            // New fields are serde-defaulted. Keep audio opt-in and move all
            // legacy users to the editor-friendly eight-second select cap.
            if !self.preferences.max_select_seconds.is_finite()
                || self.preferences.max_select_seconds <= 0.0
            {
                self.preferences.max_select_seconds = default_max_select_seconds();
            }
            self.version = 11;
            changed = true;
        }

        changed
    }

    /// Load settings from disk.  Returns `None` if the file does not exist.
    /// Returns an error only on genuine I/O / parse failures.
    pub fn load() -> AppResult<Option<Self>> {
        let path = match settings_path() {
            Some(p) => p,
            None => {
                warn!("Could not determine config directory; skipping settings load");
                return Ok(None);
            }
        };

        if !path.exists() {
            debug!("No settings file at {}", path.display());
            return Ok(None);
        }

        let data = std::fs::read_to_string(&path).map_err(|e| AppError::Io {
            path: path.clone(),
            source: e,
        })?;

        let mut settings: PersistedSettings =
            serde_json::from_str(&data).map_err(|e| AppError::ParseFailed {
                what: "settings.json",
                source: Box::new(e),
            })?;

        // Schema migration: if the file is from a newer version, reset to
        // defaults rather than silently misinterpreting fields.
        if settings.version > SCHEMA_VERSION {
            warn!(
                "Settings file version {} is newer than supported {}; using defaults",
                settings.version, SCHEMA_VERSION
            );
            return Ok(Some(Self::default()));
        }

        let migrated = settings.migrate();

        // Validate that resolved paths still exist on disk.
        let repaired_stale_paths = settings.resolved_paths.validate();
        let repaired_stale_export = settings
            .last_export
            .as_ref()
            .is_some_and(|summary| !Path::new(&summary.output_path).is_file());
        if repaired_stale_export {
            settings.last_export = None;
        }

        if (repaired_stale_paths || repaired_stale_export || migrated)
            && let Err(e) = settings.save()
        {
            warn!("Failed to persist cleaned settings after path validation: {e}");
        }

        info!("Loaded settings from {}", path.display());
        Ok(Some(settings))
    }

    /// Persist current settings to disk.  Creates the config directory if needed.
    pub fn save(&self) -> AppResult<()> {
        let path = match settings_path() {
            Some(p) => p,
            None => {
                warn!("Could not determine config directory; skipping settings save");
                return Ok(());
            }
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AppError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        let json = serde_json::to_string_pretty(self).map_err(|e| AppError::ParseFailed {
            what: "settings.json serialization",
            source: Box::new(e),
        })?;

        atomic_file::write_bytes(&path, json.as_bytes())?;

        debug!("Saved settings to {}", path.display());
        Ok(())
    }

    /// Update the resolved tool paths after a successful asset resolution.
    pub fn set_resolved_paths(&mut self, ffmpeg: &Path, ffprobe: &Path, yolo: Option<&Path>) {
        self.resolved_paths.ffmpeg = Some(ffmpeg.to_string_lossy().into_owned());
        self.resolved_paths.ffprobe = Some(ffprobe.to_string_lossy().into_owned());
        self.resolved_paths.yolo_model = yolo.map(|p| p.to_string_lossy().into_owned());
    }

    /// Returns `true` if we have cached tool paths and they all still exist.
    #[allow(dead_code)]
    pub fn has_valid_tool_paths(&self) -> bool {
        self.resolved_paths.is_valid()
    }

    /// Retrieve the cached ffmpeg path (only if the file still exists on disk).
    pub fn cached_ffmpeg(&self) -> Option<PathBuf> {
        self.resolved_paths
            .ffmpeg
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists() || p.components().count() <= 1)
    }

    /// Retrieve the cached ffprobe path (only if the file still exists on disk).
    pub fn cached_ffprobe(&self) -> Option<PathBuf> {
        self.resolved_paths
            .ffprobe
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists() || p.components().count() <= 1)
    }

    /// Retrieve the cached YOLO model path (only if it still exists on disk).
    pub fn cached_yolo(&self) -> Option<PathBuf> {
        self.resolved_paths
            .yolo_model
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists())
    }
}

impl ResolvedPaths {
    /// Check whether the persisted paths still point to real files.
    /// Clears any path that no longer exists (bare names like "ffmpeg" are
    /// kept since they rely on PATH lookup at runtime).
    fn validate(&mut self) -> bool {
        let mut changed = false;

        if let Some(ref p) = self.ffmpeg {
            let pb = PathBuf::from(p);
            if pb.components().count() > 1 && !pb.exists() {
                warn!("Cached ffmpeg path no longer exists: {p}");
                self.ffmpeg = None;
                changed = true;
            }
        }
        if let Some(ref p) = self.ffprobe {
            let pb = PathBuf::from(p);
            if pb.components().count() > 1 && !pb.exists() {
                warn!("Cached ffprobe path no longer exists: {p}");
                self.ffprobe = None;
                changed = true;
            }
        }
        if let Some(ref p) = self.yolo_model {
            let pb = PathBuf::from(p);
            if !pb.exists() {
                warn!("Cached YOLO model path no longer exists: {p}");
                self.yolo_model = None;
                changed = true;
            }
        }

        changed
    }

    /// Returns true if at least ffmpeg and ffprobe are resolved and present.
    #[allow(dead_code)]
    fn is_valid(&self) -> bool {
        let ff_ok = self
            .ffmpeg
            .as_ref()
            .map(|p| {
                let pb = PathBuf::from(p);
                pb.exists() || pb.components().count() <= 1
            })
            .unwrap_or(false);

        let fp_ok = self
            .ffprobe
            .as_ref()
            .map(|p| {
                let pb = PathBuf::from(p);
                pb.exists() || pb.components().count() <= 1
            })
            .unwrap_or(false);

        ff_ok && fp_ok
    }
}

#[cfg(test)]
mod tests {
    use super::{PersistedSettings, ResolvedPaths, SCHEMA_VERSION, UserPreferences};

    fn definitely_missing_path(file_name: &str) -> String {
        #[cfg(target_os = "windows")]
        {
            format!(r"C:\definitely-missing\{file_name}")
        }

        #[cfg(not(target_os = "windows"))]
        {
            std::path::PathBuf::from("/definitely-missing")
                .join(file_name)
                .to_string_lossy()
                .into_owned()
        }
    }

    #[test]
    fn validate_clears_missing_paths_and_reports_change() {
        let mut paths = ResolvedPaths {
            ffmpeg: Some(definitely_missing_path("ffmpeg.exe")),
            ffprobe: Some(definitely_missing_path("ffprobe.exe")),
            yolo_model: Some(definitely_missing_path("yolo.onnx")),
        };

        let changed = paths.validate();

        assert!(changed);
        assert!(paths.ffmpeg.is_none());
        assert!(paths.ffprobe.is_none());
        assert!(paths.yolo_model.is_none());
    }

    #[test]
    fn validate_keeps_bare_binary_names_without_reporting_change() {
        let mut paths = ResolvedPaths {
            ffmpeg: Some("ffmpeg".to_string()),
            ffprobe: Some("ffprobe".to_string()),
            yolo_model: None,
        };

        let changed = paths.validate();

        assert!(!changed);
        assert_eq!(paths.ffmpeg.as_deref(), Some("ffmpeg"));
        assert_eq!(paths.ffprobe.as_deref(), Some("ffprobe"));
    }

    #[test]
    fn migrate_v1_motion_threshold_updates_old_default() {
        let mut settings = PersistedSettings {
            version: 1,
            resolved_paths: ResolvedPaths::default(),
            last_export: None,
            preferences: UserPreferences {
                motion_threshold: 12.0,
                ..UserPreferences::default()
            },
        };

        let changed = settings.migrate();

        assert!(changed);
        assert_eq!(settings.version, SCHEMA_VERSION);
        assert!((settings.preferences.motion_threshold - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn migrate_v2_analysis_fps_updates_old_default() {
        let mut settings = PersistedSettings {
            version: 2,
            resolved_paths: ResolvedPaths::default(),
            last_export: None,
            preferences: UserPreferences {
                analysis_fps: 4.0,
                ..UserPreferences::default()
            },
        };

        let changed = settings.migrate();

        assert!(changed);
        assert_eq!(settings.version, SCHEMA_VERSION);
        assert_eq!(settings.preferences.analysis_fps, 18.0);
    }

    #[test]
    fn migrate_v3_accuracy_defaults_update_old_generic_values() {
        let mut settings = PersistedSettings {
            version: 3,
            resolved_paths: ResolvedPaths::default(),
            last_export: None,
            preferences: UserPreferences {
                analysis_height: 480,
                window_seconds: 1.0,
                motion_threshold: 4.0,
                person_confidence: 0.50,
                ..UserPreferences::default()
            },
        };

        let changed = settings.migrate();

        assert!(changed);
        assert_eq!(settings.version, SCHEMA_VERSION);
        assert_eq!(settings.preferences.analysis_height, 720);
        assert!((settings.preferences.window_seconds - 1.0).abs() < f32::EPSILON);
        assert!((settings.preferences.motion_threshold - 0.0).abs() < f32::EPSILON);
        assert!((settings.preferences.person_confidence - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn migrate_v9_expands_legacy_video_extensions() {
        let mut settings = PersistedSettings {
            version: 9,
            resolved_paths: ResolvedPaths::default(),
            last_export: None,
            preferences: UserPreferences {
                extensions: "mov,mp4,mxf".to_string(),
                ..UserPreferences::default()
            },
        };

        let changed = settings.migrate();

        assert!(changed);
        assert_eq!(settings.version, SCHEMA_VERSION);
        assert!(settings.preferences.extensions.contains("mkv"));
        assert!(settings.preferences.extensions.contains("m2ts"));
    }
}
