//! Per-file JSON sidecar cache for analyser results.
//!
//! Each successfully analysed source file writes one `<key>.json` entry
//! under `<output>/.cache/` atomically (write → fsync → rename) so a crash
//! or Ctrl+C mid-run never corrupts previously-finished work.  The next
//! invocation of `run_analyze` loads matching entries instead of re-running
//! ffmpeg/YOLO, giving transparent resume.
//!
//! A sidecar-per-file layout was chosen over a single SQLite DB / append
//! log because it has zero new dependencies, no lock contention between
//! rayon workers, trivial selective invalidation, and remains
//! human-readable enough for debugging.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::config::AnalysisConfig;
use crate::error::{AppError, AppResult};
use crate::media::ProbeInfo;
use crate::timeline::Segment;

/// Bumped whenever the on-disk layout changes incompatibly.
const CACHE_SCHEMA_VERSION: u32 = 10;

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    pub schema_version: u32,
    /// Stable hash of every AnalysisConfig field that influences segment
    /// output.  Mismatch → silent miss (forces a re-run).
    pub config_fingerprint: String,
    pub source_path: PathBuf,
    pub source_size: u64,
    pub source_mtime_nanos: u128,
    pub probe: ProbeInfo,
    pub segments: Vec<Segment>,
}

pub fn cache_dir(output_dir: &Path) -> PathBuf {
    output_dir.join(".cache")
}

pub fn ensure_cache_dir(output_dir: &Path) -> AppResult<PathBuf> {
    let dir = cache_dir(output_dir);
    fs::create_dir_all(&dir).map_err(|e| AppError::Io {
        path: dir.clone(),
        source: e,
    })?;
    Ok(dir)
}

/// Hash every knob that affects segment output so changing e.g. the motion
/// threshold silently invalidates the cache without needing a manual purge.
pub fn config_fingerprint(config: &AnalysisConfig) -> String {
    let model_identity = config
        .yolo_model
        .as_ref()
        .map(|p| (p.to_string_lossy().into_owned(), yolo_model_identity(p)));
    let payload = serde_json::json!({
        "schema": CACHE_SCHEMA_VERSION,
        "analysis_height": config.analysis_height,
        "analysis_fps_bits": config.analysis_fps.to_bits(),
        "window_seconds_bits": config.window_seconds.to_bits(),
        "motion_threshold_bits": config.motion_threshold.to_bits(),
        "person_confidence_bits": config.person_confidence.to_bits(),
        "enable_yolo": config.enable_yolo,
        "yolo_model": model_identity,
    });
    let encoded = serde_json::to_vec(&payload).expect("config fingerprint payload serializes");
    let digest = Sha256::digest(encoded);
    format!("v{}-{:x}", CACHE_SCHEMA_VERSION, digest)
}

fn yolo_model_identity(path: &Path) -> Option<(u64, u128)> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((metadata.len(), modified))
}

fn key_for(path: &Path) -> String {
    let digest = Sha256::digest(path.to_string_lossy().as_bytes());
    let digest_hex = format!("{:x}", digest);
    let base = path.file_name().and_then(|s| s.to_str()).unwrap_or("clip");
    // Keep the readable filename prefix short and filesystem-safe; the
    // 64-bit hash suffix disambiguates files with identical basenames.
    let safe: String = base
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(*c, '.' | '_' | '-'))
        .take(48)
        .collect();
    format!("{safe}-{}", &digest_hex[..16])
}

fn entry_path(cache_dir: &Path, source: &Path) -> PathBuf {
    cache_dir.join(format!("{}.json", key_for(source)))
}

fn file_stat(path: &Path) -> AppResult<(u64, u128)> {
    let md = fs::metadata(path).map_err(|e| AppError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mtime = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let nanos = mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok((md.len(), nanos))
}

/// Return the cached (probe, segments) pair if a valid entry exists.
///
/// A non-matching fingerprint, stale mtime/size, corrupt JSON, or older
/// schema version all produce `Ok(None)` rather than an error — a bad
/// cache entry should never fail the whole run.
pub fn load(
    cache_dir: &Path,
    path: &Path,
    config: &AnalysisConfig,
) -> AppResult<Option<(ProbeInfo, Vec<Segment>)>> {
    let cache_path = entry_path(cache_dir, path);
    if !cache_path.exists() {
        return Ok(None);
    }
    let data = match fs::read(&cache_path) {
        Ok(b) => b,
        Err(e) => {
            warn!(
                "Ignoring unreadable cache entry {}: {e}",
                cache_path.display()
            );
            return Ok(None);
        }
    };
    let entry: CacheEntry = match serde_json::from_slice(&data) {
        Ok(e) => e,
        Err(e) => {
            warn!("Ignoring corrupt cache entry {}: {e}", cache_path.display());
            return Ok(None);
        }
    };
    if entry.schema_version != CACHE_SCHEMA_VERSION
        || entry.source_path != path
        || entry.config_fingerprint != config.config_fingerprint
    {
        return Ok(None);
    }
    let (size, nanos) = file_stat(path)?;
    if entry.source_size != size || entry.source_mtime_nanos != nanos {
        return Ok(None);
    }
    debug!("cache hit: {}", path.display());
    Ok(Some((entry.probe, entry.segments)))
}

/// Write a cache entry atomically and durably.
///
/// Serialised JSON is written to `<key>.json.tmp`, `sync_all()`-ed, then
/// renamed over the final path.  On every mainstream filesystem the
/// rename is atomic, so readers either see the previous entry or the
/// fully-written new one, never a partial file.
pub fn store(
    cache_dir: &Path,
    config: &AnalysisConfig,
    probe: &ProbeInfo,
    segments: &[Segment],
) -> AppResult<()> {
    let path = &probe.source_path;
    let (size, nanos) = file_stat(path)?;
    let entry = CacheEntry {
        schema_version: CACHE_SCHEMA_VERSION,
        config_fingerprint: config.config_fingerprint.clone(),
        source_path: path.clone(),
        source_size: size,
        source_mtime_nanos: nanos,
        probe: probe.clone(),
        segments: segments.to_vec(),
    };
    let json = serde_json::to_vec(&entry).map_err(|e| AppError::ParseFailed {
        what: "serialise cache entry",
        source: Box::new(e),
    })?;

    let final_path = entry_path(cache_dir, path);
    let tmp_path = cache_dir.join(format!("{}.json.tmp", key_for(path)));

    {
        let mut f = fs::File::create(&tmp_path).map_err(|e| AppError::Io {
            path: tmp_path.clone(),
            source: e,
        })?;
        f.write_all(&json).map_err(|e| AppError::Io {
            path: tmp_path.clone(),
            source: e,
        })?;
        // fsync so the contents survive a power loss between the write and
        // the rename.  Without this, a crash can leave a zero-length file
        // that would be picked up on restart.
        f.sync_all().map_err(|e| AppError::Io {
            path: tmp_path.clone(),
            source: e,
        })?;
    }

    if final_path.exists()
        && let Err(e) = fs::remove_file(&final_path)
    {
        warn!(
            "cache: failed to remove stale entry {} before replace: {e}",
            final_path.display()
        );
    }

    fs::rename(&tmp_path, &final_path).map_err(|e| AppError::Io {
        path: final_path.clone(),
        source: e,
    })?;
    debug!("cache store: {}", final_path.display());
    Ok(())
}

/// Load every valid cache entry under `cache_dir`, regardless of whether
/// the corresponding source file is still on the user's filesystem.
///
/// Used by the "export from cache" recovery path so the XML can be
/// regenerated purely from sidecar files after a crash, without rescanning
/// the input directory or re-running ffmpeg.
pub fn load_all(cache_dir: &Path) -> AppResult<Vec<(ProbeInfo, Vec<Segment>)>> {
    if !cache_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let iter = fs::read_dir(cache_dir).map_err(|e| AppError::Io {
        path: cache_dir.to_path_buf(),
        source: e,
    })?;
    for entry in iter {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("cache read_dir: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let data = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                warn!("cache: skip unreadable {}: {e}", path.display());
                continue;
            }
        };
        let parsed: CacheEntry = match serde_json::from_slice(&data) {
            Ok(e) => e,
            Err(e) => {
                warn!("cache: skip corrupt {}: {e}", path.display());
                continue;
            }
        };
        if parsed.schema_version != CACHE_SCHEMA_VERSION {
            continue;
        }
        out.push((parsed.probe, parsed.segments));
    }
    // Deterministic ordering so the exported timeline is stable between runs.
    out.sort_by(|a, b| a.0.source_path.cmp(&b.0.source_path));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::{Segment, SegmentKind};

    fn tmp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join("video-tool-cache-test")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn make_config(motion_threshold: f32) -> AnalysisConfig {
        let mut cfg = AnalysisConfig {
            ffmpeg_bin: PathBuf::from("ffmpeg"),
            ffprobe_bin: PathBuf::from("ffprobe"),
            yolo_model: None,
            enable_yolo: true,
            config_fingerprint: String::new(),
            analysis_height: 480,
            analysis_fps: 4.0,
            window_seconds: 1.0,
            motion_threshold,
            person_confidence: 0.50,
            yolo_intra_threads: 1,
            ffmpeg_threads: 1,
            buf_frames: 4,
        };
        cfg.config_fingerprint = config_fingerprint(&cfg);
        cfg
    }

    fn sample_probe(source: PathBuf) -> ProbeInfo {
        ProbeInfo {
            source_path: source,
            width: 1920,
            height: 1080,
            duration_seconds: 4.0,
            duration_frames: 100,
            fps_num: 25,
            fps_den: 1,
            timebase: 25,
            ntsc: false,
            slow_motion: false,
            capture_fps: None,
            format_fps: None,
        }
    }

    fn sample_segment(source: &Path) -> Segment {
        Segment {
            source_path: source.to_path_buf(),
            start_frame: 0,
            end_frame: 25,
            start_seconds: 0.0,
            end_seconds: 1.0,
            kind: SegmentKind::StaticSubject,
            label_id: SegmentKind::StaticSubject.label_id(),
            motion_score: 1.23,
            zoom_score: 0.0,
            person_confidence: Some(0.91),
            window_count: 1,
        }
    }

    #[test]
    fn store_then_load_round_trips_segments() {
        let root = tmp_root("roundtrip");
        let src = root.join("clip.mov");
        fs::write(&src, b"fake-bytes").unwrap();

        let cfg = make_config(3.0);
        let probe = sample_probe(src.clone());
        let segs = vec![sample_segment(&src)];

        store(&root, &cfg, &probe, &segs).unwrap();
        let loaded = load(&root, &src, &cfg).unwrap().expect("cache hit");
        assert_eq!(loaded.1.len(), 1);
        assert_eq!(loaded.1[0].motion_score, 1.23);
        assert_eq!(loaded.1[0].person_confidence, Some(0.91));
        assert_eq!(loaded.1[0].window_count, 1);
        assert_eq!(loaded.0.width, 1920);
    }

    #[test]
    fn config_change_invalidates_cache() {
        let root = tmp_root("invalidate");
        let src = root.join("clip.mov");
        fs::write(&src, b"fake-bytes").unwrap();

        let cfg_a = make_config(3.0);
        let cfg_b = make_config(5.0); // different motion_threshold → different fingerprint
        let probe = sample_probe(src.clone());
        let segs = vec![sample_segment(&src)];

        store(&root, &cfg_a, &probe, &segs).unwrap();
        assert!(load(&root, &src, &cfg_a).unwrap().is_some());
        assert!(
            load(&root, &src, &cfg_b).unwrap().is_none(),
            "cache must miss when the analysis config fingerprint differs"
        );
    }

    #[test]
    fn yolo_model_change_invalidates_fingerprint() {
        let root = tmp_root("model-fingerprint");
        let model = root.join("yolo.onnx");
        fs::write(&model, b"model-a").unwrap();

        let mut cfg = make_config(3.0);
        cfg.yolo_model = Some(model.clone());
        let fp_a = config_fingerprint(&cfg);

        fs::write(&model, b"model-b-with-different-size").unwrap();
        let fp_b = config_fingerprint(&cfg);

        assert_ne!(
            fp_a, fp_b,
            "cache fingerprint must change when the YOLO model at the same path changes"
        );
    }

    #[test]
    fn load_all_returns_every_valid_entry_sorted() {
        let root = tmp_root("loadall");
        let a = root.join("a.mov");
        let b = root.join("b.mov");
        fs::write(&a, b"aa").unwrap();
        fs::write(&b, b"bb").unwrap();

        let cfg = make_config(3.0);
        store(&root, &cfg, &sample_probe(a.clone()), &[sample_segment(&a)]).unwrap();
        store(&root, &cfg, &sample_probe(b.clone()), &[sample_segment(&b)]).unwrap();

        let all = load_all(&root).unwrap();
        assert_eq!(all.len(), 2);
        // Sorted by source_path: a.mov first, b.mov second.
        assert!(all[0].0.source_path.ends_with("a.mov"));
        assert!(all[1].0.source_path.ends_with("b.mov"));
    }
}
