use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use tracing::{debug, error, info, warn};

use crate::analyzer;
use crate::cache;
use crate::config::{AnalysisConfig, AssetConfig};
use crate::error::{AppError, AppResult};
use crate::media::{self, ProbeInfo};
use crate::settings::PersistedSettings;
use crate::timeline::{self, Segment};
use crate::xml_exporter;

/// Plain struct — no CLI framework required.
/// The GUI populates this directly; advanced users can construct it in code.
#[derive(Debug, Clone)]
pub struct AnalyzeArgs {
    /// Input directory to scan recursively.
    pub input: PathBuf,
    /// Output directory for the generated XML.
    pub output: PathBuf,
    /// Override embedded YOLO model path.
    pub yolo_model: Option<PathBuf>,
    /// Enable YOLO person detection for static-subject selects.
    pub enable_yolo: bool,
    /// Override embedded ffmpeg path.
    pub ffmpeg_bin: Option<PathBuf>,
    /// Override embedded ffprobe path.
    pub ffprobe_bin: Option<PathBuf>,
    /// Height in pixels to downscale frames for analysis.
    pub analysis_height: u32,
    /// Analysis FPS to sample frames at (via ffmpeg fps filter).
    pub analysis_fps: f32,
    /// Window size in seconds. The analyser evaluates overlapping windows
    /// with a half-window stride for better temporal recall.
    pub window_seconds: f32,
    /// Motion threshold: global affine camera motion strength derived from
    /// translation, zoom, rotation, and shear evidence.
    pub motion_threshold: f32,
    /// Minimum person confidence to mark a static subject segment.
    pub person_confidence: f32,
    /// Maximum number of parallel worker threads.
    pub max_files: Option<usize>,
    /// ORT intra-op threads per YOLO session.
    pub yolo_intra_threads: Option<usize>,
    /// Number of threads ffmpeg uses for decoding.
    pub ffmpeg_threads: Option<usize>,
    /// BufReader capacity expressed as a multiple of the raw frame size.
    pub buf_frames: Option<usize>,
    /// Comma-separated list of extensions (e.g. mov,mp4,mxf).
    pub extensions: String,
    /// Enable verbose/debug logging.
    pub verbose: bool,
    /// Cooperative cancellation flag shared with the GUI.
    pub cancel_flag: Arc<AtomicBool>,
    /// Loaded persisted settings (if available) for fast-path tool resolution.
    #[allow(dead_code)]
    pub persisted_settings: Option<PersistedSettings>,
}

impl Default for AnalyzeArgs {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            output: PathBuf::new(),
            yolo_model: None,
            enable_yolo: true,
            ffmpeg_bin: None,
            ffprobe_bin: None,
            analysis_height: 360,
            analysis_fps: 12.0,
            window_seconds: 1.0,
            motion_threshold: 1.8,
            person_confidence: 0.42,
            max_files: None,
            yolo_intra_threads: None,
            ffmpeg_threads: None,
            buf_frames: None,
            extensions: "mov,mp4,mxf".to_string(),
            verbose: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            persisted_settings: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunSummary {
    pub files_scanned: usize,
    pub files_analyzed: usize,
    /// Files whose segments were loaded from the on-disk cache instead of
    /// being re-analysed from scratch.  Counted separately so the GUI can
    /// show "N new, M resumed".
    pub cached_files: usize,
    pub exported_segments: usize,
    pub failed_files: usize,
    /// Path of the single merged XML that was written.
    pub output_path: Option<PathBuf>,
}

type AnalyzeResult = AppResult<(ProbeInfo, Vec<Segment>)>;
type WorkerResult = (AnalyzeResult, bool);

/// Real-time progress messages sent from the engine to the GUI.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ProgressMsg {
    /// A short human-readable label describing the current setup phase
    /// (e.g. "Validating tools…", "Scanning input folder…").
    Preparing { phase: String },
    /// Current number of candidate files discovered on disk.
    Discovered { total: usize },
    /// Discovery has finished; no more files will be queued.
    DiscoveryFinished { total: usize },
    /// A worker thread has started processing this file.
    FileStarted { index: usize, path: PathBuf },
    /// A worker thread has finished processing this file (success or fail).
    FileFinished {
        index: usize,
        path: PathBuf,
        ok: bool,
        segments: usize,
        /// True when the result was served from the sidecar cache and no
        /// ffmpeg/YOLO work was performed.
        from_cache: bool,
    },
}

pub fn init_tracing(verbose: bool) {
    let filter = if verbose {
        "info,video_tool=debug"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();
}

pub fn run_analyze(
    mut args: AnalyzeArgs,
    progress_tx: Option<mpsc::Sender<ProgressMsg>>,
) -> AppResult<RunSummary> {
    if args.verbose {
        debug!("Verbose analysis logging requested");
    }
    validate_input_dir(&args.input)?;
    std::fs::create_dir_all(&args.output).map_err(|e| AppError::Io {
        path: args.output.clone(),
        source: e,
    })?;
    if args.cancel_flag.load(Ordering::Relaxed) {
        return Err(AppError::Cancelled);
    }

    let extensions = parse_extensions(&args.extensions);
    if extensions.is_empty() {
        return Err(AppError::Unsupported("no extensions specified".to_string()));
    }
    if args.max_files == Some(0) {
        return Err(AppError::Unsupported(
            "max_files must be greater than 0".to_string(),
        ));
    }

    let send = |msg: ProgressMsg| {
        if let Some(ref tx) = progress_tx {
            let _ = tx.send(msg);
        }
    };

    // Take persisted_settings out of args to avoid borrow conflicts.
    let mut persisted = args.persisted_settings.take();

    let asset_cfg = AssetConfig {
        ffmpeg_override: args.ffmpeg_bin.clone(),
        ffprobe_override: args.ffprobe_bin.clone(),
        yolo_override: args.yolo_model.clone(),
    };
    let config = AnalysisConfig::from_args(&args, asset_cfg, persisted.as_mut())?;

    send(ProgressMsg::Preparing {
        phase: "Scanning input folder…".to_string(),
    });

    // Keep the file worker count in lock-step with the balanced budget
    // computed in AnalysisConfig: ⌈cpus/2⌉ workers, each of which spawns its
    // own ffmpeg/YOLO threads.  Running more workers than the config budget
    // assumes re-introduces the oversubscription the budget was designed to
    // avoid.
    let threads = args.max_files.unwrap_or_else(default_worker_count);
    debug!("Using up to {threads} worker threads");

    // Persistent sidecar cache — every successful analyse_one_data writes
    // a JSON entry under <output>/.cache/ atomically, so a crash or Ctrl+C
    // mid-run never loses finished work.  On restart each hit is loaded
    // instead of re-processed.
    let cache_dir = cache::ensure_cache_dir(&args.output)?;
    let work_config = Arc::new(config.clone());
    let work_cache_dir = cache_dir.clone();
    let cancel_flag = Arc::clone(&args.cancel_flag);
    let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
    let (idle_tx, idle_rx) = mpsc::sync_channel::<usize>(threads.max(1));
    let mut worker_handles = Vec::with_capacity(threads);
    let mut worker_txs = Vec::with_capacity(threads);

    for worker_id in 0..threads {
        let (tx_work, rx_work) = mpsc::channel::<Option<(usize, PathBuf)>>();
        worker_txs.push(tx_work);
        let tx = result_tx.clone();
        let idle = idle_tx.clone();
        let progress = progress_tx.clone();
        let config = Arc::clone(&work_config);
        let cache_dir = work_cache_dir.clone();
        let cancel = Arc::clone(&cancel_flag);
        worker_handles.push(std::thread::spawn(move || {
            let mut worker = analyzer::AnalyzerWorker::default();
            if idle.send(worker_id).is_err() {
                return;
            }
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let message = rx_work.recv();
                let Ok(Some((idx, path))) = message else {
                    break;
                };
                if cancel.load(Ordering::Relaxed) {
                    return;
                }

                if let Some(ref tx) = progress {
                    let _ = tx.send(ProgressMsg::FileStarted {
                        index: idx,
                        path: path.clone(),
                    });
                }

                let cached = match cache::load(&cache_dir, &path, &config) {
                    Ok(entry) => entry,
                    Err(e) => {
                        warn!("cache lookup failed for {}: {e}", path.display());
                        None
                    }
                };

                let (result, from_cache) = if let Some((probe, segments)) = cached {
                    info!(
                        "{}: loaded {} cached segment(s) from sidecar",
                        path.display(),
                        segments.len()
                    );
                    (Ok((probe, segments)), true)
                } else {
                    let r = analyze_one_data(&path, &config, &cancel, &mut worker);
                    if let Ok((probe, segments)) = &r
                        && let Err(e) = cache::store(&cache_dir, &config, probe, segments)
                    {
                        warn!("cache store failed for {}: {e}", path.display());
                    }
                    (r, false)
                };

                if let Some(ref tx) = progress {
                    let (ok, segments) = match &result {
                        Ok((_, segs)) => (true, segs.len()),
                        Err(_) => (false, 0),
                    };
                    let _ = tx.send(ProgressMsg::FileFinished {
                        index: idx,
                        path: path.clone(),
                        ok,
                        segments,
                        from_cache,
                    });
                }

                let _ = tx.send((result, from_cache));

                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                if idle.send(worker_id).is_err() {
                    return;
                }
            }
        }));
    }
    drop(result_tx);
    drop(idle_tx);

    let tx_clone = progress_tx.clone();
    let mut discovered = 0usize;
    let mut pending_work = VecDeque::new();
    let discovery_result = media::discover_inputs_streaming(
        &args.input,
        &cache_dir,
        &extensions,
        &mut move |progress| {
            if let Some(ref tx) = tx_clone {
                let _ = tx.send(ProgressMsg::Preparing {
                    phase: format!(
                        "Scanning input folder… (scanned {}, found {})",
                        progress.entries_scanned, progress.matches_found
                    ),
                });
            }
        },
        |path| {
            if cancel_flag.load(Ordering::Relaxed) {
                return Err(AppError::Cancelled);
            }
            let idx = discovered;
            discovered += 1;
            send(ProgressMsg::Discovered { total: discovered });
            pending_work.push_back((idx, path));
            dispatch_pending_work(
                &idle_rx,
                &worker_txs,
                &mut pending_work,
                &cancel_flag,
                false,
            )
        },
    );
    let drain_result = if discovery_result.is_ok() {
        dispatch_pending_work(&idle_rx, &worker_txs, &mut pending_work, &cancel_flag, true)
    } else {
        Ok(())
    };
    for tx in &worker_txs {
        let _ = tx.send(None);
    }
    discovery_result?;
    drain_result?;
    send(ProgressMsg::DiscoveryFinished { total: discovered });

    if discovered == 0 {
        warn!("No input files found under {}", args.input.display());
        for handle in worker_handles {
            let _ = handle.join();
        }
        return Ok(RunSummary::default());
    }
    info!("Found {} candidate files", discovered);

    let mut raw_results: Vec<WorkerResult> = Vec::with_capacity(discovered);
    while raw_results.len() < discovered {
        match result_rx.recv() {
            Ok(result) => raw_results.push(result),
            Err(e) => {
                if args.cancel_flag.load(Ordering::Relaxed) {
                    break;
                }
                return Err(AppError::Message(format!(
                    "analysis worker disconnected unexpectedly: {e}"
                )));
            }
        }
    }
    let mut worker_panics = Vec::new();
    for handle in worker_handles {
        if let Err(payload) = handle.join() {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            worker_panics.push(msg);
        }
    }
    if args.cancel_flag.load(Ordering::Relaxed) {
        return Err(AppError::Cancelled);
    }
    if !worker_panics.is_empty() {
        return Err(AppError::Message(format!(
            "analysis worker panicked: {}",
            worker_panics.join(" | ")
        )));
    }
    if raw_results.len() != discovered {
        return Err(AppError::Message(format!(
            "analysis stopped early: received {} of {} worker results",
            raw_results.len(),
            discovered
        )));
    }

    let mut failed = 0usize;
    let mut cached_files = 0usize;
    let mut all_data: Vec<(ProbeInfo, Vec<Segment>)> = Vec::with_capacity(discovered);
    for (r, from_cache) in raw_results {
        match r {
            Ok(data) => {
                if from_cache {
                    cached_files += 1;
                }
                all_data.push(data);
            }
            Err(err) => {
                failed += 1;
                error!("{err}");
            }
        }
    }

    let total_segments: usize = all_data.iter().map(|(_, segs)| segs.len()).sum();

    // Write one merged XML for all clips.  `all_data` now aggregates both
    // freshly-analysed results and entries rehydrated from the sidecar
    // cache — the XML exporter does not need to know the difference.
    let out_path = xml_exporter::export_all(&all_data, &args.output)?;
    info!(
        "Exported {total_segments} segments across {} files ({} from cache) → {}",
        all_data.len(),
        cached_files,
        out_path.display()
    );

    let summary = RunSummary {
        files_scanned: discovered,
        files_analyzed: all_data.len(),
        cached_files,
        exported_segments: total_segments,
        failed_files: failed,
        output_path: Some(out_path),
    };

    Ok(summary)
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().div_ceil(2).clamp(1, 8))
        .unwrap_or(4)
}

fn dispatch_pending_work(
    idle_rx: &mpsc::Receiver<usize>,
    worker_txs: &[mpsc::Sender<Option<(usize, PathBuf)>>],
    pending_work: &mut VecDeque<(usize, PathBuf)>,
    cancel_flag: &Arc<AtomicBool>,
    wait_for_worker: bool,
) -> AppResult<()> {
    while !pending_work.is_empty() {
        if cancel_flag.load(Ordering::Relaxed) {
            return Err(AppError::Cancelled);
        }

        let worker_id = if wait_for_worker {
            match idle_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(worker_id) => worker_id,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(AppError::Message(
                        "analysis worker scheduler disconnected unexpectedly".to_string(),
                    ));
                }
            }
        } else {
            match idle_rx.try_recv() {
                Ok(worker_id) => worker_id,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(AppError::Message(
                        "analysis worker scheduler disconnected unexpectedly".to_string(),
                    ));
                }
            }
        };

        let Some(job) = pending_work.pop_front() else {
            break;
        };
        worker_txs[worker_id].send(Some(job)).map_err(|e| {
            AppError::Message(format!("failed to queue discovered file for analysis: {e}"))
        })?;
    }

    Ok(())
}

/// Rebuild the merged XML purely from the sidecar cache under
/// `<output>/.cache/`, without rescanning the input directory or running
/// ffmpeg.  Intended as a recovery entry point after a crash: as long as
/// at least one file completed analysis before the crash, its segments
/// are preserved on disk and can be exported.
///
/// Returns the summary of what was written, or an empty summary with no
/// output path if no cache entries were found.
#[allow(dead_code)]
pub fn export_from_cache(output: &Path) -> AppResult<RunSummary> {
    std::fs::create_dir_all(output).map_err(|e| AppError::Io {
        path: output.to_path_buf(),
        source: e,
    })?;
    let cache_dir = cache::cache_dir(output);
    let all_data = cache::load_all(&cache_dir)?;
    if all_data.is_empty() {
        warn!(
            "No cache entries found under {} — nothing to export.",
            cache_dir.display()
        );
        return Ok(RunSummary::default());
    }
    let total_segments: usize = all_data.iter().map(|(_, segs)| segs.len()).sum();
    let out_path = xml_exporter::export_all(&all_data, output)?;
    info!(
        "Exported {total_segments} segments from cache across {} files → {}",
        all_data.len(),
        out_path.display()
    );
    Ok(RunSummary {
        files_scanned: all_data.len(),
        files_analyzed: all_data.len(),
        cached_files: all_data.len(),
        exported_segments: total_segments,
        failed_files: 0,
        output_path: Some(out_path),
    })
}

/// Analyse one file and return the probe + merged segments (no XML written).
fn analyze_one_data(
    path: &Path,
    config: &AnalysisConfig,
    cancel_flag: &Arc<AtomicBool>,
    worker: &mut analyzer::AnalyzerWorker,
) -> AppResult<(ProbeInfo, Vec<Segment>)> {
    if cancel_flag.load(Ordering::Relaxed) {
        return Err(AppError::Cancelled);
    }
    let probe = media::probe_video(path, &config.ffprobe_bin)?;
    let window_segments = worker.analyze_file(path, &probe, config, cancel_flag)?;
    let merged = timeline::merge_segments(window_segments);
    let selected = timeline::select_source_segments(probe.duration_seconds, merged);
    info!("{}: {} selected segment(s)", path.display(), selected.len());
    Ok((probe, selected))
}

fn validate_input_dir(input: &Path) -> AppResult<()> {
    let md = std::fs::metadata(input).map_err(|e| AppError::Io {
        path: input.to_path_buf(),
        source: e,
    })?;
    if !md.is_dir() {
        return Err(AppError::Unsupported(format!(
            "input must be a directory: {}",
            input.display()
        )));
    }
    Ok(())
}

fn parse_extensions(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|x| !x.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AnalysisConfig;
    use crate::timeline::{Segment, SegmentKind};
    use std::fs;

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
            kind: SegmentKind::GimbalMove,
            label_id: SegmentKind::GimbalMove.label_id(),
            motion_score: 3.5,
            zoom_score: 1.2,
            person_confidence: None,
            window_count: 2,
        }
    }

    fn sample_config() -> AnalysisConfig {
        let mut cfg = AnalysisConfig {
            ffmpeg_bin: PathBuf::from("ffmpeg"),
            ffprobe_bin: PathBuf::from("ffprobe"),
            yolo_model: None,
            enable_yolo: false,
            config_fingerprint: String::new(),
            analysis_height: 360,
            analysis_fps: 12.0,
            window_seconds: 1.0,
            motion_threshold: 1.8,
            person_confidence: 0.42,
            yolo_intra_threads: 1,
            ffmpeg_threads: 1,
            buf_frames: 4,
        };
        cfg.config_fingerprint = crate::cache::config_fingerprint(&cfg);
        cfg
    }

    #[test]
    fn export_from_cache_writes_premiere_xml() {
        let root = std::env::temp_dir()
            .join("video-tool-engine-test")
            .join(format!("smoke-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let source = root.join("clip.mov");
        fs::write(&source, b"fake-bytes").unwrap();
        let probe = sample_probe(source.clone());
        let seg = sample_segment(&source);
        let cfg = sample_config();
        let cache_dir = crate::cache::ensure_cache_dir(&root).unwrap();
        crate::cache::store(&cache_dir, &cfg, &probe, &[seg]).unwrap();

        let summary = export_from_cache(&root).unwrap();
        let xml_path = summary.output_path.expect("xml path");
        let xml = fs::read_to_string(xml_path).unwrap();

        assert_eq!(summary.files_analyzed, 1);
        assert_eq!(summary.cached_files, 1);
        assert!(xml.contains("<xmeml version=\"4\">"));
        assert!(xml.contains("<name>VT_Selects</name>"));
    }
}
