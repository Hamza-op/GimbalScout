use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use std::sync::OnceLock;

use tracing::{debug, error, info, warn};
use tracing_subscriber::reload;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

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
    /// Include linked source audio in the Premiere XML when available.
    pub include_audio: bool,
    /// Maximum duration of the peak-centered select exported per source.
    pub max_select_seconds: f32,
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
    /// Comma-separated list of video extensions.
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
            include_audio: false,
            max_select_seconds: 8.0,
            ffmpeg_bin: None,
            ffprobe_bin: None,
            analysis_height: 720,
            analysis_fps: 18.0,
            window_seconds: 1.0,
            motion_threshold: 0.0,
            person_confidence: 0.42,
            max_files: None,
            yolo_intra_threads: None,
            ffmpeg_threads: None,
            buf_frames: None,
            extensions: media::DEFAULT_VIDEO_EXTENSIONS.to_string(),
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
    pub selected_duration_seconds: f64,
    pub movement_segments: usize,
    pub subject_segments: usize,
    pub slow_motion_segments: usize,
    pub static_segments: usize,
    /// Sources longer than 90 seconds that bypassed analysis and were kept
    /// from first frame to last frame.
    pub preserved_segments: usize,
    /// Number of selections with linked source audio in the XML.
    pub audio_segments: usize,
    pub failed_files: usize,
    /// Source files that still failed after a serial retry.
    pub failed_paths: Vec<PathBuf>,
    /// Path of the single merged XML that was written.
    pub output_path: Option<PathBuf>,
}

type AnalyzeResult = AppResult<(ProbeInfo, Vec<Segment>)>;
type WorkerResult = (PathBuf, AnalyzeResult, bool);
type TraceReloadHandle = reload::Handle<EnvFilter, Registry>;

static TRACE_RELOAD: OnceLock<TraceReloadHandle> = OnceLock::new();

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
    let filter = initial_tracing_filter(verbose);
    let (filter_layer, reload_handle) = reload::Layer::new(filter);
    let subscriber = tracing_subscriber::registry()
        .with(filter_layer)
        .with(tracing_subscriber::fmt::layer().with_target(false));

    if subscriber.try_init().is_ok() {
        let _ = TRACE_RELOAD.set(reload_handle);
    }
}

fn initial_tracing_filter(verbose: bool) -> EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| default_tracing_filter(verbose))
}

fn default_tracing_filter(verbose: bool) -> EnvFilter {
    let filter = if verbose {
        "info,video_tool=debug"
    } else {
        "info"
    };
    EnvFilter::new(filter)
}

fn set_tracing_verbose(verbose: bool) {
    if std::env::var_os("RUST_LOG").is_some() {
        return;
    }

    if let Some(handle) = TRACE_RELOAD.get()
        && let Err(e) = handle.reload(default_tracing_filter(verbose))
    {
        warn!("failed to update tracing filter: {e}");
    }
}

pub fn run_analyze(
    mut args: AnalyzeArgs,
    progress_tx: Option<mpsc::Sender<ProgressMsg>>,
) -> AppResult<RunSummary> {
    set_tracing_verbose(args.verbose);
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
    if !args.max_select_seconds.is_finite() || !(2.0..=30.0).contains(&args.max_select_seconds) {
        return Err(AppError::Unsupported(
            "select length must be between 2 and 30 seconds".to_string(),
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
        phase: format!("{} · scanning input folder…", config.acceleration.label()),
    });

    // Keep the file worker count in lock-step with the balanced budget
    // computed in AnalysisConfig: ⌈cpus/2⌉ workers, each of which spawns its
    // own ffmpeg/YOLO threads.  Running more workers than the config budget
    // assumes re-introduces the oversubscription the budget was designed to
    // avoid.
    let threads = args
        .max_files
        .unwrap_or_else(|| {
            if config.acceleration.gpu_heavy {
                gpu_worker_count()
            } else {
                default_worker_count()
            }
        })
        .clamp(1, if config.acceleration.gpu_heavy { 6 } else { 8 });
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

                let _ = tx.send((path.clone(), result, from_cache));

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

    let mut failed_paths = Vec::new();
    let mut cached_files = 0usize;
    let mut all_data: Vec<(ProbeInfo, Vec<Segment>)> = Vec::with_capacity(discovered);
    for (path, result, from_cache) in raw_results {
        let result = match result {
            Ok(data) => Ok(data),
            Err(AppError::Cancelled) => return Err(AppError::Cancelled),
            Err(first_error) => {
                warn!(
                    "{}: initial analysis failed ({first_error}); retrying serially",
                    path.display()
                );
                let mut retry_worker = analyzer::AnalyzerWorker::default();
                let retry =
                    analyze_one_data(&path, &work_config, &args.cancel_flag, &mut retry_worker);
                if let Ok((probe, segments)) = &retry
                    && let Err(e) = cache::store(&cache_dir, &work_config, probe, segments)
                {
                    warn!("cache store failed after retry for {}: {e}", path.display());
                }
                retry
            }
        };

        match result {
            Ok(data) => {
                if from_cache {
                    cached_files += 1;
                }
                all_data.push(data);
            }
            Err(err) => {
                error!("{}: analysis failed after retry: {err}", path.display());
                failed_paths.push(path);
            }
        }
    }

    // Sidecars retain raw analysis windows. Editorial trimming is cheap and
    // happens here, so changing select length or audio export does not force
    // another decode/YOLO pass.
    let export_data = prepare_export_data(all_data, f64::from(args.max_select_seconds));

    // Write one merged XML for all clips. `export_data` aggregates both
    // freshly-analysed results and entries rehydrated from the sidecar
    // cache — the XML exporter does not need to know the difference.
    let outcome = xml_exporter::export_all_with_options(
        &export_data,
        &args.output,
        xml_exporter::ExportOptions {
            include_audio: args.include_audio,
        },
    )?;
    let out_path = outcome.path;
    let stats = outcome.stats;
    let exported_segments = stats.total_segments;
    info!(
        "Exported {exported_segments} best selections across {} files ({} from cache) → {}",
        export_data.len(),
        cached_files,
        out_path.display()
    );

    let summary = RunSummary {
        files_scanned: discovered,
        files_analyzed: export_data.len(),
        cached_files,
        exported_segments,
        selected_duration_seconds: stats.duration_seconds,
        movement_segments: stats.movement_segments,
        subject_segments: stats.subject_segments,
        slow_motion_segments: stats.slow_motion_segments,
        static_segments: stats.static_segments,
        preserved_segments: stats.preserved_segments,
        audio_segments: stats.audio_segments,
        failed_files: failed_paths.len(),
        failed_paths,
        output_path: Some(out_path),
    };

    Ok(summary)
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().div_ceil(2).clamp(1, 8))
        .unwrap_or(4)
}

fn gpu_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().div_ceil(2).clamp(2, 6))
        .unwrap_or(3)
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
    let raw_data = cache::load_all(&cache_dir)?;
    if raw_data.is_empty() {
        warn!(
            "No cache entries found under {} — nothing to export.",
            cache_dir.display()
        );
        return Ok(RunSummary::default());
    }
    let all_data = prepare_export_data(raw_data, 8.0);
    let outcome = xml_exporter::export_all_with_options(
        &all_data,
        output,
        xml_exporter::ExportOptions::default(),
    )?;
    let out_path = outcome.path;
    let stats = outcome.stats;
    let exported_segments = stats.total_segments;
    info!(
        "Exported {exported_segments} best selections from cache across {} files → {}",
        all_data.len(),
        out_path.display()
    );
    Ok(RunSummary {
        files_scanned: all_data.len(),
        files_analyzed: all_data.len(),
        cached_files: all_data.len(),
        exported_segments,
        selected_duration_seconds: stats.duration_seconds,
        movement_segments: stats.movement_segments,
        subject_segments: stats.subject_segments,
        slow_motion_segments: stats.slow_motion_segments,
        static_segments: stats.static_segments,
        preserved_segments: stats.preserved_segments,
        audio_segments: stats.audio_segments,
        failed_files: 0,
        failed_paths: Vec::new(),
        output_path: Some(out_path),
    })
}

/// Analyse one file and return the probe + raw overlapping windows. Keeping
/// these windows in the cache lets editorial selection policy evolve without
/// rerunning expensive media inference.
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
    if timeline::requires_original_preservation(probe.duration_seconds) {
        info!(
            "{}: {:.2}s source exceeds the {:.0}s safe-selection limit; preserving the original and skipping content analysis",
            path.display(),
            probe.duration_seconds,
            timeline::SAFE_SELECTION_MAX_SECONDS,
        );
        let preserved = timeline::preserved_original_segment(
            &probe.source_path,
            probe.duration_seconds,
            probe.duration_frames,
        );
        return Ok((probe, vec![preserved]));
    }
    let windows = worker.analyze_file(path, &probe, config, cancel_flag)?;
    info!("{}: {} analysis window(s)", path.display(), windows.len());
    Ok((probe, windows))
}

fn prepare_export_data(
    raw_data: Vec<(ProbeInfo, Vec<Segment>)>,
    max_select_seconds: f64,
) -> Vec<(ProbeInfo, Vec<Segment>)> {
    raw_data
        .into_iter()
        .map(|(probe, windows)| {
            if timeline::requires_original_preservation(probe.duration_seconds) {
                let preserved = timeline::preserved_original_segment(
                    &probe.source_path,
                    probe.duration_seconds,
                    probe.duration_frames,
                );
                return (probe, vec![preserved]);
            }
            let merged = timeline::merge_segments(windows.clone());
            let mut selected = timeline::select_source_segments(
                probe.duration_seconds,
                merged,
                &timeline::SensitivityConfig::default(),
            );
            if selected.is_empty() {
                selected.push(best_static_fallback(&probe, &windows));
            }
            timeline::focus_editorial_highlights(
                &mut selected,
                &windows,
                probe.duration_seconds,
                probe.fps_num,
                probe.fps_den,
                max_select_seconds,
            );
            (probe, selected)
        })
        .collect()
}

fn best_static_fallback(probe: &ProbeInfo, windows: &[Segment]) -> Segment {
    let mut fallback = windows
        .iter()
        .max_by(|a, b| {
            timeline::segment_quality_score(a).total_cmp(&timeline::segment_quality_score(b))
        })
        .cloned()
        .unwrap_or_else(|| {
            let center = probe.duration_seconds * 0.5;
            let half_window = (probe.duration_seconds.min(1.0)) * 0.5;
            let start_seconds = (center - half_window).max(0.0);
            let end_seconds = (center + half_window).min(probe.duration_seconds);
            Segment {
                source_path: probe.source_path.clone(),
                start_frame: seconds_to_source_frame(start_seconds, probe.fps_num, probe.fps_den),
                end_frame: seconds_to_source_frame(end_seconds, probe.fps_num, probe.fps_den),
                start_seconds,
                end_seconds,
                kind: timeline::SegmentKind::Static,
                label_id: timeline::SegmentKind::Static.label_id(),
                motion_score: 0.0,
                zoom_score: 0.0,
                movement_type: timeline::MovementType::Subject,
                motion_confidence: 0.0,
                motion_smoothness: 0.0,
                person_confidence: None,
                window_count: 1,
                cinematic_score: 0.0,
            }
        });
    fallback.kind = timeline::SegmentKind::Static;
    fallback.label_id = timeline::SegmentKind::Static.label_id();
    fallback.movement_type = timeline::MovementType::Subject;
    fallback.person_confidence = None;
    fallback
}

fn seconds_to_source_frame(seconds: f64, fps_num: u32, fps_den: u32) -> u64 {
    if fps_num == 0 || fps_den == 0 {
        return 0;
    }
    (seconds.max(0.0) * fps_num as f64 / fps_den as f64).round() as u64
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
            stream_index: 0,
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
            vfr: false,
            audio: None,
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
            movement_type: crate::timeline::MovementType::PanTilt,
            motion_confidence: 0.88,
            motion_smoothness: 0.88,
            person_confidence: None,
            window_count: 1,
            cinematic_score: 0.0,
        }
    }

    fn sample_static_subject_segment(source: &Path) -> Segment {
        Segment {
            source_path: source.to_path_buf(),
            start_frame: 25,
            end_frame: 75,
            start_seconds: 1.0,
            end_seconds: 3.0,
            kind: SegmentKind::StaticSubject,
            label_id: SegmentKind::StaticSubject.label_id(),
            motion_score: 0.4,
            zoom_score: 0.0,
            movement_type: crate::timeline::MovementType::Subject,
            motion_confidence: 0.72,
            motion_smoothness: 0.85,
            person_confidence: Some(0.91),
            window_count: 1,
            cinematic_score: 0.65,
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
            acceleration: crate::config::AccelerationInfo::default(),
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

    #[test]
    fn export_from_cache_matches_direct_export_for_selected_data() {
        let root = std::env::temp_dir()
            .join("video-tool-engine-test")
            .join(format!("cache-equivalence-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let media_dir = root.join("media");
        let direct_dir = root.join("direct");
        let cached_dir = root.join("cached");
        fs::create_dir_all(&media_dir).unwrap();
        fs::create_dir_all(&direct_dir).unwrap();
        fs::create_dir_all(&cached_dir).unwrap();

        let source = media_dir.join("person.mov");
        fs::write(&source, b"fake-bytes").unwrap();
        let probe = sample_probe(source.clone());
        let seg = sample_static_subject_segment(&source);

        let direct_data = prepare_export_data(vec![(probe.clone(), vec![seg.clone()])], 8.0);
        let direct_path = crate::xml_exporter::export_all(&direct_data, &direct_dir).unwrap();
        let cfg = sample_config();
        let cache_dir = crate::cache::ensure_cache_dir(&cached_dir).unwrap();
        crate::cache::store(&cache_dir, &cfg, &probe, &[seg]).unwrap();

        let summary = export_from_cache(&cached_dir).unwrap();
        let cached_path = summary.output_path.expect("cached xml path");
        let direct_xml = fs::read_to_string(direct_path).unwrap();
        let cached_xml = fs::read_to_string(cached_path).unwrap();

        assert_eq!(cached_xml, direct_xml);
        assert!(cached_xml.contains("<label2>Caribbean</label2>"));
    }

    #[test]
    fn export_preparation_replaces_long_source_cuts_with_the_full_original() {
        let source = PathBuf::from("ceremony.mov");
        let mut probe = sample_probe(source.clone());
        probe.duration_seconds = 91.0;
        probe.duration_frames = 2_275;
        let mut old_cached_cut = sample_segment(&source);
        old_cached_cut.start_frame = 500;
        old_cached_cut.end_frame = 700;
        old_cached_cut.start_seconds = 20.0;
        old_cached_cut.end_seconds = 28.0;

        let prepared = prepare_export_data(vec![(probe, vec![old_cached_cut])], 8.0);
        let segments = &prepared[0].1;

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].kind, SegmentKind::PreservedOriginal);
        assert_eq!(segments[0].start_frame, 0);
        assert_eq!(segments[0].end_frame, 2_275);
        assert_eq!(segments[0].start_seconds, 0.0);
        assert_eq!(segments[0].end_seconds, 91.0);
    }
}
