mod detector;
mod motion;

#[cfg(test)]
mod tests;

use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use tracing::debug;

use crate::config::AnalysisConfig;
use crate::error::{AppError, AppResult};
use crate::media::ProbeInfo;
use crate::timeline::{MovementType, Segment, SegmentKind};

use self::detector::{YoloDetector, detect_person_confidence};
use self::motion::{
    MotionFeatures, MotionSampling, average_pair_motion_features, estimate_pair_camera_motion,
    normalize_motion_features_for_fps, scaled_width_even, seconds_to_timeline_frame,
};

#[derive(Default)]
pub struct AnalyzerWorker {
    detector: Option<YoloDetector>,
    detector_initialized: bool,
}

struct WindowData {
    motion: MotionFeatures,
    person_confidence: Option<f32>,
    cinematic_score: f32,
    span: WindowSpan,
}

impl AnalyzerWorker {
    pub fn analyze_file(
        &mut self,
        input: &Path,
        probe: &ProbeInfo,
        config: &AnalysisConfig,
        cancel_flag: &Arc<AtomicBool>,
    ) -> AppResult<Vec<Segment>> {
        self.ensure_detector(config)?;
        analyze_file_impl(input, probe, config, cancel_flag, &mut self.detector)
    }

    fn ensure_detector(&mut self, config: &AnalysisConfig) -> AppResult<()> {
        if !config.enable_yolo {
            self.detector = None;
            self.detector_initialized = true;
            return Ok(());
        }
        if self.detector_initialized {
            return Ok(());
        }
        self.detector = YoloDetector::from_config(config)?;
        self.detector_initialized = true;
        Ok(())
    }
}

fn analyze_file_impl(
    input: &Path,
    probe: &ProbeInfo,
    config: &AnalysisConfig,
    cancel_flag: &Arc<AtomicBool>,
    detector: &mut Option<YoloDetector>,
) -> AppResult<Vec<Segment>> {
    let window_frames = analysis_window_frames(config)?;
    let (out_w, out_h, frame_bytes, vf) = analysis_pipe_settings(probe, config)?;

    let mut child = spawn_ffmpeg(&config.ffmpeg_bin, input, &vf, config.ffmpeg_threads)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Message("failed to capture ffmpeg stdout".to_string()))?;

    let buf_capacity = (frame_bytes * config.buf_frames).max(256 * 1024);
    let mut reader = BufReader::with_capacity(buf_capacity, stdout);
    let motion_sampling = MotionSampling::new(out_w as usize, out_h as usize);
    let mut frame = vec![0u8; frame_bytes];
    let mut motion_thumb = vec![0u8; motion_sampling.pixel_count()];
    let mut prev_motion_thumb: Option<Vec<u8>> = None;
    let mut pair_features: Vec<Option<MotionFeatures>> = Vec::new();
    let mut person_samples: Vec<(usize, f32)> = Vec::new();
    let person_detection_active = config.enable_yolo && detector.is_some();
    // Four subject samples per second catches brief entrances and gestures
    // without tying expensive detector frequency to the motion-analysis FPS.
    let person_sample_step = (config.analysis_fps / 4.0).round().max(1.0) as usize;
    let mut frames_loaded = 0usize;

    let mut windows_data = Vec::new();

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            terminate_ffmpeg(&mut child);
            return Err(AppError::Cancelled);
        }

        match reader.read_exact(&mut frame) {
            Ok(()) => {
                motion::sample_motion_frame_into(&frame, &mut motion_thumb, &motion_sampling);
                let pair_feature = prev_motion_thumb.as_ref().and_then(|prev| {
                    estimate_pair_camera_motion(prev, &motion_thumb, &motion_sampling)
                });
                if frames_loaded > 0 {
                    pair_features.push(pair_feature);
                }

                if person_detection_active
                    && frames_loaded.is_multiple_of(person_sample_step)
                    && let Some(confidence) = detect_person_confidence(
                        detector,
                        &frame,
                        out_w as usize,
                        out_h as usize,
                        config,
                    )?
                {
                    person_samples.push((frames_loaded, confidence.clamp(0.0, 1.0)));
                }

                prev_motion_thumb = Some(motion_thumb.clone());
                frames_loaded += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(AppError::CommandFailed {
                    cmd: "read ffmpeg rawvideo".to_string(),
                    source: e,
                });
            }
        }
    }

    finish_ffmpeg(child, input)?;

    if frames_loaded < 2 {
        return Ok(Vec::new());
    }

    let step_frames = analysis_step_frames(window_frames);
    for start_frame in analysis_window_starts(frames_loaded, window_frames, step_frames) {
        let end_frame = (start_frame + window_frames).min(frames_loaded);
        let pair_end = end_frame.saturating_sub(1);
        let motion = normalize_motion_features_for_fps(
            average_pair_motion_features(&pair_features[start_frame..pair_end]),
            config.analysis_fps,
        );
        let person_confidence = if person_detection_active {
            robust_person_confidence(
                person_samples
                    .iter()
                    .filter(|(frame_index, _)| {
                        *frame_index >= start_frame && *frame_index < end_frame
                    })
                    .map(|(_, confidence)| *confidence),
            )
        } else {
            None
        };
        let cinematic_score =
            calculate_cinematic_score(motion, person_confidence, probe.slow_motion);
        let start_seconds = start_frame as f64 / config.analysis_fps as f64;
        let sampled_end_seconds = end_frame as f64 / config.analysis_fps as f64;
        let end_seconds = sampled_end_seconds.min(probe.duration_seconds.max(start_seconds));

        windows_data.push(WindowData {
            motion,
            person_confidence,
            cinematic_score,
            span: WindowSpan {
                start_seconds,
                end_seconds,
            },
        });
    }

    fill_single_person_dropouts(&mut windows_data, config.person_confidence);
    for window in &mut windows_data {
        window.cinematic_score =
            calculate_cinematic_score(window.motion, window.person_confidence, probe.slow_motion);
    }

    let dynamic_threshold = if config.motion_threshold <= 0.0 {
        calculate_dynamic_motion_threshold(windows_data.iter().map(|w| w.motion.motion_score))
    } else {
        config.motion_threshold
    };

    let mut segments = Vec::new();
    let mut prev_kind: Option<SegmentKind> = None;

    for w in windows_data {
        let (kind, person_confidence) = classify_from_motion_and_detector(
            w.motion,
            w.person_confidence,
            dynamic_threshold,
            config,
            probe.slow_motion,
            prev_kind,
        );

        if let Some(kind) = kind {
            segments.push(build_segment(
                input,
                kind,
                w.motion,
                person_confidence,
                w.cinematic_score,
                w.span,
                probe.timebase,
            ));
        }
        prev_kind = kind;
    }

    debug!(
        "{}: emitted {} window segments (threshold: {:.2})",
        input.display(),
        segments.len(),
        dynamic_threshold
    );
    Ok(segments)
}

fn analysis_window_starts(
    total_frames: usize,
    configured_window_frames: usize,
    step_frames: usize,
) -> Vec<usize> {
    if total_frames < 2 {
        return Vec::new();
    }
    let window_frames = configured_window_frames.min(total_frames).max(2);
    if total_frames <= window_frames {
        return vec![0];
    }

    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + window_frames <= total_frames {
        starts.push(start);
        start = start.saturating_add(step_frames.max(1));
    }

    // Anchor one final window to the clip tail so the last fraction of a
    // second is never silently ignored.
    let tail_start = total_frames - window_frames;
    if starts.last().copied() != Some(tail_start) {
        starts.push(tail_start);
    }
    starts
}

fn robust_person_confidence(scores: impl Iterator<Item = f32>) -> Option<f32> {
    let mut scores = scores
        .filter(|score| score.is_finite())
        .map(|score| score.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    if scores.is_empty() {
        return None;
    }
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let strongest = scores[0];
    if strongest >= 0.78 {
        return Some(strongest);
    }
    let second = scores.get(1).copied().unwrap_or(0.0);
    if second >= strongest * 0.55 {
        Some(strongest * 0.60 + second * 0.40)
    } else {
        // A lone moderate hit is often a false positive from decor or a
        // reflection. Require either temporal support or a very strong hit.
        Some(strongest * 0.72)
    }
}

fn fill_single_person_dropouts(windows: &mut [WindowData], threshold: f32) {
    if windows.len() < 3 {
        return;
    }
    let original = windows
        .iter()
        .map(|window| window.person_confidence)
        .collect::<Vec<_>>();
    for i in 1..windows.len() - 1 {
        let current = original[i].unwrap_or(0.0);
        let left = original[i - 1].unwrap_or(0.0);
        let right = original[i + 1].unwrap_or(0.0);
        if current < threshold && left >= threshold && right >= threshold {
            windows[i].person_confidence = Some(left.min(right));
        }
    }
}

fn calculate_dynamic_motion_threshold(scores: impl Iterator<Item = f32>) -> f32 {
    let mut scores = scores
        .filter(|score| score.is_finite())
        .map(|score| score.max(0.0))
        .collect::<Vec<_>>();
    if scores.is_empty() {
        return 1.0;
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile = |p: usize| scores[(scores.len().saturating_sub(1) * p) / 100];
    let low_motion_floor = percentile(20);
    let median = percentile(50);
    // Use the lower distribution as a per-clip noise estimate, but cap the
    // result so a clip containing continuous movement cannot normalize its
    // own useful signal away.
    let estimated = 0.55 + low_motion_floor * 0.55 + median * 0.18;
    let sustained_motion_cap = if median >= 1.0 {
        median * 0.82
    } else {
        f32::INFINITY
    };
    estimated.min(sustained_motion_cap).clamp(0.85, 2.40)
}

fn classify_from_motion_and_detector(
    motion: MotionFeatures,
    person_confidence: Option<f32>,
    active_motion_threshold: f32,
    config: &AnalysisConfig,
    source_is_slow_motion: bool,
    prev_kind: Option<SegmentKind>,
) -> (Option<SegmentKind>, Option<f32>) {
    let person_norm = person_confidence
        .map(|c| {
            if config.person_confidence > 0.0 {
                c / config.person_confidence
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    let person_enter: f32 = match prev_kind {
        Some(SegmentKind::StaticSubject) => 0.78,
        _ => 1.0,
    };

    let is_person_present = person_norm >= person_enter;

    let motion_norm = if active_motion_threshold > 0.0 {
        motion.motion_score / active_motion_threshold
    } else {
        0.0
    };

    let (motion_enter, coherent_required, smoothness_required) = match prev_kind {
        Some(SegmentKind::GimbalMove) => (0.72, 0.26, 0.30),
        Some(SegmentKind::StaticSubject) if is_person_present => (1.60, 0.42, 0.42),
        _ => (1.0, 0.34, 0.42),
    };

    let combined_coherence =
        motion.confidence * (0.35 + motion.temporal_smoothness.clamp(0.0, 1.0) * 0.65);
    let coherent_camera_move = combined_coherence >= coherent_required
        && motion.temporal_smoothness >= smoothness_required;

    if source_is_slow_motion {
        let slow_motion_enter = match prev_kind {
            Some(SegmentKind::SlowMotion) => 0.16,
            Some(SegmentKind::StaticSubject) if is_person_present => 0.45,
            _ => 0.25,
        };
        let zoom_norm = if active_motion_threshold > 0.0 {
            motion.zoom_score / active_motion_threshold
        } else {
            0.0
        };
        let coherent_slow_zoom =
            zoom_norm >= 0.20 && motion.confidence >= 0.20 && motion.temporal_smoothness >= 0.38;
        if (motion_norm >= slow_motion_enter && coherent_camera_move) || coherent_slow_zoom {
            return (Some(SegmentKind::SlowMotion), None);
        }
    }

    if motion_norm >= motion_enter && coherent_camera_move {
        return (Some(SegmentKind::GimbalMove), None);
    }

    if is_person_present {
        (Some(SegmentKind::StaticSubject), person_confidence)
    } else {
        (None, person_confidence)
    }
}

fn build_segment(
    input: &Path,
    kind: SegmentKind,
    motion: MotionFeatures,
    person_confidence: Option<f32>,
    cinematic_score: f32,
    span: WindowSpan,
    timebase: u32,
) -> Segment {
    Segment {
        source_path: input.to_path_buf(),
        start_frame: seconds_to_timeline_frame(span.start_seconds, timebase),
        end_frame: seconds_to_timeline_frame(span.end_seconds, timebase),
        start_seconds: span.start_seconds,
        end_seconds: span.end_seconds,
        kind,
        label_id: kind.label_id(),
        motion_score: motion.motion_score,
        zoom_score: motion.zoom_score,
        movement_type: segment_movement_type(kind, motion.movement_type),
        motion_confidence: motion.confidence,
        motion_smoothness: motion.temporal_smoothness,
        person_confidence,
        window_count: 1,
        cinematic_score,
    }
}

fn calculate_cinematic_score(
    motion: MotionFeatures,
    person: Option<f32>,
    slow_motion: bool,
) -> f32 {
    let motion_quality = (motion.confidence.clamp(0.0, 1.0) * 0.35
        + motion.temporal_smoothness.clamp(0.0, 1.0) * 0.65)
        .clamp(0.0, 1.0);
    let subject_signal = person.unwrap_or(0.0).clamp(0.0, 1.0);
    let slow_mo_bonus = if slow_motion { 0.15 } else { 0.0 };

    (motion_quality * 0.50 + subject_signal * 0.35 + slow_mo_bonus).clamp(0.0, 1.0)
}

fn segment_movement_type(kind: SegmentKind, movement_type: MovementType) -> MovementType {
    match kind {
        SegmentKind::GimbalMove => movement_type,
        SegmentKind::StaticSubject | SegmentKind::Static => MovementType::Subject,
        SegmentKind::SlowMotion => MovementType::SlowMotion,
    }
}

fn analysis_window_frames(config: &AnalysisConfig) -> AppResult<usize> {
    let window_frames = (config.analysis_fps * config.window_seconds).round() as usize;
    if window_frames < 2 {
        return Err(AppError::Unsupported(
            "window must contain at least 2 frames; increase window_seconds or analysis_fps"
                .to_string(),
        ));
    }
    Ok(window_frames)
}

fn analysis_step_frames(window_frames: usize) -> usize {
    (window_frames / 4).max(1)
}

fn analysis_pipe_settings(
    probe: &ProbeInfo,
    config: &AnalysisConfig,
) -> AppResult<(u32, u32, usize, String)> {
    let out_w = scaled_width_even(probe.width, probe.height, config.analysis_height);
    let out_h = config.analysis_height.max(2);
    let frame_bytes = (out_w as usize)
        .saturating_mul(out_h as usize)
        .saturating_mul(3);
    if frame_bytes == 0 {
        return Err(AppError::Unsupported(
            "invalid analysis frame size".to_string(),
        ));
    }

    // Using bicubic for high-quality downsampling of mirrorless 4K footage.
    let vf = format!(
        "scale=-2:{}:flags=bicubic,fps={}",
        config.analysis_height, config.analysis_fps
    );

    Ok((out_w, out_h, frame_bytes, vf))
}

fn spawn_ffmpeg(
    ffmpeg_bin: &Path,
    input: &Path,
    vf: &str,
    ffmpeg_threads: usize,
) -> AppResult<std::process::Child> {
    let mut cmd = Command::new(ffmpeg_bin);
    suppress_child_console(&mut cmd);
    cmd.args(["-hide_banner", "-loglevel", "error"]);

    if ffmpeg_threads > 0 {
        cmd.args(["-threads", &ffmpeg_threads.to_string()]);
    }
    cmd.arg("-i").arg(input).args([
        "-an", "-sn", "-dn", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    cmd.spawn().map_err(|e| AppError::CommandFailed {
        cmd: format!("{} ... {}", ffmpeg_bin.display(), input.display()),
        source: e,
    })
}

fn suppress_child_console(_cmd: &mut Command) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn finish_ffmpeg(child: std::process::Child, input: &Path) -> AppResult<()> {
    let output = child
        .wait_with_output()
        .map_err(|e| AppError::CommandFailed {
            cmd: "wait ffmpeg".to_string(),
            source: e,
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::CommandNonZero {
            cmd: format!("ffmpeg ({}) — {}", input.display(), stderr.trim()),
            code: output.status.code().unwrap_or(-1),
        });
    }
    Ok(())
}

fn terminate_ffmpeg(child: &mut std::process::Child) {
    if let Err(e) = child.kill() {
        debug!("ffmpeg kill failed: {e}");
    }
    let _ = child.wait();
}

#[derive(Debug, Clone, Copy)]
struct WindowSpan {
    start_seconds: f64,
    end_seconds: f64,
}
