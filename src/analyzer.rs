mod detector;
mod motion;

#[cfg(test)]
mod tests;

use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use tracing::debug;

use crate::config::AnalysisConfig;
use crate::error::{AppError, AppResult};
use crate::media::ProbeInfo;
use crate::timeline::{MovementType, Segment, SegmentKind};

use self::detector::{YoloDetector, detect_person_confidence};
use self::motion::{
    MotionFeatures, MotionSampling, average_pair_motion_features_at_fps,
    estimate_pair_camera_motion, scaled_width_even, seconds_to_timeline_frame,
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
    if probe.vfr {
        return Err(AppError::Unsupported(format!(
            "variable-frame-rate media is not supported for exact XML trims: {}",
            input.display()
        )));
    }
    let window_frames = analysis_window_frames(config)?;
    let person_detection_active = config.enable_yolo && detector.is_some();
    let (out_w, out_h, frame_bytes, vf, pix_fmt) =
        analysis_pipe_settings(probe, config, person_detection_active)?;

    let (mut child, stderr_thread) = spawn_ffmpeg(
        &config.ffmpeg_bin,
        input,
        probe.stream_index,
        &vf,
        pix_fmt,
        config.ffmpeg_threads,
    )?;
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_ffmpeg(&mut child, stderr_thread);
            return Err(AppError::Message(
                "failed to capture ffmpeg stdout".to_string(),
            ));
        }
    };

    let buf_capacity = frame_bytes
        .saturating_mul(config.buf_frames.max(1))
        .clamp(256 * 1024, 8 * 1024 * 1024);
    let mut reader = BufReader::with_capacity(buf_capacity, stdout);
    let motion_sampling = MotionSampling::new(out_w as usize, out_h as usize);
    let mut frame = vec![0u8; frame_bytes];
    let mut motion_thumb = vec![0u8; motion_sampling.pixel_count()];
    let mut prev_motion_thumb = vec![0u8; motion_sampling.pixel_count()];
    let mut have_prev_motion_thumb = false;
    let mut pair_features: Vec<Option<MotionFeatures>> = Vec::new();
    let mut person_samples: Vec<(usize, f32)> = Vec::new();
    // Four subject samples per second catches brief entrances and gestures
    // without tying expensive detector frequency to the motion-analysis FPS.
    let person_sample_step = (config.analysis_fps / 4.0).round().max(1.0) as usize;
    let mut frames_loaded = 0usize;

    let mut windows_data = Vec::new();

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            terminate_ffmpeg(&mut child, stderr_thread);
            return Err(AppError::Cancelled);
        }

        match reader.read(&mut frame[..1]) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                terminate_ffmpeg(&mut child, stderr_thread);
                return Err(AppError::CommandFailed {
                    cmd: "read ffmpeg rawvideo".to_string(),
                    source: e,
                });
            }
        }
        if let Err(e) = reader.read_exact(&mut frame[1..]) {
            terminate_ffmpeg(&mut child, stderr_thread);
            return Err(AppError::CommandFailed {
                cmd: "read complete ffmpeg rawvideo frame".to_string(),
                source: e,
            });
        }
        if person_detection_active {
            motion::sample_motion_frame_into(&frame, &mut motion_thumb, &motion_sampling);
        } else {
            motion::sample_motion_gray_into(&frame, &mut motion_thumb, &motion_sampling);
        }
        let pair_feature = if have_prev_motion_thumb {
            estimate_pair_camera_motion(&prev_motion_thumb, &motion_thumb, &motion_sampling)
        } else {
            None
        };
        if have_prev_motion_thumb {
            pair_features.push(pair_feature);
        }

        if person_detection_active
            && frames_loaded.is_multiple_of(person_sample_step)
            && let Some(confidence) =
                detect_person_confidence(detector, &frame, out_w as usize, out_h as usize, config)?
        {
            person_samples.push((frames_loaded, confidence.clamp(0.0, 1.0)));
        }

        std::mem::swap(&mut prev_motion_thumb, &mut motion_thumb);
        have_prev_motion_thumb = true;
        frames_loaded += 1;
    }

    finish_ffmpeg(child, stderr_thread, input)?;

    if frames_loaded < 2 {
        return Ok(Vec::new());
    }

    let mut person_cursor = 0usize;
    for start_frame in analysis_window_starts(frames_loaded, window_frames) {
        let end_frame = (start_frame + window_frames).min(frames_loaded);
        let pair_end = end_frame.saturating_sub(1);
        let motion = average_pair_motion_features_at_fps(
            &pair_features[start_frame..pair_end],
            config.analysis_fps,
        );
        let person_confidence = if person_detection_active {
            while person_cursor < person_samples.len()
                && person_samples[person_cursor].0 < start_frame
            {
                person_cursor += 1;
            }
            let mut person_end = person_cursor;
            while person_end < person_samples.len() && person_samples[person_end].0 < end_frame {
                person_end += 1;
            }
            robust_person_confidence(
                person_samples[person_cursor..person_end]
                    .iter()
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
                (probe.fps_num, probe.fps_den),
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

fn analysis_window_starts(total_frames: usize, configured_window_frames: usize) -> Vec<usize> {
    if total_frames < 2 {
        return Vec::new();
    }
    let window_frames = configured_window_frames.min(total_frames).max(2);
    if total_frames <= window_frames {
        return vec![0];
    }

    let stride = (window_frames / 2).max(1);
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start < total_frames {
        let end = (start + window_frames).min(total_frames);
        if end.saturating_sub(start) >= 2 {
            starts.push(start);
        }
        if end == total_frames {
            break;
        }
        start = start.saturating_add(stride);
    }
    starts
}

fn robust_person_confidence(scores: impl Iterator<Item = f32>) -> Option<f32> {
    let mut strongest = 0.0f32;
    let mut second = 0.0f32;
    let mut count = 0usize;
    for score in scores.filter(|score| score.is_finite()) {
        let score = score.clamp(0.0, 1.0);
        count += 1;
        if score >= strongest {
            second = strongest;
            strongest = score;
        } else if score > second {
            second = score;
        }
    }
    if count == 0 {
        return None;
    }
    if strongest >= 0.78 {
        return Some(strongest);
    }
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
    fps: (u32, u32),
) -> Segment {
    let (fps_num, fps_den) = fps;
    Segment {
        source_path: input.to_path_buf(),
        start_frame: seconds_to_timeline_frame(span.start_seconds, fps_num, fps_den),
        end_frame: seconds_to_timeline_frame(span.end_seconds, fps_num, fps_den),
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

fn analysis_pipe_settings(
    probe: &ProbeInfo,
    config: &AnalysisConfig,
    color_output: bool,
) -> AppResult<(u32, u32, usize, String, &'static str)> {
    let out_h = if color_output {
        config.analysis_height.max(2)
    } else {
        motion::motion_decode_height(config.analysis_height)
    };
    let out_w = scaled_width_even(probe.width, probe.height, out_h);
    let channels = if color_output { 3 } else { 1 };
    let frame_bytes = (out_w as usize)
        .saturating_mul(out_h as usize)
        .saturating_mul(channels);
    if frame_bytes == 0 {
        return Err(AppError::Unsupported(
            "invalid analysis frame size".to_string(),
        ));
    }

    // Using bicubic for high-quality downsampling of mirrorless 4K footage.
    let vf = format!(
        "scale={out_w}:{out_h}:flags=bicubic,fps={}",
        config.analysis_fps
    );

    Ok((
        out_w,
        out_h,
        frame_bytes,
        vf,
        if color_output { "bgr24" } else { "gray" },
    ))
}

fn spawn_ffmpeg(
    ffmpeg_bin: &Path,
    input: &Path,
    stream_index: usize,
    vf: &str,
    pix_fmt: &str,
    ffmpeg_threads: usize,
) -> AppResult<(std::process::Child, JoinHandle<Vec<u8>>)> {
    let mut cmd = Command::new(ffmpeg_bin);
    suppress_child_console(&mut cmd);
    cmd.args(["-hide_banner", "-loglevel", "error"]);

    if ffmpeg_threads > 0 {
        cmd.args(["-threads", &ffmpeg_threads.to_string()]);
    }
    cmd.args(["-noautorotate", "-i"]).arg(input).args([
        "-map",
        &format!("0:{stream_index}"),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf,
        "-pix_fmt",
        pix_fmt,
        "-f",
        "rawvideo",
        "pipe:1",
    ]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| AppError::CommandFailed {
        cmd: format!("{} ... {}", ffmpeg_bin.display(), input.display()),
        source: e,
    })?;
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AppError::Message(
                "failed to capture ffmpeg stderr".to_string(),
            ));
        }
    };
    let stderr_thread = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut reader = std::io::BufReader::new(stderr);
        let _ = reader.read_to_end(&mut bytes);
        bytes
    });
    Ok((child, stderr_thread))
}

fn suppress_child_console(_cmd: &mut Command) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn finish_ffmpeg(
    mut child: std::process::Child,
    stderr_thread: JoinHandle<Vec<u8>>,
    input: &Path,
) -> AppResult<()> {
    let status = match child.wait() {
        Ok(status) => status,
        Err(e) => {
            let _ = stderr_thread.join();
            return Err(AppError::CommandFailed {
                cmd: "wait ffmpeg".to_string(),
                source: e,
            });
        }
    };
    let stderr = stderr_thread.join().unwrap_or_default();
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        return Err(AppError::CommandNonZero {
            cmd: format!("ffmpeg ({}) — {}", input.display(), stderr.trim()),
            code: status.code().unwrap_or(-1),
        });
    }
    Ok(())
}

fn terminate_ffmpeg(child: &mut std::process::Child, stderr_thread: JoinHandle<Vec<u8>>) {
    if let Err(e) = child.kill() {
        debug!("ffmpeg kill failed: {e}");
    }
    let _ = child.wait();
    let _ = stderr_thread.join();
}

#[derive(Debug, Clone, Copy)]
struct WindowSpan {
    start_seconds: f64,
    end_seconds: f64,
}
