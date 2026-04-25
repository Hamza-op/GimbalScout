mod detector;
mod motion;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
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
use crate::timeline::{Segment, SegmentKind};

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
    let mut ring: VecDeque<Vec<u8>> = (0..window_frames).map(|_| vec![0u8; frame_bytes]).collect();

    let motion_sampling = MotionSampling::new(out_w as usize, out_h as usize);
    let mut motion_ring: VecDeque<Vec<u8>> = (0..window_frames)
        .map(|_| vec![0u8; motion_sampling.pixel_count()])
        .collect();
    let pair_window = window_frames.saturating_sub(1).max(1);
    let mut prev_motion_thumb: Option<Vec<u8>> = None;
    let mut pair_features: VecDeque<Option<MotionFeatures>> = VecDeque::with_capacity(pair_window);

    let step_frames = analysis_step_frames(window_frames);
    let mut frames_loaded: usize = 0;
    let mut frames_since_emit: usize = 0;
    let mut window_start_frame: usize = 0;
    let mut segments = Vec::new();
    let mut prev_kind: Option<SegmentKind> = None;

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            terminate_ffmpeg(&mut child);
            return Err(AppError::Cancelled);
        }

        let mut frame = ring.pop_front().expect("frame buffer available");
        let mut motion_thumb = motion_ring.pop_front().expect("motion buffer available");
        match reader.read_exact(&mut frame) {
            Ok(()) => {
                motion::sample_motion_frame_into(&frame, &mut motion_thumb, &motion_sampling);
                let pair_feature = prev_motion_thumb.as_ref().and_then(|prev| {
                    estimate_pair_camera_motion(prev, &motion_thumb, &motion_sampling)
                });
                pair_features.push_back(pair_feature);
                while pair_features.len() > pair_window {
                    pair_features.pop_front();
                }
                prev_motion_thumb = Some(motion_thumb.clone());
                ring.push_back(frame);
                motion_ring.push_back(motion_thumb);

                frames_loaded += 1;
                if frames_loaded < window_frames {
                    continue;
                }
                if frames_loaded > window_frames {
                    frames_since_emit += 1;
                    if frames_since_emit < step_frames {
                        continue;
                    }
                }

                let motion = normalize_motion_features_for_fps(
                    average_pair_motion_features(&pair_features),
                    config.analysis_fps,
                );
                let center_index = window_frames / 2;
                let center_frame = ring.get(center_index).expect("center frame in window");
                let (kind, person_confidence) = classify_from_motion_and_detector(
                    motion,
                    detector,
                    DetectorFrame {
                        bgr: center_frame,
                        width: out_w as usize,
                        height: out_h as usize,
                    },
                    config,
                    probe.slow_motion,
                    prev_kind,
                )?;

                if let Some(kind) = kind {
                    segments.push(build_segment(
                        input,
                        kind,
                        motion.motion_score,
                        motion.zoom_score,
                        person_confidence,
                        WindowSpan {
                            start_seconds: window_start_frame as f64 / config.analysis_fps as f64,
                            duration_seconds: config.window_seconds as f64,
                        },
                        probe.timebase,
                    ));
                }
                prev_kind = kind;
                window_start_frame += step_frames;
                frames_since_emit = 0;
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
    debug!(
        "{}: emitted {} window segments",
        input.display(),
        segments.len()
    );
    Ok(segments)
}

fn classify_from_motion_and_detector(
    motion: MotionFeatures,
    detector: &mut Option<YoloDetector>,
    frame: DetectorFrame<'_>,
    config: &AnalysisConfig,
    source_is_slow_motion: bool,
    prev_kind: Option<SegmentKind>,
) -> AppResult<(Option<SegmentKind>, Option<f32>)> {
    let motion_norm = if config.motion_threshold > 0.0 {
        motion.motion_score / config.motion_threshold
    } else {
        0.0
    };

    let motion_enter: f32 = match prev_kind {
        Some(SegmentKind::GimbalMove | SegmentKind::SlowMotion) => 0.85,
        _ => 1.0,
    };

    if source_is_slow_motion {
        let slow_motion_enter = match prev_kind {
            Some(SegmentKind::SlowMotion) => 0.14,
            _ => 0.20,
        };
        let zoom_norm = if config.motion_threshold > 0.0 {
            motion.zoom_score / config.motion_threshold
        } else {
            0.0
        };
        if motion_norm >= slow_motion_enter || zoom_norm >= 0.15 {
            return Ok((Some(SegmentKind::SlowMotion), None));
        }
    }

    if motion_norm >= motion_enter {
        return Ok((Some(SegmentKind::GimbalMove), None));
    }

    let person_confidence =
        detect_person_confidence(detector, frame.bgr, frame.width, frame.height, config)?;
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
        Some(SegmentKind::StaticSubject) => 0.85,
        _ => 1.0,
    };

    if person_norm >= person_enter {
        Ok((Some(SegmentKind::StaticSubject), person_confidence))
    } else {
        Ok((None, person_confidence))
    }
}

fn build_segment(
    input: &Path,
    kind: SegmentKind,
    motion_score: f32,
    zoom_score: f32,
    person_confidence: Option<f32>,
    span: WindowSpan,
    timebase: u32,
) -> Segment {
    let end_seconds = span.start_seconds + span.duration_seconds;

    Segment {
        source_path: input.to_path_buf(),
        start_frame: seconds_to_timeline_frame(span.start_seconds, timebase),
        end_frame: seconds_to_timeline_frame(end_seconds, timebase),
        start_seconds: span.start_seconds,
        end_seconds,
        kind,
        label_id: kind.label_id(),
        motion_score,
        zoom_score,
        person_confidence,
        window_count: 1,
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

    let vf = format!(
        "scale=-2:{}:flags=fast_bilinear,fps={}",
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
    cmd.args(["-hwaccel", "auto"]);
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

fn suppress_child_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn finish_ffmpeg(mut child: std::process::Child, input: &Path) -> AppResult<()> {
    let status = child.wait().map_err(|e| AppError::CommandFailed {
        cmd: "wait ffmpeg".to_string(),
        source: e,
    })?;
    if !status.success() {
        return Err(AppError::CommandNonZero {
            cmd: format!("ffmpeg ({})", input.display()),
            code: status.code().unwrap_or(-1),
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
    duration_seconds: f64,
}

#[derive(Debug, Clone, Copy)]
struct DetectorFrame<'a> {
    bgr: &'a [u8],
    width: usize,
    height: usize,
}
