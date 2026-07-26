use super::motion::{
    MotionFeatures, MotionSampling, average_pair_motion_features, estimate_pair_camera_motion,
};
use crate::timeline::MovementType;

#[cfg(feature = "yolo")]
use super::detector::best_person_confidence_2d;
#[cfg(feature = "yolo")]
use ndarray::array;

fn synthetic_motion_sampling() -> MotionSampling {
    MotionSampling::new(96, 64)
}

fn pattern_frame(width: usize, height: usize, shift_x: isize, shift_y: isize) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let sx = (x as isize - shift_x).clamp(0, width as isize - 1) as usize;
            let sy = (y as isize - shift_y).clamp(0, height as isize - 1) as usize;
            let checker = (((sx / 4) + (sy / 4)) % 2) as u8 * 90;
            let grad = ((sx * 3 + sy * 5) % 120) as u8;
            out[y * width + x] = 40 + checker + grad;
        }
    }
    out
}

fn zoom_pattern_frame(width: usize, height: usize, scale: f32) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    for y in 0..height {
        for x in 0..width {
            let src_x = (((x as f32 - cx) / scale) + cx)
                .round()
                .clamp(0.0, width as f32 - 1.0) as usize;
            let src_y = (((y as f32 - cy) / scale) + cy)
                .round()
                .clamp(0.0, height as f32 - 1.0) as usize;
            let checker = (((src_x / 4) + (src_y / 4)) % 2) as u8 * 90;
            let grad = ((src_x * 3 + src_y * 5) % 120) as u8;
            out[y * width + x] = 40 + checker + grad;
        }
    }
    out
}

fn rotated_pattern_frame(width: usize, height: usize, radians: f32) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let cos_r = radians.cos();
    let sin_r = radians.sin();
    for y in 0..height {
        for x in 0..width {
            let rx = x as f32 - cx;
            let ry = y as f32 - cy;
            let src_x = (cos_r * rx + sin_r * ry + cx)
                .round()
                .clamp(0.0, width as f32 - 1.0) as usize;
            let src_y = (-sin_r * rx + cos_r * ry + cy)
                .round()
                .clamp(0.0, height as f32 - 1.0) as usize;
            let checker = (((src_x / 4) + (src_y / 4)) % 2) as u8 * 90;
            let grad = ((src_x * 3 + src_y * 5) % 120) as u8;
            out[y * width + x] = 40 + checker + grad;
        }
    }
    out
}

#[cfg(feature = "yolo")]
#[test]
fn person_score_rows_first_ultralytics_style() {
    let output = array![[0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.9]];
    assert_eq!(best_person_confidence_2d(output.view()), Some(0.9));
}

#[cfg(feature = "yolo")]
#[test]
fn person_score_rows_first_obj_times_class() {
    let output = array![
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.9]
    ];
    let score = best_person_confidence_2d(output.view()).expect("expected Some score");
    assert!((score - 0.72).abs() < 1e-5, "expected ~0.72, got {score}");
}

#[test]
fn dominant_camera_motion_detects_global_translation() {
    let sampling = synthetic_motion_sampling();
    let a = pattern_frame(sampling.thumb_w, sampling.thumb_h, 0, 0);
    let b = pattern_frame(sampling.thumb_w, sampling.thumb_h, 3, 1);

    let score = estimate_pair_camera_motion(&a, &b, &sampling).expect("score");

    assert!(
        score.motion_score > 2.0,
        "expected clear global motion, got {}",
        score.motion_score
    );
}

#[test]
fn dominant_camera_motion_ignores_local_object_motion() {
    let sampling = synthetic_motion_sampling();
    let mut a = pattern_frame(sampling.thumb_w, sampling.thumb_h, 0, 0);
    let mut b = a.clone();

    for y in 18..30 {
        for x in 18..30 {
            a[y * sampling.thumb_w + x] = 10;
        }
    }
    for y in 18..30 {
        for x in 28..40 {
            b[y * sampling.thumb_w + x] = 240;
        }
    }

    let score = estimate_pair_camera_motion(&a, &b, &sampling).expect("score");

    assert!(
        score.motion_score < 1.0,
        "expected local motion to be rejected, got {}",
        score.motion_score
    );
}

#[test]
fn dominant_camera_motion_stays_low_for_static_frame() {
    let sampling = synthetic_motion_sampling();
    let a = pattern_frame(sampling.thumb_w, sampling.thumb_h, 0, 0);

    let score = estimate_pair_camera_motion(&a, &a, &sampling).expect("score");

    assert!(
        score.motion_score < 0.25,
        "expected static frame to stay near zero, got {}",
        score.motion_score
    );
    assert!(
        score.zoom_score < 0.25,
        "expected no zoom in static frame, got {}",
        score.zoom_score
    );
}

#[test]
fn dominant_camera_motion_detects_zoom() {
    let sampling = synthetic_motion_sampling();
    let a = zoom_pattern_frame(sampling.thumb_w, sampling.thumb_h, 1.0);
    let b = zoom_pattern_frame(sampling.thumb_w, sampling.thumb_h, 1.18);

    let score = estimate_pair_camera_motion(&a, &b, &sampling).expect("score");

    assert!(
        score.zoom_score > 1.0,
        "expected zoom score, got {}",
        score.zoom_score
    );
    assert!(
        score.motion_score >= score.zoom_score,
        "combined motion should include zoom priority: motion={}, zoom={}",
        score.motion_score,
        score.zoom_score
    );
    assert_eq!(score.movement_type, MovementType::Zoom);
}

#[test]
fn dominant_camera_motion_detects_rotation() {
    let sampling = synthetic_motion_sampling();
    let a = rotated_pattern_frame(sampling.thumb_w, sampling.thumb_h, 0.0);
    let b = rotated_pattern_frame(sampling.thumb_w, sampling.thumb_h, 0.10);

    let score = estimate_pair_camera_motion(&a, &b, &sampling).expect("score");

    assert!(
        score.motion_score > 1.0,
        "expected rotation to register as camera movement, got {}",
        score.motion_score
    );
    assert_eq!(score.movement_type, MovementType::Roll);
}

#[test]
fn motion_matching_ignores_uniform_exposure_change() {
    let sampling = synthetic_motion_sampling();
    let a = pattern_frame(sampling.thumb_w, sampling.thumb_h, 0, 0);
    let b = a
        .iter()
        .map(|value| value.saturating_add(4))
        .collect::<Vec<_>>();

    let score = estimate_pair_camera_motion(&a, &b, &sampling).expect("score");

    assert!(
        score.motion_score < 0.35,
        "exposure-only change should not look like a camera move, got {}",
        score.motion_score
    );
}

#[test]
fn temporal_smoothness_penalizes_direction_reversals() {
    let smooth = (0..8)
        .map(|_| {
            Some(MotionFeatures {
                motion_score: 2.0,
                translation_x: 2.0,
                confidence: 0.9,
                temporal_smoothness: 1.0,
                ..MotionFeatures::default()
            })
        })
        .collect::<Vec<_>>();
    let jitter = (0..8)
        .map(|i| {
            Some(MotionFeatures {
                motion_score: 2.0,
                translation_x: if i % 2 == 0 { 2.0 } else { -2.0 },
                confidence: 0.9,
                temporal_smoothness: 1.0,
                ..MotionFeatures::default()
            })
        })
        .collect::<Vec<_>>();

    let smooth_score = average_pair_motion_features(&smooth);
    let jitter_score = average_pair_motion_features(&jitter);

    assert!(smooth_score.temporal_smoothness > 0.85);
    assert!(
        jitter_score.temporal_smoothness < 0.55,
        "direction reversals should reduce temporal confidence, got {}",
        jitter_score.temporal_smoothness
    );
}

#[test]
fn motion_reference_scale_is_resolution_invariant() {
    let low = MotionSampling::new(96, 64);
    let high = MotionSampling::new(192, 128);
    let low_reference = low.source_scale * low.thumb_h as f32;
    let high_reference = high.source_scale * high.thumb_h as f32;
    assert!(
        (low_reference - high_reference).abs() < 1e-4,
        "motion score scale should use a fixed reference height"
    );
}

#[test]
fn short_clips_and_clip_tails_get_analysis_windows() {
    assert_eq!(super::analysis_window_starts(10, 18, 4), vec![0]);
    assert_eq!(super::analysis_window_starts(30, 18, 4), vec![0, 4, 8, 12]);
}

#[test]
fn automatic_threshold_is_bounded_by_clip_content() {
    let quiet = super::calculate_dynamic_motion_threshold([0.0, 0.1, 0.2, 0.3].into_iter());
    let continuous_move =
        super::calculate_dynamic_motion_threshold([2.8, 3.0, 3.2, 3.4].into_iter());
    assert!((0.85..=1.10).contains(&quiet));
    assert!((2.35..=2.40).contains(&continuous_move));
}

#[test]
fn person_confidence_requires_temporal_support_for_moderate_hits() {
    let sustained = super::robust_person_confidence([0.58, 0.55, 0.02].into_iter()).expect("score");
    let isolated = super::robust_person_confidence([0.58, 0.02, 0.01].into_iter()).expect("score");
    assert!(sustained > 0.50);
    assert!(isolated < sustained);
}

#[test]
#[ignore = "integration test requires the bundled FFmpeg executable"]
fn end_to_end_synthetic_camera_move_is_detected() {
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let ffmpeg = std::path::PathBuf::from("assets/ffmpeg.exe");
    if !ffmpeg.exists() {
        return;
    }
    let root = std::env::temp_dir().join(format!("video-tool-motion-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&root);
    let raw = root.join("move.raw");
    let encoded = root.join("move.mp4");
    let width = 640usize;
    let height = 360usize;
    let frame_count = 36usize;
    let mut bytes = Vec::with_capacity(width * height * 3 * frame_count);
    for frame_index in 0..frame_count {
        let shift = frame_index as isize;
        for y in 0..height {
            for x in 0..width {
                let sx = (x as isize - shift).clamp(0, width as isize - 1) as usize;
                let sy = y;
                let value = 35
                    + ((((sx / 12) + (sy / 12)) % 2) * 120) as u8
                    + ((sx * 3 + sy * 5) % 80) as u8;
                bytes.extend_from_slice(&[
                    value,
                    value.saturating_add(8),
                    value.saturating_add(16),
                ]);
            }
        }
    }
    std::fs::write(&raw, bytes).expect("write synthetic source");
    let status = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "640x360",
            "-r",
            "12",
            "-i",
        ])
        .arg(&raw)
        .args(["-c:v", "ffv1", "-pix_fmt", "yuv444p", "-y"])
        .arg(&encoded)
        .status()
        .expect("run ffmpeg encoder");
    assert!(status.success());

    let probe = crate::media::ProbeInfo {
        source_path: encoded.clone(),
        width: width as u32,
        height: height as u32,
        duration_seconds: 3.0,
        duration_frames: frame_count as u64,
        fps_num: 12,
        fps_den: 1,
        timebase: 12,
        ntsc: false,
        slow_motion: false,
        capture_fps: None,
        format_fps: None,
    };
    let config = crate::config::AnalysisConfig {
        ffmpeg_bin: ffmpeg,
        ffprobe_bin: std::path::PathBuf::from("ffprobe"),
        yolo_model: None,
        enable_yolo: false,
        config_fingerprint: "integration".to_string(),
        analysis_height: 360,
        analysis_fps: 12.0,
        window_seconds: 0.75,
        motion_threshold: 0.0,
        person_confidence: 0.42,
        yolo_intra_threads: 1,
        ffmpeg_threads: 1,
        buf_frames: 2,
        acceleration: Default::default(),
    };
    let mut worker = super::AnalyzerWorker::default();
    let segments = worker
        .analyze_file(&encoded, &probe, &config, &Arc::new(AtomicBool::new(false)))
        .expect("analyze synthetic source");
    assert!(
        segments
            .iter()
            .any(|segment| segment.kind == crate::timeline::SegmentKind::GimbalMove),
        "expected a camera move, got {segments:?}"
    );
    let merged = crate::timeline::merge_segments(segments);
    let selected = crate::timeline::select_source_segments(
        probe.duration_seconds,
        merged,
        &crate::timeline::SensitivityConfig::default(),
    );
    assert!(
        selected
            .iter()
            .any(|segment| segment.kind == crate::timeline::SegmentKind::GimbalMove),
        "expected the camera move to survive editorial filtering: {selected:?}"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(feature = "yolo")]
#[test]
#[ignore = "model smoke test is opt-in because it runs ONNX inference"]
fn bundled_yolo_model_loads_and_returns_a_person_score() {
    let model = std::path::PathBuf::from("assets/yolo.onnx");
    if !model.exists() {
        return;
    }
    let config = crate::config::AnalysisConfig {
        ffmpeg_bin: std::path::PathBuf::from("ffmpeg"),
        ffprobe_bin: std::path::PathBuf::from("ffprobe"),
        yolo_model: Some(model),
        enable_yolo: true,
        config_fingerprint: "model-smoke".to_string(),
        analysis_height: 360,
        analysis_fps: 12.0,
        window_seconds: 1.0,
        motion_threshold: 0.0,
        person_confidence: 0.42,
        yolo_intra_threads: 1,
        ffmpeg_threads: 1,
        buf_frames: 1,
        acceleration: Default::default(),
    };
    let mut detector = super::detector::YoloDetector::from_config(&config)
        .expect("load bundled model")
        .expect("model should be available");
    let frame = vec![128u8; 640 * 360 * 3];
    let score = detector
        .detect_person_confidence(&frame, 640, 360)
        .expect("run bundled model");
    assert!(score.is_some(), "model output should be parsed");
}
