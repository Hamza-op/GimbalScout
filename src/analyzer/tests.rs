use super::motion::{MotionSampling, estimate_pair_camera_motion};

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
}
