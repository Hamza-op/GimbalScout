use crate::timeline::MovementType;

const MOTION_THUMB_HEIGHT: usize = 144;
const MOTION_THUMB_WIDTH_MAX: usize = 288;
/// Motion scores are expressed in pixels at this reference height. This keeps
/// thresholds stable when the user changes the analysis-resolution preset.
const MOTION_SCORE_REFERENCE_HEIGHT: f32 = 360.0;
const MOTION_GRID_X: usize = 11;
const MOTION_GRID_Y: usize = 7;
const MOTION_PATCH_RADIUS: isize = 5;
const MOTION_SEARCH_RADIUS: isize = 10;
const MOTION_MIN_TEXTURE: f32 = 12.0;
const MOTION_MIN_INLIERS: usize = 10;
const MOTION_MODEL_INLIER_TOLERANCE: f32 = 1.5;
const MOTION_MAX_RANSAC_SEEDS: usize = 16;
const MOTION_SCORE_BASELINE_FPS: f32 = 6.0;
const RANSAC_MAX_ITERATIONS: usize = 200;
const SEARCH_STEPS: [isize; 3] = [4, 2, 1];

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MotionFeatures {
    pub(crate) motion_score: f32,
    pub(crate) zoom_score: f32,
    pub(crate) rotation_score: f32,
    pub(crate) shear_score: f32,
    pub(crate) translation_x: f32,
    pub(crate) translation_y: f32,
    pub(crate) zoom_velocity: f32,
    pub(crate) rotation_velocity: f32,
    pub(crate) confidence: f32,
    pub(crate) temporal_smoothness: f32,
    pub(crate) movement_type: MovementType,
}

#[derive(Debug, Clone, Copy)]
struct MotionVector {
    dx: f32,
    dy: f32,
    cx: usize,
    cy: usize,
    match_sad: f32,
}

#[derive(Debug, Clone, Copy)]
enum CameraMotionModel {
    Affine {
        tx: f32,
        ty: f32,
        ax: f32,
        bx: f32,
        cx: f32,
        dy: f32,
        mean_residual: f32,
        inliers: usize,
    },
}

impl CameraMotionModel {
    fn mean_residual(&self) -> f32 {
        let CameraMotionModel::Affine { mean_residual, .. } = *self;
        mean_residual
    }

    fn inliers(&self) -> usize {
        let CameraMotionModel::Affine { inliers, .. } = *self;
        inliers
    }

    fn set_inliers(&mut self, val: usize) {
        let CameraMotionModel::Affine { inliers, .. } = self;
        *inliers = val;
    }

    fn set_mean_residual(&mut self, val: f32) {
        let CameraMotionModel::Affine { mean_residual, .. } = self;
        *mean_residual = val;
    }
}

#[derive(Debug, Clone, Copy)]
struct AffineDecomposition {
    uniform_scale: f32,
    rotation_radians: f32,
    shear: f32,
    anisotropy: f32,
}

pub(crate) struct MotionSampling {
    pub(crate) thumb_w: usize,
    pub(crate) thumb_h: usize,
    pub(crate) source_scale: f32,
    source_indices: Vec<usize>,
    patch_centers: Vec<(usize, usize)>,
}

impl MotionSampling {
    pub(crate) fn new(src_w: usize, src_h: usize) -> Self {
        let source_w = src_w.max(1);
        let source_h = src_h.max(1);
        // Preserve aspect ratio while respecting the thumbnail budget. The
        // previous implementation always forced height=144, squeezing
        // ultrawide footage into a 2:1 image and creating false motion.
        let mut thumb_h = source_h.min(MOTION_THUMB_HEIGHT);
        let mut thumb_w = ((source_w as f64 * thumb_h as f64 / source_h as f64)
            .round()
            .max(1.0)) as usize;
        if thumb_w > MOTION_THUMB_WIDTH_MAX {
            thumb_w = MOTION_THUMB_WIDTH_MAX;
            thumb_h = ((source_h as f64 * thumb_w as f64 / source_w as f64)
                .round()
                .max(1.0)) as usize;
        }
        thumb_w = thumb_w.max(24.min(source_w));
        thumb_h = thumb_h.max(24.min(source_h));
        let source_scale = MOTION_SCORE_REFERENCE_HEIGHT / thumb_h as f32;

        let mut source_indices = Vec::with_capacity(thumb_w * thumb_h);
        for y in 0..thumb_h {
            let src_y = ((y * source_h) / thumb_h).min(source_h.saturating_sub(1));
            for x in 0..thumb_w {
                let src_x = ((x * source_w) / thumb_w).min(source_w.saturating_sub(1));
                source_indices.push((src_y * source_w + src_x) * 3);
            }
        }

        let margin = (MOTION_SEARCH_RADIUS + MOTION_PATCH_RADIUS + 1).max(1) as usize;
        let patch_centers = build_patch_centers(thumb_w, thumb_h, margin);

        Self {
            thumb_w,
            thumb_h,
            source_scale,
            source_indices,
            patch_centers,
        }
    }

    pub(crate) fn pixel_count(&self) -> usize {
        self.thumb_w * self.thumb_h
    }
}

pub(crate) fn sample_motion_frame_into(src_bgr: &[u8], out: &mut [u8], s: &MotionSampling) {
    debug_assert_eq!(out.len(), s.pixel_count());
    for (dst, &src_idx) in out.iter_mut().zip(s.source_indices.iter()) {
        let luma = (src_bgr[src_idx] as u32 * 29
            + src_bgr[src_idx + 1] as u32 * 150
            + src_bgr[src_idx + 2] as u32 * 77)
            >> 8;
        *dst = luma as u8;
    }
}

pub(crate) fn sample_motion_gray_into(src_gray: &[u8], out: &mut [u8], s: &MotionSampling) {
    debug_assert_eq!(out.len(), s.pixel_count());
    for (dst, &src_idx) in out.iter_mut().zip(s.source_indices.iter()) {
        *dst = src_gray[src_idx / 3];
    }
}

#[cfg(test)]
pub(crate) fn average_pair_motion_features(features: &[Option<MotionFeatures>]) -> MotionFeatures {
    average_pair_motion_features_at_fps(features, MOTION_SCORE_BASELINE_FPS)
}

pub(crate) fn average_pair_motion_features_at_fps(
    features: &[Option<MotionFeatures>],
    analysis_fps: f32,
) -> MotionFeatures {
    let valid = features.iter().flatten().copied().collect::<Vec<_>>();
    let pairs = valid.len();

    if pairs == 0 {
        MotionFeatures::default()
    } else {
        let fps_scale = if analysis_fps.is_finite() && analysis_fps > 0.0 {
            analysis_fps / MOTION_SCORE_BASELINE_FPS
        } else {
            1.0
        };
        let scaled = valid
            .iter()
            .map(|feature| MotionFeatures {
                motion_score: feature.motion_score * fps_scale,
                zoom_score: feature.zoom_score * fps_scale,
                rotation_score: feature.rotation_score * fps_scale,
                shear_score: feature.shear_score * fps_scale,
                translation_x: feature.translation_x * fps_scale,
                translation_y: feature.translation_y * fps_scale,
                zoom_velocity: feature.zoom_velocity * fps_scale,
                rotation_velocity: feature.rotation_velocity * fps_scale,
                ..*feature
            })
            .collect::<Vec<_>>();
        let temporal_smoothness = calculate_temporal_smoothness(&scaled);
        MotionFeatures {
            // A trimmed mean ignores an isolated scene cut, flash, or single
            // tracking failure while preserving sustained camera movement.
            motion_score: trimmed_mean(scaled.iter().map(|f| f.motion_score)),
            zoom_score: trimmed_mean(scaled.iter().map(|f| f.zoom_score)),
            rotation_score: trimmed_mean(scaled.iter().map(|f| f.rotation_score)),
            shear_score: trimmed_mean(scaled.iter().map(|f| f.shear_score)),
            translation_x: scaled.iter().map(|f| f.translation_x).sum::<f32>() / pairs as f32,
            translation_y: scaled.iter().map(|f| f.translation_y).sum::<f32>() / pairs as f32,
            zoom_velocity: scaled.iter().map(|f| f.zoom_velocity).sum::<f32>() / pairs as f32,
            rotation_velocity: scaled.iter().map(|f| f.rotation_velocity).sum::<f32>()
                / pairs as f32,
            confidence: trimmed_mean(scaled.iter().map(|f| f.confidence)),
            temporal_smoothness,
            movement_type: dominant_movement_type(features),
        }
    }
}

fn trimmed_mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut values = values.filter(|v| v.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let trim = if values.len() >= 8 {
        values.len() / 10
    } else {
        0
    };
    let kept = &values[trim..values.len() - trim];
    kept.iter().sum::<f32>() / kept.len() as f32
}

fn calculate_temporal_smoothness(features: &[MotionFeatures]) -> f32 {
    if features.is_empty() {
        return 0.0;
    }

    let velocities = features
        .iter()
        .map(|f| {
            [
                f.translation_x,
                f.translation_y,
                f.zoom_velocity * 1.15,
                f.rotation_velocity * 0.85,
            ]
        })
        .collect::<Vec<_>>();
    let magnitude = |v: [f32; 4]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let magnitudes = velocities
        .iter()
        .copied()
        .map(magnitude)
        .collect::<Vec<_>>();
    let total_energy = magnitudes.iter().sum::<f32>();
    if total_energy <= 0.05 {
        return 0.0;
    }

    let mut resultant = [0.0f32; 4];
    for velocity in &velocities {
        for axis in 0..4 {
            resultant[axis] += velocity[axis];
        }
    }
    let direction_consistency = (magnitude(resultant) / total_energy).clamp(0.0, 1.0);

    let mean_energy = total_energy / magnitudes.len() as f32;
    let mut normalized_change = 0.0f32;
    let mut transitions = 0usize;
    for pair in velocities.windows(2) {
        let delta = [
            pair[1][0] - pair[0][0],
            pair[1][1] - pair[0][1],
            pair[1][2] - pair[0][2],
            pair[1][3] - pair[0][3],
        ];
        normalized_change += magnitude(delta) / (mean_energy + 0.35);
        transitions += 1;
    }
    let mean_change = normalized_change / transitions.max(1) as f32;
    let continuity = (1.0 / (1.0 + mean_change * 0.8)).clamp(0.0, 1.0);

    let active = magnitudes
        .iter()
        .filter(|&&energy| energy >= (mean_energy * 0.20).max(0.20))
        .count();
    let persistence = (active as f32 / magnitudes.len() as f32 / 0.55).clamp(0.0, 1.0);

    (direction_consistency * 0.55 + continuity * 0.30 + persistence * 0.15).clamp(0.0, 1.0)
}

fn shi_tomasi_score(frame: &[u8], width: usize, height: usize, cx: usize, cy: usize) -> f32 {
    let mut sum_xx = 0.0f32;
    let mut sum_yy = 0.0f32;
    let mut sum_xy = 0.0f32;

    for py in -2..=2 {
        let y = cy as isize + py;
        if y <= 0 || y >= height as isize - 1 {
            continue;
        }
        let y_idx = y as usize * width;
        for px in -2..=2 {
            let x = cx as isize + px;
            if x <= 0 || x >= width as isize - 1 {
                continue;
            }
            let idx = y_idx + x as usize;

            // Central differences
            let ix = (frame[idx + 1] as f32 - frame[idx - 1] as f32) * 0.5;
            let iy = (frame[idx + width] as f32 - frame[idx - width] as f32) * 0.5;

            sum_xx += ix * ix;
            sum_yy += iy * iy;
            sum_xy += ix * iy;
        }
    }

    let trace = sum_xx + sum_yy;
    let det = sum_xx * sum_yy - sum_xy * sum_xy;
    let val = trace * trace - 4.0 * det;
    (trace - val.max(0.0).sqrt()) * 0.5
}

pub(crate) fn estimate_pair_camera_motion(
    prev: &[u8],
    next: &[u8],
    s: &MotionSampling,
) -> Option<MotionFeatures> {
    if prev.len() != next.len() || prev.is_empty() {
        return None;
    }

    let mut vectors = Vec::with_capacity(s.patch_centers.len());
    for &(cx_init, cy_init) in &s.patch_centers {
        // Dynamic Shi-Tomasi corner relocation locally
        let mut best_cx = cx_init;
        let mut best_cy = cy_init;
        let mut best_score = -1.0f32;

        for dy in -3..=3 {
            let cy_cand = (cy_init as isize + dy) as usize;
            for dx in -3..=3 {
                let cx_cand = (cx_init as isize + dx) as usize;
                let score = shi_tomasi_score(prev, s.thumb_w, s.thumb_h, cx_cand, cy_cand);
                if score > best_score {
                    best_score = score;
                    best_cx = cx_cand;
                    best_cy = cy_cand;
                }
            }
        }

        let cx = best_cx;
        let cy = best_cy;

        let texture = patch_texture(prev, s.thumb_w, s.thumb_h, cx, cy, MOTION_PATCH_RADIUS);
        if texture < MOTION_MIN_TEXTURE {
            continue;
        }
        if let Some((dx, dy, match_sad)) =
            best_patch_shift(prev, next, s.thumb_w, s.thumb_h, cx, cy)
        {
            vectors.push(MotionVector {
                dx,
                dy,
                cx,
                cy,
                match_sad,
            });
        }
    }

    let observed_support = vectors.len() as f32 / s.patch_centers.len().max(1) as f32;
    let fallback_zoom = estimate_scale_zoom_score(prev, next, s) * observed_support;
    let Some(model) = fit_camera_motion_model(&vectors, s) else {
        return Some(MotionFeatures {
            motion_score: fallback_zoom * 1.15,
            zoom_score: fallback_zoom,
            rotation_score: 0.0,
            shear_score: 0.0,
            translation_x: 0.0,
            translation_y: 0.0,
            zoom_velocity: fallback_zoom,
            rotation_velocity: 0.0,
            confidence: (observed_support * 0.5).clamp(0.0, 1.0),
            temporal_smoothness: 1.0,
            movement_type: MovementType::Zoom,
        });
    };

    let decomposition = decompose_any_motion_model(model);
    let inlier_ratio = model.inliers() as f32 / vectors.len().max(1) as f32;
    let frame_coverage = (model.inliers() as f32 / s.patch_centers.len().max(1) as f32).sqrt();
    let support = inlier_ratio * (0.55 + 0.45 * frame_coverage);
    let coherence =
        (1.0 - model.mean_residual() / (MOTION_SEARCH_RADIUS as f32 + 1.0)).clamp(0.0, 1.0);

    let CameraMotionModel::Affine { tx, ty, .. } = model;

    let translation_score = (tx.powi(2) + ty.powi(2)).sqrt() * s.source_scale * support * coherence;
    let zoom_edge_radius = (s.thumb_w.min(s.thumb_h) as f32) * 0.5;
    let zoom_consistency = (1.0
        - decomposition.anisotropy / (decomposition.uniform_scale.abs() + 0.08))
        .clamp(0.0, 1.0);
    let model_zoom_score = decomposition.uniform_scale.abs()
        * zoom_edge_radius
        * s.source_scale
        * support
        * coherence
        * zoom_consistency;
    let rotation_score = decomposition.rotation_radians.abs()
        * zoom_edge_radius
        * s.source_scale
        * support
        * coherence;
    let shear_score =
        decomposition.shear.abs() * zoom_edge_radius * s.source_scale * support * coherence * 0.5;
    let zoom_score = model_zoom_score.max(fallback_zoom);
    let mean_match_sad =
        vectors.iter().map(|vector| vector.match_sad).sum::<f32>() / vectors.len().max(1) as f32;
    let match_quality = (1.0 - mean_match_sad / 48.0).clamp(0.0, 1.0);
    let confidence =
        (inlier_ratio * coherence * (0.55 + 0.45 * frame_coverage) * match_quality).clamp(0.0, 1.0);
    let movement_type =
        classify_movement_type(translation_score, zoom_score, rotation_score, shear_score);
    Some(MotionFeatures {
        motion_score: translation_score
            .max(zoom_score * 1.15)
            .max(rotation_score * 0.85)
            .max(shear_score),
        zoom_score,
        rotation_score,
        shear_score,
        translation_x: tx * s.source_scale,
        translation_y: ty * s.source_scale,
        zoom_velocity: decomposition.uniform_scale * zoom_edge_radius * s.source_scale,
        rotation_velocity: decomposition.rotation_radians * zoom_edge_radius * s.source_scale,
        confidence,
        temporal_smoothness: 1.0,
        movement_type,
    })
}

fn dominant_movement_type(features: &[Option<MotionFeatures>]) -> MovementType {
    let mut totals = [0.0f32; 4];
    let mut weights = [0.0f32; 4];
    for feature in features.iter().flatten() {
        let index = match feature.movement_type {
            MovementType::Zoom => 0,
            MovementType::Roll => 1,
            MovementType::Complex => 2,
            MovementType::PanTilt => 3,
            MovementType::Subject | MovementType::SlowMotion => continue,
        };
        let weight = (0.35 + feature.confidence.clamp(0.0, 1.0) * 0.65).max(0.05);
        totals[index] += movement_type_score(*feature) * weight;
        weights[index] += weight;
    }
    let mut best = (MovementType::PanTilt, 0.0f32);
    for (index, movement) in [
        MovementType::Zoom,
        MovementType::Roll,
        MovementType::Complex,
        MovementType::PanTilt,
    ]
    .into_iter()
    .enumerate()
    {
        let score = totals[index] / weights[index].max(1e-6);
        if score > best.1 {
            best = (movement, score);
        }
    }
    best.0
}

fn movement_type_score(feature: MotionFeatures) -> f32 {
    match feature.movement_type {
        MovementType::Zoom => feature.zoom_score,
        MovementType::Roll => feature.rotation_score,
        MovementType::Complex => feature.shear_score.max(feature.rotation_score),
        MovementType::PanTilt => feature.motion_score,
        MovementType::Subject | MovementType::SlowMotion => feature.motion_score,
    }
}

fn classify_movement_type(
    translation_score: f32,
    zoom_score: f32,
    rotation_score: f32,
    shear_score: f32,
) -> MovementType {
    let strongest = translation_score
        .max(zoom_score)
        .max(rotation_score)
        .max(shear_score);
    if strongest <= 0.0 {
        return MovementType::PanTilt;
    }
    if zoom_score >= strongest * 0.82 {
        MovementType::Zoom
    } else if rotation_score >= strongest * 0.65 && rotation_score >= shear_score * 0.75 {
        MovementType::Roll
    } else if shear_score >= strongest * 0.70 {
        MovementType::Complex
    } else {
        MovementType::PanTilt
    }
}

pub(crate) fn scaled_width_even(src_w: u32, src_h: u32, target_h: u32) -> u32 {
    if src_h == 0 || target_h == 0 {
        return 0;
    }
    // Match FFmpeg's `scale=-2:H` choice: preserve aspect ratio, round down
    // to a positive even width. An upward round can disagree by two pixels
    // and corrupt the headerless rawvideo framing.
    let numerator = u64::from(src_w).saturating_mul(u64::from(target_h));
    let mut width = (numerator / u64::from(src_h)).max(2) as u32;
    if width % 2 == 1 {
        width = width.saturating_sub(1).max(2);
    }
    width
}

pub(crate) fn seconds_to_timeline_frame(seconds: f64, fps_num: u32, fps_den: u32) -> u64 {
    if fps_den == 0 {
        return 0;
    }
    let v = seconds * f64::from(fps_num) / f64::from(fps_den);
    if v.is_finite() && v >= 0.0 {
        v.round() as u64
    } else {
        0
    }
}

fn build_patch_centers(width: usize, height: usize, margin: usize) -> Vec<(usize, usize)> {
    let safe_x0 = margin.min(width.saturating_sub(1));
    let safe_x1 = width.saturating_sub(margin + 1).max(safe_x0);
    let safe_y0 = margin.min(height.saturating_sub(1));
    let safe_y1 = height.saturating_sub(margin + 1).max(safe_y0);

    let mut out = Vec::with_capacity(MOTION_GRID_X * MOTION_GRID_Y);
    for gy in 0..MOTION_GRID_Y {
        let y = interpolate_grid(safe_y0, safe_y1, gy, MOTION_GRID_Y);
        for gx in 0..MOTION_GRID_X {
            let x = interpolate_grid(safe_x0, safe_x1, gx, MOTION_GRID_X);
            out.push((x, y));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn interpolate_grid(start: usize, end: usize, index: usize, count: usize) -> usize {
    if count <= 1 || start >= end {
        return start;
    }
    start + ((end - start) * index) / (count - 1)
}

fn fit_camera_motion_model(
    vectors: &[MotionVector],
    s: &MotionSampling,
) -> Option<CameraMotionModel> {
    if vectors.is_empty() {
        return None;
    }

    let center_x = (s.thumb_w as f32 - 1.0) * 0.5;
    let center_y = (s.thumb_h as f32 - 1.0) * 0.5;

    // An affine model captures pan, tilt, zoom, roll, and modest perspective
    // change while resisting the parallax and moving-subject overfit that a
    // full homography can produce in real wedding footage.
    let mut best = solve_camera_motion_model(vectors, center_x, center_y)
        .and_then(|model| score_camera_motion_model(model, vectors, center_x, center_y));

    let seeds = select_motion_seed_vectors(vectors, center_x, center_y);
    if seeds.len() >= 3 {
        let mut iterations = 0usize;
        'ransac_affine: for i in 0..seeds.len() - 2 {
            for j in i + 1..seeds.len() - 1 {
                for k in j + 1..seeds.len() {
                    iterations += 1;
                    if iterations > RANSAC_MAX_ITERATIONS {
                        break 'ransac_affine;
                    }
                    let sample = [seeds[i], seeds[j], seeds[k]];
                    if sample_triangle_area(sample[0], sample[1], sample[2]) < 10.0 {
                        continue;
                    }
                    let Some(model) = solve_camera_motion_model(&sample, center_x, center_y) else {
                        continue;
                    };
                    let Some(candidate) =
                        score_camera_motion_model(model, vectors, center_x, center_y)
                    else {
                        continue;
                    };
                    let is_better = match best {
                        Some((best_model, best_inliers, best_mean_residual)) => {
                            candidate.1 > best_inliers
                                || (candidate.1 == best_inliers
                                    && candidate.2 + 1e-4 < best_mean_residual)
                                || (candidate.1 == best_inliers
                                    && (candidate.2 - best_mean_residual).abs() < 1e-4
                                    && candidate.0.mean_residual() < best_model.mean_residual())
                        }
                        None => true,
                    };
                    if is_better {
                        best = Some(candidate);
                    }
                }
            }
        }
    }

    let (mut best_model, best_inliers, _) = best?;
    if best_inliers < MOTION_MIN_INLIERS {
        return None;
    }

    let mut inlier_vectors = vectors
        .iter()
        .copied()
        .filter(|vector| {
            camera_motion_residual(*vector, center_x, center_y, best_model)
                <= MOTION_MODEL_INLIER_TOLERANCE
        })
        .collect::<Vec<_>>();

    if inlier_vectors.len() < MOTION_MIN_INLIERS {
        return None;
    }

    let min_x = inlier_vectors.iter().map(|v| v.cx).min().unwrap_or(0);
    let max_x = inlier_vectors.iter().map(|v| v.cx).max().unwrap_or(0);
    let min_y = inlier_vectors.iter().map(|v| v.cy).min().unwrap_or(0);
    let max_y = inlier_vectors.iter().map(|v| v.cy).max().unwrap_or(0);

    let spread_x = max_x.saturating_sub(min_x);
    let spread_y = max_y.saturating_sub(min_y);

    if (spread_x as f32) < (s.thumb_w as f32 * 0.25)
        || (spread_y as f32) < (s.thumb_h as f32 * 0.25)
    {
        return None;
    }

    let refined =
        solve_camera_motion_model(&inlier_vectors, center_x, center_y).unwrap_or(best_model);
    inlier_vectors.retain(|vector| {
        camera_motion_residual(*vector, center_x, center_y, refined)
            <= MOTION_MODEL_INLIER_TOLERANCE
    });
    if inlier_vectors.len() < MOTION_MIN_INLIERS {
        return None;
    }

    let mean_residual = inlier_vectors
        .iter()
        .map(|vector| camera_motion_residual(*vector, center_x, center_y, refined))
        .sum::<f32>()
        / inlier_vectors.len().max(1) as f32;

    best_model = refined;
    best_model.set_inliers(inlier_vectors.len());
    best_model.set_mean_residual(mean_residual);

    Some(best_model)
}

fn solve_camera_motion_model(
    vectors: &[MotionVector],
    center_x: f32,
    center_y: f32,
) -> Option<CameraMotionModel> {
    if vectors.is_empty() {
        return None;
    }

    let mut m = [[0.0f32; 6]; 6];
    let mut rhs = [0.0f32; 6];

    for vector in vectors {
        let rx = vector.cx as f32 - center_x;
        let ry = vector.cy as f32 - center_y;
        let rows = [
            ([1.0, 0.0, rx, ry, 0.0, 0.0], vector.dx),
            ([0.0, 1.0, 0.0, 0.0, rx, ry], vector.dy),
        ];
        for (row, target) in rows {
            for i in 0..6 {
                rhs[i] += row[i] * target;
                for j in 0..6 {
                    m[i][j] += row[i] * row[j];
                }
            }
        }
    }

    let solution = solve_linear_system_6(m, rhs)?;
    Some(CameraMotionModel::Affine {
        tx: solution[0],
        ty: solution[1],
        ax: solution[2],
        bx: solution[3],
        cx: solution[4],
        dy: solution[5],
        mean_residual: f32::MAX,
        inliers: 0,
    })
}

fn score_camera_motion_model(
    model: CameraMotionModel,
    vectors: &[MotionVector],
    center_x: f32,
    center_y: f32,
) -> Option<(CameraMotionModel, usize, f32)> {
    let mut inliers = 0usize;
    let mut residual_sum = 0.0f32;
    for vector in vectors {
        let residual = camera_motion_residual(*vector, center_x, center_y, model);
        if residual <= MOTION_MODEL_INLIER_TOLERANCE {
            inliers += 1;
            residual_sum += residual;
        }
    }
    if inliers == 0 {
        return None;
    }
    Some((model, inliers, residual_sum / inliers as f32))
}

fn select_motion_seed_vectors(
    vectors: &[MotionVector],
    center_x: f32,
    center_y: f32,
) -> Vec<MotionVector> {
    let mut seeds = Vec::new();
    for quadrant in 0..4 {
        let candidate = vectors
            .iter()
            .copied()
            .filter(|vector| motion_quadrant(*vector, center_x, center_y) == quadrant)
            .max_by(|a, b| {
                motion_radius_sq(*a, center_x, center_y)
                    .partial_cmp(&motion_radius_sq(*b, center_x, center_y))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(vector) = candidate {
            seeds.push(vector);
        }
    }

    let mut ranked = vectors.to_vec();
    ranked.sort_by(|a, b| {
        motion_radius_sq(*b, center_x, center_y)
            .partial_cmp(&motion_radius_sq(*a, center_x, center_y))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for vector in ranked {
        if seeds
            .iter()
            .any(|seed| seed.cx == vector.cx && seed.cy == vector.cy)
        {
            continue;
        }
        seeds.push(vector);
        if seeds.len() >= MOTION_MAX_RANSAC_SEEDS {
            break;
        }
    }
    seeds
}

fn motion_quadrant(vector: MotionVector, center_x: f32, center_y: f32) -> usize {
    let right = vector.cx as f32 >= center_x;
    let bottom = vector.cy as f32 >= center_y;
    match (right, bottom) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

fn motion_radius_sq(vector: MotionVector, center_x: f32, center_y: f32) -> f32 {
    let rx = vector.cx as f32 - center_x;
    let ry = vector.cy as f32 - center_y;
    rx * rx + ry * ry
}

fn sample_triangle_area(a: MotionVector, b: MotionVector, c: MotionVector) -> f32 {
    let abx = b.cx as f32 - a.cx as f32;
    let aby = b.cy as f32 - a.cy as f32;
    let acx = c.cx as f32 - a.cx as f32;
    let acy = c.cy as f32 - a.cy as f32;
    (abx * acy - aby * acx).abs() * 0.5
}

fn decompose_any_motion_model(model: CameraMotionModel) -> AffineDecomposition {
    let CameraMotionModel::Affine { ax, bx, cx, dy, .. } = model;
    let m00 = 1.0 + ax;
    let m01 = bx;
    let m10 = cx;
    let m11 = 1.0 + dy;

    let scale_x = (m00 * m00 + m10 * m10).sqrt().max(1e-4);
    let rot_cos = m00 / scale_x;
    let rot_sin = m10 / scale_x;
    let shear_projection = rot_cos * m01 + rot_sin * m11;
    let ortho_y_x = m01 - shear_projection * rot_cos;
    let ortho_y_y = m11 - shear_projection * rot_sin;
    let scale_y = (ortho_y_x * ortho_y_x + ortho_y_y * ortho_y_y)
        .sqrt()
        .max(1e-4);
    let shear = shear_projection / scale_y;

    AffineDecomposition {
        uniform_scale: ((scale_x + scale_y) * 0.5) - 1.0,
        rotation_radians: rot_sin.atan2(rot_cos),
        shear,
        anisotropy: (scale_x - scale_y).abs(),
    }
}

fn camera_motion_residual(
    vector: MotionVector,
    center_x: f32,
    center_y: f32,
    model: CameraMotionModel,
) -> f32 {
    let rx = vector.cx as f32 - center_x;
    let ry = vector.cy as f32 - center_y;

    let CameraMotionModel::Affine {
        tx,
        ty,
        ax,
        bx,
        cx,
        dy,
        ..
    } = model;
    let predicted_dx = tx + ax * rx + bx * ry;
    let predicted_dy = ty + cx * rx + dy * ry;
    ((vector.dx - predicted_dx).powi(2) + (vector.dy - predicted_dy).powi(2)).sqrt()
}

fn solve_linear_system_6(mut m: [[f32; 6]; 6], mut rhs: [f32; 6]) -> Option<[f32; 6]> {
    let mut max_abs_val = 0.0f32;
    for row in &m {
        for cell in row {
            let val = cell.abs();
            if val > max_abs_val {
                max_abs_val = val;
            }
        }
    }
    let epsilon = (1e-5 * max_abs_val).max(1e-12);

    for pivot in 0..6 {
        let mut best_row = pivot;
        let mut best_val = m[pivot][pivot].abs();
        let mut row = pivot + 1;
        while row < 6 {
            let candidate = m[row][pivot].abs();
            if candidate > best_val {
                best_val = candidate;
                best_row = row;
            }
            row += 1;
        }
        if best_val <= epsilon {
            return None;
        }
        if best_row != pivot {
            m.swap(pivot, best_row);
            rhs.swap(pivot, best_row);
        }

        let inv_pivot = 1.0 / m[pivot][pivot];
        let mut col = pivot;
        while col < 6 {
            m[pivot][col] *= inv_pivot;
            col += 1;
        }
        rhs[pivot] *= inv_pivot;

        let mut row = 0usize;
        while row < 6 {
            if row == pivot {
                row += 1;
                continue;
            }
            let factor = m[row][pivot];
            if factor.abs() < 1e-6 {
                row += 1;
                continue;
            }
            let mut col = pivot;
            while col < 6 {
                m[row][col] -= factor * m[pivot][col];
                col += 1;
            }
            rhs[row] -= factor * rhs[pivot];
            row += 1;
        }
    }

    Some(rhs)
}

fn estimate_scale_zoom_score(prev: &[u8], next: &[u8], s: &MotionSampling) -> f32 {
    const CANDIDATE_SCALES: [f32; 12] = [
        0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12,
    ];

    let baseline = zoom_scale_sad(prev, next, s, 1.0);
    if baseline <= 0.0 {
        return 0.0;
    }

    let mut best_scale = 1.0f32;
    let mut best_sad = baseline;
    for scale in CANDIDATE_SCALES {
        let sad = zoom_scale_sad(prev, next, s, scale);
        if sad < best_sad {
            best_sad = sad;
            best_scale = scale;
        }
    }

    if best_scale == 1.0 {
        return 0.0;
    }

    let improvement = ((baseline - best_sad) / baseline).max(0.0);
    if improvement < 0.08 {
        return 0.0;
    }

    let scale_delta = (best_scale - 1.0).abs();
    scale_delta * s.thumb_w.min(s.thumb_h) as f32 * improvement
}

fn zoom_scale_sad(prev: &[u8], next: &[u8], s: &MotionSampling, scale: f32) -> f32 {
    let center_x = (s.thumb_w as f32 - 1.0) * 0.5;
    let center_y = (s.thumb_h as f32 - 1.0) * 0.5;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0usize;

    for y in (4..s.thumb_h.saturating_sub(4)).step_by(3) {
        for x in (4..s.thumb_w.saturating_sub(4)).step_by(3) {
            let src_x = (((x as f32 - center_x) / scale) + center_x).round() as isize;
            let src_y = (((y as f32 - center_y) / scale) + center_y).round() as isize;
            if src_x < 0 || src_y < 0 || src_x >= s.thumb_w as isize || src_y >= s.thumb_h as isize
            {
                continue;
            }

            sum_a += prev[src_y as usize * s.thumb_w + src_x as usize] as f32;
            sum_b += next[y * s.thumb_w + x] as f32;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    let mean_a = sum_a / count as f32;
    let mean_b = sum_b / count as f32;
    let mut sad = 0.0f32;
    for y in (4..s.thumb_h.saturating_sub(4)).step_by(3) {
        for x in (4..s.thumb_w.saturating_sub(4)).step_by(3) {
            let src_x = (((x as f32 - center_x) / scale) + center_x).round() as isize;
            let src_y = (((y as f32 - center_y) / scale) + center_y).round() as isize;
            if src_x < 0 || src_y < 0 || src_x >= s.thumb_w as isize || src_y >= s.thumb_h as isize
            {
                continue;
            }
            let a = prev[src_y as usize * s.thumb_w + src_x as usize] as f32 - mean_a;
            let b = next[y * s.thumb_w + x] as f32 - mean_b;
            sad += (a - b).abs();
        }
    }
    sad / count as f32
}

fn patch_texture(
    frame: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: isize,
) -> f32 {
    let mut energy = 0u32;
    let mut count = 0u32;
    for y in (cy as isize - radius)..=(cy as isize + radius) {
        for x in (cx as isize - radius)..=(cx as isize + radius) {
            if x <= 0 || y <= 0 || x >= width as isize - 1 || y >= height as isize - 1 {
                continue;
            }
            let idx = y as usize * width + x as usize;
            let gx = (frame[idx + 1] as i32 - frame[idx - 1] as i32).unsigned_abs();
            let gy = (frame[idx + width] as i32 - frame[idx - width] as i32).unsigned_abs();
            energy += gx + gy;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        energy as f32 / count as f32
    }
}

fn best_patch_shift(
    prev: &[u8],
    next: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
) -> Option<(f32, f32, f32)> {
    let search = PatchSearch {
        width,
        height,
        cx,
        cy,
    };
    let mut best = evaluate_patch_shift(prev, next, search, 0, 0)?;
    let mut second_best = f32::MAX;
    let mut best_dx = 0isize;
    let mut best_dy = 0isize;

    for step in SEARCH_STEPS {
        loop {
            let mut improved = false;
            let candidates = [
                (best_dx, best_dy - step),
                (best_dx - step, best_dy),
                (best_dx + step, best_dy),
                (best_dx, best_dy + step),
                (best_dx - step, best_dy - step),
                (best_dx + step, best_dy - step),
                (best_dx - step, best_dy + step),
                (best_dx + step, best_dy + step),
            ];
            for (dx, dy) in candidates {
                if dx.abs() > MOTION_SEARCH_RADIUS || dy.abs() > MOTION_SEARCH_RADIUS {
                    continue;
                }
                let Some(mean_sad) = evaluate_patch_shift(prev, next, search, dx, dy) else {
                    continue;
                };
                if mean_sad < best {
                    second_best = best;
                    best = mean_sad;
                    best_dx = dx;
                    best_dy = dy;
                    improved = true;
                } else if mean_sad < second_best {
                    second_best = mean_sad;
                }
            }
            if !improved {
                break;
            }
        }
    }

    // Sub-pixel parabolic interpolation
    let c = best;
    let mut sub_x = 0.0f32;
    let mut sub_y = 0.0f32;

    if best_dx.abs() < MOTION_SEARCH_RADIUS {
        let l = evaluate_patch_shift(prev, next, search, best_dx - 1, best_dy).unwrap_or(best);
        let r = evaluate_patch_shift(prev, next, search, best_dx + 1, best_dy).unwrap_or(best);
        let denom = l - 2.0 * c + r;
        if denom > 1e-4 {
            sub_x = ((l - r) / (2.0 * denom)).clamp(-0.5, 0.5);
        }
    }

    if best_dy.abs() < MOTION_SEARCH_RADIUS {
        let u = evaluate_patch_shift(prev, next, search, best_dx, best_dy - 1).unwrap_or(best);
        let d = evaluate_patch_shift(prev, next, search, best_dx, best_dy + 1).unwrap_or(best);
        let denom = u - 2.0 * c + d;
        if denom > 1e-4 {
            sub_y = ((u - d) / (2.0 * denom)).clamp(-0.5, 0.5);
        }
    }

    // A patch at the search boundary or with no meaningful second-best
    // separation is ambiguous (repeated texture, cuts, or a moving subject).
    // Reject it instead of feeding a plausible-looking but false vector into
    // the global camera model.
    if best_dx.abs() >= MOTION_SEARCH_RADIUS
        || best_dy.abs() >= MOTION_SEARCH_RADIUS
        || !second_best.is_finite()
        || second_best - best < 0.04 * best.max(1.0)
    {
        return None;
    }

    Some((best_dx as f32 + sub_x, best_dy as f32 + sub_y, best))
}

fn evaluate_patch_shift(
    prev: &[u8],
    next: &[u8],
    search: PatchSearch,
    dx: isize,
    dy: isize,
) -> Option<f32> {
    let mut sum_a = 0u32;
    let mut sum_b = 0u32;
    let mut valid = 0usize;
    for py in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
        let y0 = search.cy as isize + py;
        let y1 = y0 + dy;
        if y0 < 0 || y1 < 0 || y0 >= search.height as isize || y1 >= search.height as isize {
            continue;
        }
        for px in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
            let x0 = search.cx as isize + px;
            let x1 = x0 + dx;
            if x0 < 0 || x1 < 0 || x0 >= search.width as isize || x1 >= search.width as isize {
                continue;
            }
            let a = prev[y0 as usize * search.width + x0 as usize] as i32;
            let b = next[y1 as usize * search.width + x1 as usize] as i32;
            sum_a += a as u32;
            sum_b += b as u32;
            valid += 1;
        }
    }
    if valid == 0 {
        return None;
    }

    // Zero-mean SAD is insensitive to uniform exposure changes. Camera flashes
    // and auto-exposure ramps are common in wedding footage and should not be
    // mistaken for camera movement.
    let mean_a = sum_a as f32 / valid as f32;
    let mean_b = sum_b as f32 / valid as f32;
    let mut sad = 0.0f32;
    for py in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
        let y0 = search.cy as isize + py;
        let y1 = y0 + dy;
        if y0 < 0 || y1 < 0 || y0 >= search.height as isize || y1 >= search.height as isize {
            continue;
        }
        for px in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
            let x0 = search.cx as isize + px;
            let x1 = x0 + dx;
            if x0 < 0 || x1 < 0 || x0 >= search.width as isize || x1 >= search.width as isize {
                continue;
            }
            let a = prev[y0 as usize * search.width + x0 as usize] as f32 - mean_a;
            let b = next[y1 as usize * search.width + x1 as usize] as f32 - mean_b;
            sad += (a - b).abs();
        }
    }
    Some(sad / valid as f32)
}

#[derive(Debug, Clone, Copy)]
struct PatchSearch {
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
}
