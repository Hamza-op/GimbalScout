use std::collections::VecDeque;

const MOTION_THUMB_HEIGHT: usize = 144;
const MOTION_THUMB_WIDTH_MAX: usize = 288;
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
}

#[derive(Debug, Clone, Copy)]
struct MotionVector {
    dx: f32,
    dy: f32,
    cx: usize,
    cy: usize,
}

#[derive(Debug, Clone, Copy)]
struct CameraMotionModel {
    tx: f32,
    ty: f32,
    ax: f32,
    bx: f32,
    cx: f32,
    dy: f32,
    mean_residual: f32,
    inliers: usize,
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
    source_scale: f32,
    source_indices: Vec<usize>,
    patch_centers: Vec<(usize, usize)>,
}

impl MotionSampling {
    pub(crate) fn new(src_w: usize, src_h: usize) -> Self {
        let thumb_h = src_h.clamp(24, MOTION_THUMB_HEIGHT);
        let thumb_w = ((src_w.max(1) * thumb_h) / src_h.max(1)).clamp(24, MOTION_THUMB_WIDTH_MAX);
        let source_scale =
            ((src_w as f32 / thumb_w as f32) + (src_h as f32 / thumb_h as f32)) * 0.5;

        let mut source_indices = Vec::with_capacity(thumb_w * thumb_h);
        for y in 0..thumb_h {
            let src_y = ((y * src_h.max(1)) / thumb_h).min(src_h.saturating_sub(1));
            for x in 0..thumb_w {
                let src_x = ((x * src_w.max(1)) / thumb_w).min(src_w.saturating_sub(1));
                source_indices.push((src_y * src_w + src_x) * 3);
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

pub(crate) fn average_pair_motion_features(
    features: &VecDeque<Option<MotionFeatures>>,
) -> MotionFeatures {
    let mut motion_sum = 0.0f32;
    let mut zoom_sum = 0.0f32;
    let mut pairs = 0usize;
    for feature in features.iter().flatten() {
        motion_sum += feature.motion_score;
        zoom_sum += feature.zoom_score;
        pairs += 1;
    }

    if pairs == 0 {
        MotionFeatures::default()
    } else {
        MotionFeatures {
            motion_score: motion_sum / pairs as f32,
            zoom_score: zoom_sum / pairs as f32,
        }
    }
}

pub(crate) fn normalize_motion_features_for_fps(
    features: MotionFeatures,
    analysis_fps: f32,
) -> MotionFeatures {
    let fps_scale = if analysis_fps.is_finite() && analysis_fps > 0.0 {
        analysis_fps / MOTION_SCORE_BASELINE_FPS
    } else {
        1.0
    };
    MotionFeatures {
        motion_score: features.motion_score * fps_scale,
        zoom_score: features.zoom_score * fps_scale,
    }
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
    for &(cx, cy) in &s.patch_centers {
        let texture = patch_texture(prev, s.thumb_w, s.thumb_h, cx, cy, MOTION_PATCH_RADIUS);
        if texture < MOTION_MIN_TEXTURE {
            continue;
        }
        if let Some((dx, dy)) = best_patch_shift(prev, next, s.thumb_w, s.thumb_h, cx, cy) {
            vectors.push(MotionVector {
                dx: dx as f32,
                dy: dy as f32,
                cx,
                cy,
            });
        }
    }

    let observed_support = vectors.len() as f32 / s.patch_centers.len().max(1) as f32;
    let fallback_zoom = estimate_scale_zoom_score(prev, next, s) * observed_support;
    let Some(model) = fit_camera_motion_model(&vectors, s) else {
        return Some(MotionFeatures {
            motion_score: fallback_zoom * 1.15,
            zoom_score: fallback_zoom,
        });
    };

    let decomposition = decompose_affine_motion_model(model);
    let support = model.inliers as f32 / s.patch_centers.len().max(1) as f32;
    let coherence =
        (1.0 - model.mean_residual / (MOTION_SEARCH_RADIUS as f32 + 1.0)).clamp(0.0, 1.0);
    let translation_score =
        (model.tx.powi(2) + model.ty.powi(2)).sqrt() * s.source_scale * support * coherence;
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
    Some(MotionFeatures {
        motion_score: translation_score
            .max(zoom_score * 1.15)
            .max(rotation_score * 0.85)
            .max(shear_score),
        zoom_score,
    })
}

pub(crate) fn scaled_width_even(src_w: u32, src_h: u32, target_h: u32) -> u32 {
    if src_h == 0 || target_h == 0 {
        return 0;
    }
    let w = (src_w as f64) * (target_h as f64) / (src_h as f64);
    let mut w = w.round().max(2.0) as u32;
    if w % 2 == 1 {
        w += 1;
    }
    w
}

pub(crate) fn seconds_to_timeline_frame(seconds: f64, timebase: u32) -> u64 {
    let v = seconds * timebase as f64;
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
    let mut best = solve_camera_motion_model(vectors, center_x, center_y)
        .and_then(|model| score_camera_motion_model(model, vectors, center_x, center_y));

    let seeds = select_motion_seed_vectors(vectors, center_x, center_y);
    if seeds.len() >= 3 {
        let mut iterations = 0usize;
        'ransac: for i in 0..seeds.len() - 2 {
            for j in i + 1..seeds.len() - 1 {
                for k in j + 1..seeds.len() {
                    iterations += 1;
                    if iterations > RANSAC_MAX_ITERATIONS {
                        break 'ransac;
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
                                    && candidate.0.mean_residual < best_model.mean_residual)
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

    let (best_model, best_inliers, _) = best?;
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

    Some(CameraMotionModel {
        mean_residual,
        inliers: inlier_vectors.len(),
        ..refined
    })
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

    let solution = solve_linear_system(m, rhs)?;
    Some(CameraMotionModel {
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

fn decompose_affine_motion_model(model: CameraMotionModel) -> AffineDecomposition {
    let m00 = 1.0 + model.ax;
    let m01 = model.bx;
    let m10 = model.cx;
    let m11 = 1.0 + model.dy;

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
    let predicted_dx = model.tx + model.ax * rx + model.bx * ry;
    let predicted_dy = model.ty + model.cx * rx + model.dy * ry;
    ((vector.dx - predicted_dx).powi(2) + (vector.dy - predicted_dy).powi(2)).sqrt()
}

fn solve_linear_system<const N: usize>(
    mut m: [[f32; N]; N],
    mut rhs: [f32; N],
) -> Option<[f32; N]> {
    for pivot in 0..N {
        let mut best_row = pivot;
        let mut best_val = m[pivot][pivot].abs();
        let mut row = pivot + 1;
        while row < N {
            let candidate = m[row][pivot].abs();
            if candidate > best_val {
                best_val = candidate;
                best_row = row;
            }
            row += 1;
        }
        if best_val < 1e-4 {
            return None;
        }
        if best_row != pivot {
            m.swap(pivot, best_row);
            rhs.swap(pivot, best_row);
        }

        let inv_pivot = 1.0 / m[pivot][pivot];
        let mut col = pivot;
        while col < N {
            m[pivot][col] *= inv_pivot;
            col += 1;
        }
        rhs[pivot] *= inv_pivot;

        let mut row = 0usize;
        while row < N {
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
            while col < N {
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
    let mut sum = 0u64;
    let mut count = 0u64;

    for y in (4..s.thumb_h.saturating_sub(4)).step_by(3) {
        for x in (4..s.thumb_w.saturating_sub(4)).step_by(3) {
            let src_x = (((x as f32 - center_x) / scale) + center_x).round() as isize;
            let src_y = (((y as f32 - center_y) / scale) + center_y).round() as isize;
            if src_x < 0 || src_y < 0 || src_x >= s.thumb_w as isize || src_y >= s.thumb_h as isize
            {
                continue;
            }

            let a = prev[src_y as usize * s.thumb_w + src_x as usize] as i32;
            let b = next[y * s.thumb_w + x] as i32;
            sum += (a - b).unsigned_abs() as u64;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
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
) -> Option<(isize, isize)> {
    let mut best = evaluate_patch_shift(prev, next, width, height, cx, cy, 0, 0)?;
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
                let Some(mean_sad) =
                    evaluate_patch_shift(prev, next, width, height, cx, cy, dx, dy)
                else {
                    continue;
                };
                if mean_sad < best {
                    best = mean_sad;
                    best_dx = dx;
                    best_dy = dy;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }

    Some((best_dx, best_dy))
}

fn evaluate_patch_shift(
    prev: &[u8],
    next: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    dx: isize,
    dy: isize,
) -> Option<u32> {
    let mut sad = 0u32;
    let mut valid = 0u32;
    for py in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
        let y0 = cy as isize + py;
        let y1 = y0 + dy;
        if y0 < 0 || y1 < 0 || y0 >= height as isize || y1 >= height as isize {
            continue;
        }
        for px in -MOTION_PATCH_RADIUS..=MOTION_PATCH_RADIUS {
            let x0 = cx as isize + px;
            let x1 = x0 + dx;
            if x0 < 0 || x1 < 0 || x0 >= width as isize || x1 >= width as isize {
                continue;
            }
            let a = prev[y0 as usize * width + x0 as usize] as i32;
            let b = next[y1 as usize * width + x1 as usize] as i32;
            sad += (a - b).unsigned_abs();
            valid += 1;
        }
    }
    if valid == 0 { None } else { Some(sad / valid) }
}
