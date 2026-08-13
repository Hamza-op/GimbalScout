use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentKind {
    GimbalMove,
    StaticSubject,
    SlowMotion,
    Static, // For clips with no detected movement
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MovementType {
    #[default]
    PanTilt,
    Zoom,
    Roll,
    Complex,
    Subject,
    SlowMotion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub source_path: PathBuf,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub kind: SegmentKind,
    /// Legacy FCP7 numeric label id. Retained for diagnostics.
    #[allow(dead_code)]
    pub label_id: u8,
    /// Running mean motion energy across every window merged into this segment.
    /// Analyser emits one window at a time with `window_count == 1`; the merge
    /// step accumulates a weighted mean so the exported score reflects the
    /// whole run rather than a single spiking window.
    pub motion_score: f32,
    /// Mean zoom-in / zoom-out evidence across the merged windows. Averaging
    /// prevents one ambiguous patch match from relabelling an entire run.
    pub zoom_score: f32,
    /// Dominant movement interpretation for editor-facing metadata.
    #[serde(default)]
    pub movement_type: MovementType,
    /// Coherence of the camera-motion fit, in [0,1]. Higher means a broad
    /// part of the frame agrees on the same camera move.
    #[serde(default = "default_motion_confidence")]
    pub motion_confidence: f32,
    /// Temporal direction/speed consistency in [0,1]. Unlike spatial
    /// confidence, this separates a smooth move from whole-frame shake.
    #[serde(default = "default_motion_smoothness")]
    pub motion_smoothness: f32,
    /// Peak person-detection confidence observed across the merged windows.
    /// Peak (rather than mean) matches editorial intent: a clip that contains
    /// a clearly-detected person anywhere inside it is a person shot.
    pub person_confidence: Option<f32>,
    /// Number of analysis windows merged into this segment.
    pub window_count: u32,
    /// Score indicating how "cinematic" this segment is (higher is better).
    /// Combines motion smoothness, subject presence, and slow-motion quality.
    #[serde(default)]
    pub cinematic_score: f32,
}

fn default_motion_confidence() -> f32 {
    0.0
}

fn default_motion_smoothness() -> f32 {
    0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityConfig {
    pub operator_spike_max_seconds: f64,
    pub edge_spike_margin_seconds: f64,
    pub jerk_max_seconds: f64,
    pub edge_jerk_max_seconds: f64,
    pub edge_jerk_margin_seconds: f64,
    pub tail_jerk_max_seconds: f64,
    pub tail_jerk_edge_seconds: f64,
    pub tail_jerk_start_seconds: f64,
    pub min_editorial_duration_seconds: f64,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            operator_spike_max_seconds: 1.75,
            edge_spike_margin_seconds: 1.5,
            jerk_max_seconds: 2.4,
            edge_jerk_max_seconds: 3.2,
            edge_jerk_margin_seconds: 1.5,
            tail_jerk_max_seconds: 3.2,
            tail_jerk_edge_seconds: 1.5,
            tail_jerk_start_seconds: 3.6,
            min_editorial_duration_seconds: 1.25,
        }
    }
}

const SPIKE_MOTION_SCORE: f32 = 3.8;
const SPIKE_ZOOM_SCORE: f32 = 2.5;
const JERK_LOW_CONFIDENCE: f32 = 0.70;
const MIN_STABLE_WINDOWS: u32 = 2;
const SINGLE_WINDOW_GIMBAL_MOTION: f32 = 3.1;
const SINGLE_WINDOW_GIMBAL_ZOOM: f32 = 1.8;
const SINGLE_WINDOW_STATIC_PERSON: f32 = 0.72;
const SINGLE_WINDOW_SLOWMO_MOTION: f32 = 1.2;
const MULTI_WINDOW_SCORE_GIMBAL: f32 = 0.50;
const MULTI_WINDOW_SCORE_STATIC_SUBJECT: f32 = 0.47;
const MULTI_WINDOW_SCORE_SLOWMO: f32 = 0.49;

/// Merge adjacent same-kind windows into runs.
///
/// Two improvements over a strict equality join:
///
/// 1. **Gap tolerance** — allows merging across a rounding drift or a single
///    dropped window (gap up to 0.5× the window duration).  Without this a
///    tiny float mismatch between consecutive window boundaries left the
///    timeline fragmented into 1-second clips.
/// 2. **Isolated-window smoothing** — a single opposite-kind window sandwiched
///    between two same-kind runs is reclassified to match its neighbours so
///    brief detector wobble does not fracture otherwise stable runs.
pub fn merge_segments(mut windows: Vec<Segment>) -> Vec<Segment> {
    if windows.is_empty() {
        return windows;
    }

    windows.sort_by(|a, b| {
        a.source_path.cmp(&b.source_path).then_with(|| {
            a.start_seconds
                .partial_cmp(&b.start_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    // Pass 1: flip single-window isolated outliers to match their neighbours.
    smooth_isolated_windows(&mut windows);

    // Pass 2: greedy merge with a gap tolerance derived from each window's
    // own duration so the merge is independent of the configured window_seconds.
    let mut merged: Vec<Segment> = Vec::with_capacity(windows.len());
    for seg in windows {
        let should_merge = match merged.last() {
            Some(prev) => {
                prev.kind == seg.kind
                    && prev.source_path == seg.source_path
                    && within_merge_gap(prev, &seg)
            }
            None => false,
        };
        if should_merge {
            let prev = merged.last_mut().expect("last exists in merge branch");
            prev.end_seconds = seg.end_seconds;
            prev.end_frame = seg.end_frame;
            // Running mean weighted by window_count keeps the exported metric
            // representative for long runs of mixed-intensity motion.
            let pw = prev.window_count as f32;
            let sw = seg.window_count.max(1) as f32;
            prev.motion_score = (prev.motion_score * pw + seg.motion_score * sw) / (pw + sw);
            prev.motion_confidence =
                (prev.motion_confidence * pw + seg.motion_confidence * sw) / (pw + sw);
            prev.motion_smoothness =
                (prev.motion_smoothness * pw + seg.motion_smoothness * sw) / (pw + sw);
            prev.cinematic_score =
                (prev.cinematic_score * pw + seg.cinematic_score * sw) / (pw + sw);
            prev.zoom_score = weighted_mean(
                prev.zoom_score,
                pw as u32,
                seg.zoom_score,
                seg.window_count.max(1),
            );
            prev.movement_type = dominant_movement(prev, &seg);
            prev.person_confidence = match (prev.person_confidence, seg.person_confidence) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            prev.window_count = prev.window_count.saturating_add(seg.window_count.max(1));
        } else {
            merged.push(seg);
        }
    }

    merged
}

/// Reclassify an isolated 1-window segment whose two neighbours (inside the
/// same source file, within merge distance) share the opposite kind.  This
/// runs before the greedy merge so the flipped window folds into the run.
fn smooth_isolated_windows(windows: &mut [Segment]) {
    if windows.len() < 3 {
        return;
    }
    for i in 1..windows.len() - 1 {
        let (left, mid, right) = {
            let (head, tail) = windows.split_at_mut(i);
            let (mid_slice, tail) = tail.split_at_mut(1);
            (&head[i - 1], &mut mid_slice[0], &tail[0])
        };
        if mid.window_count != 1 {
            continue;
        }
        if left.kind != right.kind || left.kind == mid.kind {
            continue;
        }
        if left.source_path != mid.source_path || mid.source_path != right.source_path {
            continue;
        }
        // Only smooth when the two neighbours are close enough that the
        // isolated window is clearly interrupting a single coherent run.
        let lg = mid.start_seconds - left.end_seconds;
        let rg = right.start_seconds - mid.end_seconds;
        let window_span = (mid.end_seconds - mid.start_seconds).max(1e-3);
        if lg.abs() <= 1.5 * window_span && rg.abs() <= 1.5 * window_span {
            relabel_segment(mid, left.kind, left.movement_type);
        }
    }
}

fn relabel_segment(segment: &mut Segment, kind: SegmentKind, movement_type: MovementType) {
    segment.kind = kind;
    segment.label_id = kind.label_id();
    segment.movement_type = match kind {
        SegmentKind::GimbalMove => movement_type,
        SegmentKind::StaticSubject | SegmentKind::Static => MovementType::Subject,
        SegmentKind::SlowMotion => MovementType::SlowMotion,
    };
    if kind != SegmentKind::StaticSubject {
        segment.person_confidence = None;
    }
}

fn within_merge_gap(prev: &Segment, seg: &Segment) -> bool {
    let window_span = (seg.end_seconds - seg.start_seconds).max(1e-3);
    let gap = seg.start_seconds - prev.end_seconds;
    // Accept small negative overlaps (rounding) and forward gaps up to 0.5×
    // the window duration so a single dropped window never breaks a run.
    gap <= 0.5 * window_span && gap >= -window_span
}

impl SegmentKind {
    pub fn label_id(self) -> u8 {
        match self {
            SegmentKind::GimbalMove => 4,
            SegmentKind::StaticSubject => 1,
            SegmentKind::SlowMotion => 5,
            SegmentKind::Static => 2, // e.g. Cerulean/Rose/Unique
        }
    }
}

/// Reduce analyser runs to exportable selects for one source clip.
///
/// This keeps every confident detection in source order. Filtering is limited
/// to obvious operator noise such as edge spikes and final handle jerks; score
/// ranking must not hide valid movement elsewhere in the clip.
pub fn select_source_segments(
    source_duration_seconds: f64,
    mut segments: Vec<Segment>,
    config: &SensitivityConfig,
) -> Vec<Segment> {
    segments.retain(|seg| !looks_like_jerk_movement(source_duration_seconds, seg, config));
    segments.retain(|seg| !looks_like_tail_operator_jerk(source_duration_seconds, seg, config));
    segments.retain(|seg| !looks_like_operator_spike(source_duration_seconds, seg, config));
    segments.retain(|seg| passes_editorial_confidence(seg, config));

    coalesce_overlapping_selects(&mut segments);

    segments.sort_by(|a, b| {
        a.source_path
            .cmp(&b.source_path)
            .then_with(|| {
                a.start_seconds
                    .partial_cmp(&b.start_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.end_seconds
                    .partial_cmp(&b.end_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    segments
}

/// Turn long detection runs into concise, editor-friendly selects centered on
/// their strongest original analysis window. Short detections gain up to one
/// second of context on either side, while every result stays within the
/// requested duration and source-media bounds.
pub fn focus_editorial_highlights(
    segments: &mut [Segment],
    analysis_windows: &[Segment],
    source_duration_seconds: f64,
    fps_num: u32,
    fps_den: u32,
    max_select_seconds: f64,
) {
    if !source_duration_seconds.is_finite() || source_duration_seconds <= 0.0 {
        return;
    }
    let maximum = max_select_seconds
        .clamp(2.0, 30.0)
        .min(source_duration_seconds);

    for segment in segments {
        let overlaps = |window: &&Segment| {
            window.source_path == segment.source_path
                && window.end_seconds > segment.start_seconds
                && window.start_seconds < segment.end_seconds
        };
        let peak = analysis_windows
            .iter()
            .filter(overlaps)
            .filter(|window| window.kind == segment.kind)
            .max_by(|a, b| segment_quality_score(a).total_cmp(&segment_quality_score(b)))
            .or_else(|| {
                analysis_windows
                    .iter()
                    .filter(overlaps)
                    .max_by(|a, b| segment_quality_score(a).total_cmp(&segment_quality_score(b)))
            });

        let original_duration = (segment.end_seconds - segment.start_seconds).max(0.0);
        let target_duration = (original_duration + 2.0).min(maximum).max(0.001);
        let center = peak
            .map(|window| (window.start_seconds + window.end_seconds) * 0.5)
            .unwrap_or((segment.start_seconds + segment.end_seconds) * 0.5)
            .clamp(0.0, source_duration_seconds);
        let latest_start = (source_duration_seconds - target_duration).max(0.0);
        let start = (center - target_duration * 0.5).clamp(0.0, latest_start);
        let end = (start + target_duration).min(source_duration_seconds);

        segment.start_seconds = start;
        segment.end_seconds = end;
        segment.start_frame = seconds_to_frame(start, fps_num, fps_den);
        segment.end_frame =
            seconds_to_frame(end, fps_num, fps_den).max(segment.start_frame.saturating_add(1));

        // Keep the run's stability evidence, but rank/export it using the
        // actual peak that the editor will see rather than a diluted average
        // across a long merged run.
        if let Some(peak) = peak {
            segment.motion_score = peak.motion_score;
            segment.zoom_score = peak.zoom_score;
            segment.movement_type = peak.movement_type;
            segment.motion_confidence = peak.motion_confidence;
            segment.motion_smoothness = peak.motion_smoothness;
            segment.person_confidence = peak.person_confidence;
            segment.cinematic_score = peak.cinematic_score;
        }
    }
}

fn seconds_to_frame(seconds: f64, fps_num: u32, fps_den: u32) -> u64 {
    if fps_num == 0 || fps_den == 0 {
        return 0;
    }
    (seconds.max(0.0) * fps_num as f64 / fps_den as f64).round() as u64
}

pub(crate) fn segment_quality_score(seg: &Segment) -> f32 {
    let duration_seconds = (seg.end_seconds - seg.start_seconds).max(0.0);
    let duration_score = (duration_seconds / 3.0).clamp(0.0, 1.0) as f32 * 0.18;
    let motion_score = (seg.motion_score / 4.0).clamp(0.0, 1.5) * 0.24;
    let zoom_score = (seg.zoom_score / 2.5).clamp(0.0, 1.0) * 0.08;
    let coherence_score = seg.motion_confidence.clamp(0.0, 1.0) * 0.12;
    let smoothness_score = seg.motion_smoothness.clamp(0.0, 1.0) * 0.18;
    let person_score = seg.person_confidence.unwrap_or(0.0).clamp(0.0, 1.0) * 0.22;
    let cinematic_score = seg.cinematic_score.clamp(0.0, 1.0) * 0.12;
    let stability_bonus = (seg.window_count.saturating_sub(1).min(4) as f32) * 0.03;
    let kind_bonus = match seg.kind {
        SegmentKind::GimbalMove => 0.02,
        SegmentKind::StaticSubject => 0.06,
        SegmentKind::SlowMotion => 0.08,
        SegmentKind::Static => 0.00,
    };

    duration_score
        + motion_score
        + zoom_score
        + coherence_score
        + smoothness_score
        + person_score
        + cinematic_score
        + stability_bonus
        + kind_bonus
}

fn multi_window_score_threshold(kind: SegmentKind) -> f32 {
    match kind {
        SegmentKind::GimbalMove => MULTI_WINDOW_SCORE_GIMBAL,
        SegmentKind::StaticSubject => MULTI_WINDOW_SCORE_STATIC_SUBJECT,
        SegmentKind::SlowMotion => MULTI_WINDOW_SCORE_SLOWMO,
        SegmentKind::Static => 0.0,
    }
}

fn coalesce_overlapping_selects(segments: &mut Vec<Segment>) {
    if segments.len() < 2 {
        return;
    }

    segments.sort_by(|a, b| {
        a.source_path
            .cmp(&b.source_path)
            .then_with(|| {
                a.start_seconds
                    .partial_cmp(&b.start_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.end_seconds
                    .partial_cmp(&b.end_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
    for seg in segments.drain(..) {
        let should_merge = merged.last().is_some_and(|prev| {
            prev.source_path == seg.source_path
                && prev.kind == seg.kind
                && seg.start_seconds <= prev.end_seconds
        });
        if should_merge {
            let prev = merged.last_mut().expect("last exists in merge branch");
            merge_select_into(prev, seg);
        } else if let Some(prev) = merged.last_mut().filter(|prev| {
            prev.source_path == seg.source_path && seg.start_seconds < prev.end_seconds
        }) {
            // Never export overlapping selects of different kinds. Split the
            // overlap at its midpoint so each label owns a deterministic
            // source interval.
            let boundary = (prev.end_seconds + seg.start_seconds) * 0.5;
            let prev_ratio = ((boundary - prev.start_seconds)
                / (prev.end_seconds - prev.start_seconds).max(1e-6))
            .clamp(0.0, 1.0);
            prev.end_seconds = boundary.max(prev.start_seconds);
            prev.end_frame = prev.start_frame
                + ((prev.end_frame.saturating_sub(prev.start_frame)) as f64 * prev_ratio).round()
                    as u64;
            let mut trimmed = seg;
            let seg_ratio = ((boundary - trimmed.start_seconds)
                / (trimmed.end_seconds - trimmed.start_seconds).max(1e-6))
            .clamp(0.0, 1.0);
            trimmed.start_seconds = boundary.min(trimmed.end_seconds);
            trimmed.start_frame = trimmed.start_frame
                + ((trimmed.end_frame.saturating_sub(trimmed.start_frame)) as f64 * seg_ratio)
                    .round() as u64;
            if trimmed.end_seconds > trimmed.start_seconds
                && trimmed.end_frame > trimmed.start_frame
            {
                merged.push(trimmed);
            }
        } else {
            merged.push(seg);
        }
    }
    *segments = merged;
}

fn merge_select_into(prev: &mut Segment, seg: Segment) {
    let prev_windows = prev.window_count.max(1);
    let seg_windows = seg.window_count.max(1);
    let total_windows = prev_windows.saturating_add(seg_windows);
    prev.start_seconds = prev.start_seconds.min(seg.start_seconds);
    prev.end_seconds = prev.end_seconds.max(seg.end_seconds);
    prev.start_frame = prev.start_frame.min(seg.start_frame);
    prev.end_frame = prev.end_frame.max(seg.end_frame);
    prev.motion_score = (prev.motion_score * prev_windows as f32
        + seg.motion_score * seg_windows as f32)
        / total_windows as f32;
    prev.motion_confidence = (prev.motion_confidence * prev_windows as f32
        + seg.motion_confidence * seg_windows as f32)
        / total_windows as f32;
    prev.motion_smoothness = (prev.motion_smoothness * prev_windows as f32
        + seg.motion_smoothness * seg_windows as f32)
        / total_windows as f32;
    prev.cinematic_score = (prev.cinematic_score * prev_windows as f32
        + seg.cinematic_score * seg_windows as f32)
        / total_windows as f32;
    prev.zoom_score = weighted_mean(prev.zoom_score, prev_windows, seg.zoom_score, seg_windows);
    prev.movement_type = dominant_movement(prev, &seg);
    prev.person_confidence = match (prev.person_confidence, seg.person_confidence) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    prev.window_count = total_windows;
}

fn weighted_mean(a: f32, aw: u32, b: f32, bw: u32) -> f32 {
    let total = aw.saturating_add(bw).max(1) as f32;
    (a * aw as f32 + b * bw as f32) / total
}

fn dominant_movement(a: &Segment, b: &Segment) -> MovementType {
    let a_weight = a.window_count.max(1) as f32;
    let b_weight = b.window_count.max(1) as f32;
    let evidence = |segment: &Segment, movement: MovementType| match movement {
        MovementType::Zoom => segment.zoom_score,
        MovementType::Subject | MovementType::SlowMotion => 0.0,
        MovementType::PanTilt | MovementType::Roll | MovementType::Complex => {
            let continuity_bonus = if segment.movement_type == movement {
                1.15
            } else {
                1.0
            };
            segment.motion_score * continuity_bonus
        }
    };
    let choices = [
        MovementType::Zoom,
        MovementType::Roll,
        MovementType::Complex,
        MovementType::PanTilt,
    ];
    choices
        .into_iter()
        .max_by(|left, right| {
            let l = evidence(a, *left) * a_weight + evidence(b, *left) * b_weight;
            let r = evidence(a, *right) * a_weight + evidence(b, *right) * b_weight;
            l.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(a.movement_type)
}

fn passes_editorial_confidence(seg: &Segment, config: &SensitivityConfig) -> bool {
    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    let score = segment_quality_score(seg);

    if seg.window_count >= MIN_STABLE_WINDOWS {
        return score >= multi_window_score_threshold(seg.kind);
    }

    let single_window_ok = match seg.kind {
        SegmentKind::GimbalMove => {
            let strong_short_move = duration >= 0.50
                && seg.motion_smoothness >= 0.75
                && (seg.motion_score >= 5.0 || seg.zoom_score >= 3.0);
            (duration >= config.min_editorial_duration_seconds || strong_short_move)
                && (seg.motion_score >= SINGLE_WINDOW_GIMBAL_MOTION
                    || seg.zoom_score >= SINGLE_WINDOW_GIMBAL_ZOOM)
        }
        SegmentKind::StaticSubject => {
            duration >= config.min_editorial_duration_seconds
                && seg.person_confidence.unwrap_or(0.0) >= SINGLE_WINDOW_STATIC_PERSON
        }
        // Static is emitted only as the deliberate best-available fallback,
        // after detector/motion candidates have failed. It must not require a
        // person signal or no-signal clips would disappear entirely.
        SegmentKind::Static => duration > 0.0,
        SegmentKind::SlowMotion => {
            duration >= config.min_editorial_duration_seconds
                && (seg.motion_score >= SINGLE_WINDOW_SLOWMO_MOTION || seg.zoom_score >= 0.8)
        }
    };

    single_window_ok && score >= multi_window_score_threshold(seg.kind)
}

fn looks_like_operator_spike(
    source_duration_seconds: f64,
    seg: &Segment,
    config: &SensitivityConfig,
) -> bool {
    if seg.kind == SegmentKind::SlowMotion || seg.person_confidence.is_some() {
        return false;
    }

    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    if duration > config.operator_spike_max_seconds {
        return false;
    }

    let is_high_energy =
        seg.motion_score >= SPIKE_MOTION_SCORE || seg.zoom_score >= SPIKE_ZOOM_SCORE;
    if !is_high_energy {
        return false;
    }
    // A short, high-energy move is still a useful editorial shot when the
    // motion direction is coherent. Only discard it when temporal evidence
    // says it is an operator snap/jerk.
    if seg.motion_smoothness >= 0.72 {
        return false;
    }

    let touches_clip_edge = seg.start_seconds <= config.edge_spike_margin_seconds
        || source_duration_seconds - seg.end_seconds <= config.edge_spike_margin_seconds;

    touches_clip_edge || duration <= 1.5
}

fn looks_like_jerk_movement(
    source_duration_seconds: f64,
    seg: &Segment,
    config: &SensitivityConfig,
) -> bool {
    if seg.kind != SegmentKind::GimbalMove {
        return false;
    }

    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    if duration <= 0.0 {
        return true;
    }

    let edge = touches_clip_edge(
        source_duration_seconds,
        seg,
        config.edge_jerk_margin_seconds,
    );
    let short_unstable = duration <= config.jerk_max_seconds
        && (seg.motion_smoothness < 0.48
            || (seg.window_count <= 2
                && seg.motion_confidence < JERK_LOW_CONFIDENCE
                && seg.motion_smoothness < 0.68));
    let edge_unstable = edge
        && duration <= config.edge_jerk_max_seconds
        && seg.motion_smoothness < 0.62
        && (seg.motion_confidence < 0.80 || seg.window_count <= 3);
    let high_energy_snap = duration <= config.jerk_max_seconds
        && seg.motion_smoothness < 0.72
        && (seg.motion_score >= SPIKE_MOTION_SCORE || seg.zoom_score >= SPIKE_ZOOM_SCORE);

    let movement_is_jerk_prone = matches!(
        seg.movement_type,
        MovementType::PanTilt | MovementType::Roll | MovementType::Complex
    );

    movement_is_jerk_prone && (short_unstable || edge_unstable || high_energy_snap)
}

fn looks_like_tail_operator_jerk(
    source_duration_seconds: f64,
    seg: &Segment,
    config: &SensitivityConfig,
) -> bool {
    if seg.kind != SegmentKind::GimbalMove {
        return false;
    }
    if !source_duration_seconds.is_finite() || source_duration_seconds <= 0.0 {
        return false;
    }

    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    if duration > config.tail_jerk_max_seconds {
        return false;
    }

    touches_clip_edge(source_duration_seconds, seg, config.tail_jerk_edge_seconds)
        && seg.start_seconds >= source_duration_seconds - config.tail_jerk_start_seconds
        && seg.motion_smoothness < 0.62
}

fn touches_clip_edge(source_duration_seconds: f64, seg: &Segment, margin_seconds: f64) -> bool {
    if !source_duration_seconds.is_finite() || source_duration_seconds <= 0.0 {
        return false;
    }
    seg.start_seconds <= margin_seconds
        || source_duration_seconds - seg.end_seconds <= margin_seconds
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn window(
        p: &Path,
        start: f64,
        end: f64,
        kind: SegmentKind,
        motion: f32,
        person: Option<f32>,
    ) -> Segment {
        Segment {
            source_path: p.to_path_buf(),
            start_frame: (start * 10.0) as u64,
            end_frame: (end * 10.0) as u64,
            start_seconds: start,
            end_seconds: end,
            kind,
            label_id: kind.label_id(),
            motion_score: motion,
            zoom_score: 0.0,
            movement_type: MovementType::PanTilt,
            motion_confidence: 0.9,
            motion_smoothness: 0.9,
            person_confidence: person,
            window_count: 1,
            cinematic_score: 0.0,
        }
    }

    #[test]
    fn merge_adjacent_runs() {
        let p = PathBuf::from("a.mov");
        let s1 = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 2.0, None);
        let s2 = window(&p, 1.0, 2.0, SegmentKind::GimbalMove, 1.0, None);
        let merged = merge_segments(vec![s1, s2]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].start_seconds, 0.0);
        assert_eq!(merged[0].end_seconds, 2.0);
        assert_eq!(merged[0].window_count, 2);
        // Running mean of 2.0 and 1.0 is 1.5 — not the old "max".
        assert!((merged[0].motion_score - 1.5).abs() < 1e-5);
    }

    #[test]
    fn merge_across_small_gap() {
        // A 0.02s drift between windows must not fracture the run.
        let p = PathBuf::from("a.mov");
        let s1 = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 2.0, None);
        let s2 = window(&p, 1.02, 2.0, SegmentKind::GimbalMove, 2.0, None);
        let merged = merge_segments(vec![s1, s2]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].window_count, 2);
    }

    #[test]
    fn long_runs_do_not_expand_their_gap_tolerance() {
        let p = PathBuf::from("a.mov");
        let long_run = window(&p, 0.0, 20.0, SegmentKind::GimbalMove, 2.0, None);
        let later = window(&p, 25.0, 26.0, SegmentKind::GimbalMove, 2.0, None);
        let merged = merge_segments(vec![long_run, later]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn isolated_window_is_smoothed_into_neighbours() {
        // Static-Gimbal-Static with a single isolated Gimbal window should
        // collapse to one continuous StaticSubject segment.
        let p = PathBuf::from("a.mov");
        let s1 = window(&p, 0.0, 1.0, SegmentKind::StaticSubject, 0.5, Some(0.9));
        let odd = window(&p, 1.0, 2.0, SegmentKind::GimbalMove, 3.5, None);
        let s3 = window(&p, 2.0, 3.0, SegmentKind::StaticSubject, 0.4, Some(0.95));
        let merged = merge_segments(vec![s1, odd, s3]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].kind, SegmentKind::StaticSubject);
        assert_eq!(merged[0].window_count, 3);
        // Peak person confidence survives the merge.
        assert_eq!(merged[0].person_confidence, Some(0.95));
    }

    #[test]
    fn isolated_subject_window_relabel_clears_stale_subject_metadata() {
        let p = PathBuf::from("a.mov");
        let left = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 2.0, None);
        let middle = window(&p, 1.0, 2.0, SegmentKind::StaticSubject, 0.4, Some(0.92));
        let right = window(&p, 2.0, 3.0, SegmentKind::GimbalMove, 2.0, None);
        let merged = merge_segments(vec![left, middle, right]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].kind, SegmentKind::GimbalMove);
        assert_eq!(merged[0].movement_type, MovementType::PanTilt);
        assert_eq!(merged[0].person_confidence, None);
    }

    #[test]
    fn keeps_multiple_segments_per_clip_in_time_order() {
        let p = PathBuf::from("a.mov");
        let s1 = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 2.0, None);
        let s2 = window(&p, 1.0, 2.0, SegmentKind::StaticSubject, 0.2, Some(0.9));
        let merged = merge_segments(vec![s1, s2]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].kind, SegmentKind::GimbalMove);
        assert_eq!(merged[1].kind, SegmentKind::StaticSubject);
        assert_eq!(merged[0].start_seconds, 0.0);
        assert_eq!(merged[1].start_seconds, 1.0);
    }

    #[test]
    fn distinct_segments_preserve_their_zoom_scores() {
        let p = PathBuf::from("a.mov");
        let mut zoom = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 1.5, None);
        zoom.zoom_score = 3.0;
        let pan = window(&p, 4.5, 6.5, SegmentKind::GimbalMove, 2.5, None);

        let merged = merge_segments(vec![pan, zoom]);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].zoom_score, 3.0);
        assert_eq!(merged[0].start_seconds, 0.0);
        assert_eq!(merged[1].zoom_score, 0.0);
        assert_eq!(merged[1].start_seconds, 4.5);
    }

    #[test]
    fn short_clip_keeps_all_confident_cuts() {
        let p = PathBuf::from("a.mov");
        let subject = window(&p, 0.0, 5.0, SegmentKind::StaticSubject, 0.2, Some(0.82));
        let strong = window(&p, 20.0, 23.0, SegmentKind::GimbalMove, 4.0, None);

        let selected =
            select_source_segments(45.0, vec![subject, strong], &SensitivityConfig::default());

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].start_seconds, 0.0);
        assert_eq!(selected[1].start_seconds, 20.0);
    }

    #[test]
    fn long_clip_keeps_all_non_overlapping_confident_cuts_in_time_order() {
        let p = PathBuf::from("a.mov");
        let a = window(&p, 0.0, 20.0, SegmentKind::GimbalMove, 4.0, None);
        let overlap = window(&p, 10.0, 30.0, SegmentKind::GimbalMove, 5.0, None);
        let b = window(&p, 70.0, 80.0, SegmentKind::SlowMotion, 2.0, None);

        let selected =
            select_source_segments(130.0, vec![a, overlap, b], &SensitivityConfig::default());

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].start_seconds, 0.0);
        assert_eq!(selected[0].end_seconds, 30.0);
        assert_eq!(selected[1].start_seconds, 70.0);
    }

    #[test]
    fn overlapping_mixed_kinds_collapse_to_one_export_select() {
        let p = PathBuf::from("a.mov");
        let subject = window(&p, 0.0, 6.0, SegmentKind::StaticSubject, 0.5, Some(0.9));
        let motion = window(&p, 5.0, 7.0, SegmentKind::GimbalMove, 2.5, None);
        let zoom_subject = window(&p, 6.0, 8.0, SegmentKind::StaticSubject, 1.0, Some(0.95));

        let selected = select_source_segments(
            10.0,
            vec![subject, motion, zoom_subject],
            &SensitivityConfig::default(),
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].start_seconds, 0.0);
        assert_eq!(selected[0].end_seconds, 8.0);
        assert_eq!(selected[0].kind, SegmentKind::StaticSubject);
        assert_eq!(selected[0].person_confidence, Some(0.95));
    }

    #[test]
    fn rejects_short_high_energy_operator_spike_at_clip_end() {
        let p = PathBuf::from("a.mov");
        let mut spike = window(&p, 8.75, 10.0, SegmentKind::GimbalMove, 7.7, None);
        spike.zoom_score = 5.98;
        spike.motion_smoothness = 0.20;

        let selected = select_source_segments(10.08, vec![spike], &SensitivityConfig::default());

        assert!(selected.is_empty());
    }

    #[test]
    fn keeps_short_high_energy_move_when_temporally_smooth() {
        let p = PathBuf::from("a.mov");
        let mut move_seg = window(&p, 4.0, 5.0, SegmentKind::GimbalMove, 7.7, None);
        move_seg.zoom_score = 5.98;
        move_seg.motion_smoothness = 0.90;
        assert_eq!(
            select_source_segments(10.0, vec![move_seg], &SensitivityConfig::default()).len(),
            1
        );
    }

    #[test]
    fn rejects_short_tail_jerk_even_without_spike_energy() {
        let p = PathBuf::from("a.mov");
        let mut jerk = window(&p, 18.1, 19.8, SegmentKind::GimbalMove, 2.4, None);
        jerk.window_count = 2;
        jerk.motion_smoothness = 0.30;

        let selected = select_source_segments(20.0, vec![jerk], &SensitivityConfig::default());

        assert!(selected.is_empty());
    }

    #[test]
    fn rejects_tail_camera_move_when_it_can_be_operator_jerk() {
        let p = PathBuf::from("a.mov");
        let mut move_seg = window(&p, 16.5, 19.4, SegmentKind::GimbalMove, 2.4, None);
        move_seg.window_count = 3;
        move_seg.motion_smoothness = 0.30;

        let selected = select_source_segments(20.0, vec![move_seg], &SensitivityConfig::default());

        assert!(selected.is_empty());
    }

    #[test]
    fn keeps_real_move_that_does_not_touch_clip_tail() {
        let p = PathBuf::from("a.mov");
        let mut move_seg = window(&p, 12.0, 15.0, SegmentKind::GimbalMove, 2.4, None);
        move_seg.window_count = 3;

        let selected = select_source_segments(20.0, vec![move_seg], &SensitivityConfig::default());

        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn rejects_borderline_single_window_gimbal_move() {
        let p = PathBuf::from("a.mov");
        let mut move_seg = window(&p, 3.75, 8.0, SegmentKind::GimbalMove, 2.79, None);
        move_seg.zoom_score = 1.66;

        let selected = select_source_segments(8.16, vec![move_seg], &SensitivityConfig::default());

        assert!(selected.is_empty());
    }

    #[test]
    fn keeps_single_window_static_subject_only_with_strong_person_signal() {
        let p = PathBuf::from("a.mov");
        let weak = window(&p, 2.0, 3.5, SegmentKind::StaticSubject, 0.3, Some(0.61));
        let strong = window(&p, 2.0, 3.5, SegmentKind::StaticSubject, 0.3, Some(0.82));

        assert!(select_source_segments(20.0, vec![weak], &SensitivityConfig::default()).is_empty());
        assert_eq!(
            select_source_segments(20.0, vec![strong], &SensitivityConfig::default()).len(),
            1
        );
    }

    #[test]
    fn keeps_multi_window_segment_even_when_per_window_scores_are_borderline() {
        let p = PathBuf::from("a.mov");
        let a = window(&p, 4.0, 5.0, SegmentKind::GimbalMove, 1.9, None);
        let b = window(&p, 5.0, 6.0, SegmentKind::GimbalMove, 2.0, None);
        let c = window(&p, 6.0, 7.0, SegmentKind::GimbalMove, 1.9, None);
        let d = window(&p, 7.0, 8.0, SegmentKind::GimbalMove, 2.0, None);

        let merged = merge_segments(vec![a, b, c, d]);
        let selected = select_source_segments(20.0, merged, &SensitivityConfig::default());

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].window_count, 4);
    }

    #[test]
    fn long_run_is_focused_on_its_strongest_window() {
        let p = PathBuf::from("a.mov");
        let mut weak = window(&p, 2.0, 3.0, SegmentKind::GimbalMove, 2.0, None);
        weak.cinematic_score = 0.2;
        let mut peak = window(&p, 18.0, 19.0, SegmentKind::GimbalMove, 6.0, None);
        peak.cinematic_score = 0.9;
        let mut run = window(&p, 0.0, 24.0, SegmentKind::GimbalMove, 3.0, None);
        run.window_count = 20;

        focus_editorial_highlights(
            std::slice::from_mut(&mut run),
            &[weak, peak],
            30.0,
            25,
            1,
            8.0,
        );

        assert_eq!(run.start_seconds, 14.5);
        assert_eq!(run.end_seconds, 22.5);
        assert_eq!(run.motion_score, 6.0);
        assert_eq!(run.start_frame, 363);
        assert_eq!(run.end_frame, 563);
    }

    #[test]
    fn short_select_receives_context_without_crossing_source_bounds() {
        let p = PathBuf::from("a.mov");
        let peak = window(&p, 0.0, 1.0, SegmentKind::Static, 0.0, None);
        let mut select = peak.clone();

        focus_editorial_highlights(
            std::slice::from_mut(&mut select),
            std::slice::from_ref(&peak),
            12.0,
            25,
            1,
            8.0,
        );

        assert_eq!(select.start_seconds, 0.0);
        assert_eq!(select.end_seconds, 3.0);
    }
}
