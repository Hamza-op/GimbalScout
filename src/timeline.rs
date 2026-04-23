use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentKind {
    GimbalMove,
    StaticSubject,
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
    /// Peak zoom-in / zoom-out evidence across the merged windows.
    pub zoom_score: f32,
    /// Peak person-detection confidence observed across the merged windows.
    /// Peak (rather than mean) matches editorial intent: a clip that contains
    /// a clearly-detected person anywhere inside it is a person shot.
    pub person_confidence: Option<f32>,
    /// Number of analysis windows merged into this segment.
    pub window_count: u32,
}

const LONG_CLIP_SECONDS: f64 = 60.0;
const OPERATOR_SPIKE_MAX_SECONDS: f64 = 1.75;
const EDGE_SPIKE_MARGIN_SECONDS: f64 = 0.75;
const SPIKE_MOTION_SCORE: f32 = 4.5;
const SPIKE_ZOOM_SCORE: f32 = 3.0;
const MIN_STABLE_WINDOWS: u32 = 2;
const SINGLE_WINDOW_GIMBAL_MOTION: f32 = 3.1;
const SINGLE_WINDOW_GIMBAL_ZOOM: f32 = 1.8;
const SINGLE_WINDOW_STATIC_PERSON: f32 = 0.72;
const SINGLE_WINDOW_SLOWMO_MOTION: f32 = 1.2;
const MIN_EDITORIAL_DURATION_SECONDS: f64 = 1.25;

/// Merge adjacent same-kind windows into runs.
///
/// Two improvements over a strict equality join:
///
/// 1. **Gap tolerance** — allows merging across a rounding drift or a single
///    dropped window (gap up to 1.5× the window duration).  Without this a
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
            prev.window_count = prev.window_count.saturating_add(seg.window_count.max(1));
            prev.zoom_score = prev.zoom_score.max(seg.zoom_score);
            prev.person_confidence = match (prev.person_confidence, seg.person_confidence) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
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
            mid.kind = left.kind;
            mid.label_id = left.kind.label_id();
        }
    }
}

fn within_merge_gap(prev: &Segment, seg: &Segment) -> bool {
    let window_span = ((prev.end_seconds - prev.start_seconds)
        .max(seg.end_seconds - seg.start_seconds))
    .max(1e-3);
    let gap = seg.start_seconds - prev.end_seconds;
    // Accept small negative overlaps (rounding) and forward gaps up to 1.5×
    // the window duration so a single dropped window never breaks a run.
    gap <= 1.5 * window_span && gap >= -window_span
}

impl SegmentKind {
    pub fn label_id(self) -> u8 {
        match self {
            SegmentKind::GimbalMove => 4,
            SegmentKind::StaticSubject => 1,
            SegmentKind::SlowMotion => 5,
        }
    }
}

/// Reduce analyser runs to editorial selects for one source clip.
///
/// Short clips get one best cut. Longer clips may contribute one cut per
/// started minute, which prevents repeated overlapping timestamps while still
/// allowing long source media to surface multiple useful moments.
pub fn select_source_segments(
    source_duration_seconds: f64,
    mut segments: Vec<Segment>,
) -> Vec<Segment> {
    segments.retain(|seg| !looks_like_operator_spike(source_duration_seconds, seg));
    segments.retain(|seg| passes_editorial_confidence(seg));

    if segments.len() <= 1 {
        return segments;
    }

    segments.sort_by(|a, b| {
        segment_score(b)
            .partial_cmp(&segment_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.start_seconds
                    .partial_cmp(&b.start_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let max_cuts = if source_duration_seconds > LONG_CLIP_SECONDS {
        (source_duration_seconds / LONG_CLIP_SECONDS)
            .ceil()
            .max(1.0) as usize
    } else {
        1
    };

    let mut selected = Vec::with_capacity(max_cuts.min(segments.len()));
    let mut occupied_minutes = std::collections::HashSet::new();
    for seg in segments {
        if selected.len() >= max_cuts {
            break;
        }
        if selected.iter().any(|picked| overlaps(picked, &seg)) {
            continue;
        }
        if source_duration_seconds > LONG_CLIP_SECONDS {
            let minute = (seg.start_seconds / LONG_CLIP_SECONDS).floor().max(0.0) as u64;
            if !occupied_minutes.insert(minute) {
                continue;
            }
        }
        selected.push(seg);
    }

    selected.sort_by(|a, b| {
        a.start_seconds
            .partial_cmp(&b.start_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    selected
}

fn segment_score(seg: &Segment) -> f32 {
    let duration = (seg.end_seconds - seg.start_seconds).max(0.0) as f32;
    let person = seg.person_confidence.unwrap_or(0.0) * 2.0;
    let kind_bonus = match seg.kind {
        SegmentKind::SlowMotion => 1.1,
        SegmentKind::GimbalMove => 0.8,
        SegmentKind::StaticSubject => 0.4,
    };
    seg.motion_score.max(seg.zoom_score * 1.25).max(person) + duration.min(8.0) * 0.08 + kind_bonus
}

fn overlaps(a: &Segment, b: &Segment) -> bool {
    a.start_seconds < b.end_seconds && b.start_seconds < a.end_seconds
}

fn passes_editorial_confidence(seg: &Segment) -> bool {
    if seg.window_count >= MIN_STABLE_WINDOWS {
        return true;
    }

    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    match seg.kind {
        SegmentKind::GimbalMove => {
            duration >= MIN_EDITORIAL_DURATION_SECONDS
                && (seg.motion_score >= SINGLE_WINDOW_GIMBAL_MOTION
                    || seg.zoom_score >= SINGLE_WINDOW_GIMBAL_ZOOM)
        }
        SegmentKind::StaticSubject => {
            duration >= MIN_EDITORIAL_DURATION_SECONDS
                && seg.person_confidence.unwrap_or(0.0) >= SINGLE_WINDOW_STATIC_PERSON
        }
        SegmentKind::SlowMotion => {
            duration >= MIN_EDITORIAL_DURATION_SECONDS
                && (seg.motion_score >= SINGLE_WINDOW_SLOWMO_MOTION || seg.zoom_score >= 0.8)
        }
    }
}

fn looks_like_operator_spike(source_duration_seconds: f64, seg: &Segment) -> bool {
    if seg.kind == SegmentKind::SlowMotion || seg.person_confidence.is_some() {
        return false;
    }

    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    if duration > OPERATOR_SPIKE_MAX_SECONDS {
        return false;
    }

    let is_high_energy =
        seg.motion_score >= SPIKE_MOTION_SCORE || seg.zoom_score >= SPIKE_ZOOM_SCORE;
    if !is_high_energy {
        return false;
    }

    let touches_clip_edge = seg.start_seconds <= EDGE_SPIKE_MARGIN_SECONDS
        || source_duration_seconds - seg.end_seconds <= EDGE_SPIKE_MARGIN_SECONDS;

    touches_clip_edge || duration <= 1.25
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
            person_confidence: person,
            window_count: 1,
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
    fn short_clip_keeps_only_best_cut() {
        let p = PathBuf::from("a.mov");
        let weak = window(&p, 0.0, 5.0, SegmentKind::StaticSubject, 0.2, Some(0.45));
        let strong = window(&p, 20.0, 23.0, SegmentKind::GimbalMove, 4.0, None);

        let selected = select_source_segments(45.0, vec![weak, strong]);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].start_seconds, 20.0);
    }

    #[test]
    fn long_clip_keeps_non_overlapping_minute_selects() {
        let p = PathBuf::from("a.mov");
        let a = window(&p, 0.0, 20.0, SegmentKind::GimbalMove, 4.0, None);
        let overlap = window(&p, 10.0, 30.0, SegmentKind::GimbalMove, 5.0, None);
        let b = window(&p, 70.0, 80.0, SegmentKind::SlowMotion, 2.0, None);

        let selected = select_source_segments(130.0, vec![a, overlap, b]);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].start_seconds, 10.0);
        assert_eq!(selected[1].start_seconds, 70.0);
    }

    #[test]
    fn rejects_short_high_energy_operator_spike_at_clip_end() {
        let p = PathBuf::from("a.mov");
        let mut spike = window(&p, 8.75, 10.0, SegmentKind::GimbalMove, 7.7, None);
        spike.zoom_score = 5.98;

        let selected = select_source_segments(10.08, vec![spike]);

        assert!(selected.is_empty());
    }

    #[test]
    fn rejects_borderline_single_window_gimbal_move() {
        let p = PathBuf::from("a.mov");
        let mut move_seg = window(&p, 3.75, 8.0, SegmentKind::GimbalMove, 2.79, None);
        move_seg.zoom_score = 1.66;

        let selected = select_source_segments(8.16, vec![move_seg]);

        assert!(selected.is_empty());
    }

    #[test]
    fn keeps_single_window_static_subject_only_with_strong_person_signal() {
        let p = PathBuf::from("a.mov");
        let weak = window(&p, 2.0, 3.5, SegmentKind::StaticSubject, 0.3, Some(0.61));
        let strong = window(&p, 2.0, 3.5, SegmentKind::StaticSubject, 0.3, Some(0.82));

        assert!(select_source_segments(20.0, vec![weak]).is_empty());
        assert_eq!(select_source_segments(20.0, vec![strong]).len(), 1);
    }

    #[test]
    fn keeps_multi_window_segment_even_when_per_window_scores_are_borderline() {
        let p = PathBuf::from("a.mov");
        let a = window(&p, 0.0, 1.0, SegmentKind::GimbalMove, 1.9, None);
        let b = window(&p, 1.0, 2.0, SegmentKind::GimbalMove, 2.0, None);

        let merged = merge_segments(vec![a, b]);
        let selected = select_source_segments(20.0, merged);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].window_count, 2);
    }
}
