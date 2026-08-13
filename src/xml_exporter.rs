use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};

use crate::atomic_file;
use crate::error::{AppError, AppResult};
use crate::media::ProbeInfo;
use crate::timeline::{MovementType, Segment, SegmentKind, segment_quality_score};

struct SequenceExport<'a> {
    selected: &'a [(&'a ProbeInfo, &'a Segment)],
    timebase: u32,
    ntsc: bool,
    width: u32,
    height: u32,
    total_frames: u64,
    include_audio: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExportOptions {
    pub include_audio: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SelectionStats {
    pub total_segments: usize,
    pub duration_seconds: f64,
    pub movement_segments: usize,
    pub subject_segments: usize,
    pub slow_motion_segments: usize,
    pub static_segments: usize,
    pub audio_segments: usize,
}

#[derive(Debug, Clone)]
pub struct ExportOutcome {
    pub path: PathBuf,
    pub stats: SelectionStats,
}

fn write_project<W: Write>(w: &mut Writer<W>, export: &SequenceExport<'_>) -> AppResult<()> {
    let mut project = BytesStart::new("project");
    project.push_attribute(("id", "project-1"));
    w.write_event(Event::Start(project)).map_err(xml_err)?;
    write_text_elem(w, "name", "Video Tool Selects")?;
    w.write_event(Event::Start(BytesStart::new("children")))
        .map_err(xml_err)?;

    let mut bin = BytesStart::new("bin");
    bin.push_attribute(("id", "bin-1"));
    w.write_event(Event::Start(bin)).map_err(xml_err)?;
    write_text_elem(w, "name", "Detected Movements")?;
    w.write_event(Event::Start(BytesStart::new("children")))
        .map_err(xml_err)?;

    write_selects_sequence(w, export)?;

    w.write_event(Event::End(BytesEnd::new("children")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("bin")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("children")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("project")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_selects_sequence<W: Write>(
    w: &mut Writer<W>,
    export: &SequenceExport<'_>,
) -> AppResult<()> {
    let mut sequence = BytesStart::new("sequence");
    sequence.push_attribute(("id", "sequence-1"));
    w.write_event(Event::Start(sequence)).map_err(xml_err)?;
    write_text_elem(w, "name", "VT_Selects")?;
    write_text_elem(w, "duration", &export.total_frames.to_string())?;
    write_rate(w, export.timebase, export.ntsc)?;

    w.write_event(Event::Start(BytesStart::new("media")))
        .map_err(xml_err)?;
    w.write_event(Event::Start(BytesStart::new("video")))
        .map_err(xml_err)?;

    // <format>/<samplecharacteristics> — required by Premiere to establish the
    // sequence frame rate and pixel aspect ratio without a Translation Report.
    w.write_event(Event::Start(BytesStart::new("format")))
        .map_err(xml_err)?;
    write_samplecharacteristics(w, export.timebase, export.ntsc, export.width, export.height)?;
    w.write_event(Event::End(BytesEnd::new("format")))
        .map_err(xml_err)?;

    w.write_event(Event::Start(BytesStart::new("track")))
        .map_err(xml_err)?;

    let audio_clip_indices = export
        .selected
        .iter()
        .scan(0usize, |audio_index, (probe, _)| {
            if export.include_audio && probe.audio.is_some() {
                *audio_index += 1;
                Some(Some(*audio_index))
            } else {
                Some(None)
            }
        })
        .collect::<Vec<_>>();
    let mut timeline_cursor = 0u64;
    for (zero_based_index, (probe, seg)) in export.selected.iter().enumerate() {
        let clip_index = zero_based_index + 1;
        let file_id = format!("file-{clip_index}");
        let master_id = format!("masterclip-{clip_index}");
        let seq_start = timeline_cursor;
        let seq_end = seq_start + segment_duration_frames(seg, export.timebase, export.ntsc);
        timeline_cursor = seq_end;

        write_clipitem(
            w,
            probe,
            seg,
            clip_index,
            &file_id,
            &master_id,
            seq_start,
            seq_end,
            audio_clip_indices[zero_based_index],
            export.include_audio,
        )?;
    }

    w.write_event(Event::End(BytesEnd::new("track")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("video")))
        .map_err(xml_err)?;

    if audio_clip_indices.iter().any(Option::is_some) {
        write_audio_tracks(w, export, &audio_clip_indices)?;
    }
    w.write_event(Event::End(BytesEnd::new("media")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(xml_err)?;
    Ok(())
}

/// Write one Premiere-friendly XML project containing the highest-scoring
/// valid select from every analyzed source.
#[allow(dead_code)]
pub fn export_all(entries: &[(ProbeInfo, Vec<Segment>)], out_dir: &Path) -> AppResult<PathBuf> {
    export_all_with_options(entries, out_dir, ExportOptions::default()).map(|outcome| outcome.path)
}

pub fn export_all_with_options(
    entries: &[(ProbeInfo, Vec<Segment>)],
    out_dir: &Path,
    options: ExportOptions,
) -> AppResult<ExportOutcome> {
    let out_path = out_dir.join("analysis.premiere.xml");

    let mut selected = select_best_per_source(entries)?;
    selected.sort_by(|(probe_a, _), (probe_b, _)| probe_a.source_path.cmp(&probe_b.source_path));

    let seed = select_sequence_probe(&selected);
    let (seq_timebase, seq_ntsc) = seed.map(|p| (p.timebase, p.ntsc)).unwrap_or((25, false));
    let (seq_width, seq_height) = seed.map(|p| (p.width, p.height)).unwrap_or((1920, 1080));
    let total_frames: u64 = selected
        .iter()
        .map(|(_, seg)| segment_duration_frames(seg, seq_timebase, seq_ntsc))
        .sum();

    let mut w = Writer::new_with_indent(Vec::new(), b' ', 2);

    w.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(xml_err)?;
    w.write_event(Event::Text(BytesText::new("\n")))
        .map_err(xml_err)?;

    let mut xmeml = BytesStart::new("xmeml");
    xmeml.push_attribute(("version", "4"));
    w.write_event(Event::Start(xmeml)).map_err(xml_err)?;
    let export = SequenceExport {
        selected: &selected,
        timebase: seq_timebase,
        ntsc: seq_ntsc,
        width: seq_width,
        height: seq_height,
        total_frames,
        include_audio: options.include_audio,
    };
    write_project(&mut w, &export)?;
    w.write_event(Event::End(BytesEnd::new("xmeml")))
        .map_err(xml_err)?;
    w.write_event(Event::Text(BytesText::new("\n")))
        .map_err(xml_err)?;

    let xml = w.into_inner();
    let stats = selection_stats(&selected, options.include_audio);
    validate_generated_xml(&xml, selected.len(), stats.audio_segments)?;
    atomic_file::write_bytes(&out_path, &xml)?;
    Ok(ExportOutcome {
        path: out_path,
        stats,
    })
}

#[cfg(test)]
pub(crate) fn selection_count(entries: &[(ProbeInfo, Vec<Segment>)]) -> AppResult<usize> {
    select_best_per_source(entries).map(|selected| selected.len())
}

fn select_best_per_source(
    entries: &[(ProbeInfo, Vec<Segment>)],
) -> AppResult<Vec<(&ProbeInfo, &Segment)>> {
    let mut sources = BTreeMap::<&Path, Option<(&ProbeInfo, &Segment)>>::new();
    for (probe, segments) in entries {
        let slot = sources.entry(&probe.source_path).or_insert(None);
        for segment in segments.iter().filter(|segment| {
            segment.source_path == probe.source_path
                && valid_source_trim(probe, segment)
                && segment_duration_frames(segment, probe.timebase, probe.ntsc) > 0
        }) {
            let replace = slot
                .as_ref()
                .is_none_or(|(_, current)| selection_is_better(segment, current));
            if replace {
                *slot = Some((probe, segment));
            }
        }
    }

    if sources.is_empty() {
        return Err(AppError::Message(
            "no analyzed sources are available for XML export".to_string(),
        ));
    }
    let missing = sources
        .iter()
        .filter_map(|(path, selected)| selected.is_none().then_some(path.display().to_string()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(AppError::Message(format!(
            "refusing to write an incomplete XML; no valid selection for: {}",
            missing.join(", ")
        )));
    }

    Ok(sources.into_values().flatten().collect())
}

fn selection_stats(selected: &[(&ProbeInfo, &Segment)], include_audio: bool) -> SelectionStats {
    let mut stats = SelectionStats {
        total_segments: selected.len(),
        ..SelectionStats::default()
    };
    for (probe, segment) in selected {
        stats.duration_seconds += (segment.end_seconds - segment.start_seconds).max(0.0);
        match segment.kind {
            SegmentKind::GimbalMove => stats.movement_segments += 1,
            SegmentKind::StaticSubject => stats.subject_segments += 1,
            SegmentKind::SlowMotion => stats.slow_motion_segments += 1,
            SegmentKind::Static => stats.static_segments += 1,
        }
        if include_audio && probe.audio.is_some() {
            stats.audio_segments += 1;
        }
    }
    stats
}

fn selection_is_better(candidate: &Segment, current: &Segment) -> bool {
    segment_quality_score(candidate)
        .total_cmp(&segment_quality_score(current))
        .then_with(|| current.start_seconds.total_cmp(&candidate.start_seconds))
        .is_gt()
}

// ─── Private helpers ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn write_clipitem<W: Write>(
    w: &mut Writer<W>,
    probe: &ProbeInfo,
    seg: &Segment,
    index: usize,
    file_id: &str,
    master_id: &str,
    seq_start: u64,
    seq_end: u64,
    audio_clip_index: Option<usize>,
    include_audio: bool,
) -> AppResult<()> {
    let mut clip = BytesStart::new("clipitem");
    let clip_id = format!("clipitem-{index}");
    clip.push_attribute(("id", clip_id.as_str()));
    w.write_event(Event::Start(clip)).map_err(xml_err)?;

    let name = source_clip_name(probe);
    // FCP7 element order: name, masterclipid, enabled, duration, rate, in, out, start, end, file, labels, comments.
    write_text_elem(w, "name", &name)?;
    write_text_elem(w, "masterclipid", master_id)?;
    write_text_elem(w, "enabled", "TRUE")?;
    // Clipitem duration is the full source media duration in source-rate frames.
    write_text_elem(w, "duration", &probe.duration_frames.to_string())?;
    // Each clipitem carries its own rate so Premiere can re-interpret properly.
    write_rate(w, probe.timebase, probe.ntsc)?;

    // in/out = trim points inside the source clip (source timebase frames).
    let source_in = seg.start_frame.min(probe.duration_frames);
    let source_out = seg.end_frame.min(probe.duration_frames).max(source_in + 1);
    write_text_elem(w, "in", &source_in.to_string())?;
    write_text_elem(w, "out", &source_out.to_string())?;
    // start/end = position on the merged sequence timeline (sequence frames).
    write_text_elem(w, "start", &seq_start.to_string())?;
    write_text_elem(w, "end", &seq_end.to_string())?;

    write_file_ref(w, probe, file_id, include_audio)?;
    write_source_track(w, "video", 1)?;
    write_clip_labels(w, seg)?;
    write_clip_comments(w, seg)?;
    if let Some(audio_index) = audio_clip_index {
        write_link(w, &clip_id, "video", 1, index)?;
        write_link(
            w,
            &format!("audio-clipitem-{audio_index}"),
            "audio",
            1,
            audio_index,
        )?;
    }

    w.write_event(Event::End(BytesEnd::new("clipitem")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_audio_tracks<W: Write>(
    w: &mut Writer<W>,
    export: &SequenceExport<'_>,
    audio_clip_indices: &[Option<usize>],
) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("audio")))
        .map_err(xml_err)?;
    if let Some(audio) = export
        .selected
        .iter()
        .find_map(|(probe, _)| probe.audio.as_ref())
    {
        w.write_event(Event::Start(BytesStart::new("format")))
            .map_err(xml_err)?;
        write_audio_samplecharacteristics(w, audio.sample_rate, audio.bit_depth)?;
        w.write_event(Event::End(BytesEnd::new("format")))
            .map_err(xml_err)?;
        write_text_elem(w, "channelcount", &audio.channels.to_string())?;
    }
    w.write_event(Event::Start(BytesStart::new("track")))
        .map_err(xml_err)?;

    let mut timeline_cursor = 0u64;
    for (zero_based_index, ((probe, segment), audio_index)) in export
        .selected
        .iter()
        .zip(audio_clip_indices.iter())
        .enumerate()
    {
        let seq_start = timeline_cursor;
        let seq_end = seq_start + segment_duration_frames(segment, export.timebase, export.ntsc);
        timeline_cursor = seq_end;
        if let Some(audio_index) = audio_index {
            write_audio_clipitem(
                w,
                probe,
                segment,
                zero_based_index + 1,
                *audio_index,
                seq_start,
                seq_end,
            )?;
        }
    }

    write_text_elem(w, "enabled", "TRUE")?;
    write_text_elem(w, "locked", "FALSE")?;
    w.write_event(Event::End(BytesEnd::new("track")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("audio")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_audio_clipitem<W: Write>(
    w: &mut Writer<W>,
    probe: &ProbeInfo,
    segment: &Segment,
    video_clip_index: usize,
    audio_clip_index: usize,
    seq_start: u64,
    seq_end: u64,
) -> AppResult<()> {
    let clip_id = format!("audio-clipitem-{audio_clip_index}");
    let mut clip = BytesStart::new("clipitem");
    clip.push_attribute(("id", clip_id.as_str()));
    w.write_event(Event::Start(clip)).map_err(xml_err)?;

    write_text_elem(w, "name", &source_clip_name(probe))?;
    write_text_elem(w, "masterclipid", &format!("masterclip-{video_clip_index}"))?;
    write_text_elem(w, "enabled", "TRUE")?;
    write_text_elem(w, "duration", &probe.duration_frames.to_string())?;
    write_rate(w, probe.timebase, probe.ntsc)?;
    let source_in = segment.start_frame.min(probe.duration_frames);
    let source_out = segment
        .end_frame
        .min(probe.duration_frames)
        .max(source_in.saturating_add(1));
    write_text_elem(w, "in", &source_in.to_string())?;
    write_text_elem(w, "out", &source_out.to_string())?;
    write_text_elem(w, "start", &seq_start.to_string())?;
    write_text_elem(w, "end", &seq_end.to_string())?;

    let mut file = BytesStart::new("file");
    let file_id = format!("file-{video_clip_index}");
    file.push_attribute(("id", file_id.as_str()));
    w.write_event(Event::Empty(file)).map_err(xml_err)?;
    write_source_track(w, "audio", 1)?;
    write_link(
        w,
        &format!("clipitem-{video_clip_index}"),
        "video",
        1,
        video_clip_index,
    )?;
    write_link(w, &clip_id, "audio", 1, audio_clip_index)?;

    w.write_event(Event::End(BytesEnd::new("clipitem")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_source_track<W: Write>(
    w: &mut Writer<W>,
    media_type: &str,
    track_index: usize,
) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("sourcetrack")))
        .map_err(xml_err)?;
    write_text_elem(w, "mediatype", media_type)?;
    write_text_elem(w, "trackindex", &track_index.to_string())?;
    w.write_event(Event::End(BytesEnd::new("sourcetrack")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_link<W: Write>(
    w: &mut Writer<W>,
    clip_ref: &str,
    media_type: &str,
    track_index: usize,
    clip_index: usize,
) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("link")))
        .map_err(xml_err)?;
    write_text_elem(w, "linkclipref", clip_ref)?;
    write_text_elem(w, "mediatype", media_type)?;
    write_text_elem(w, "trackindex", &track_index.to_string())?;
    write_text_elem(w, "clipindex", &clip_index.to_string())?;
    write_text_elem(w, "groupindex", "1")?;
    w.write_event(Event::End(BytesEnd::new("link")))
        .map_err(xml_err)?;
    Ok(())
}

fn segment_duration_frames(seg: &Segment, timebase: u32, ntsc: bool) -> u64 {
    let seconds = (seg.end_seconds - seg.start_seconds).max(0.0);
    let fps = if ntsc {
        f64::from(timebase) * 1000.0 / 1001.0
    } else {
        f64::from(timebase)
    };
    (seconds * fps).round() as u64
}

fn valid_source_trim(probe: &ProbeInfo, seg: &Segment) -> bool {
    seg.start_seconds.is_finite()
        && seg.end_seconds.is_finite()
        && seg.start_seconds >= 0.0
        && seg.end_seconds > seg.start_seconds
        && seg.start_frame < probe.duration_frames
        && seg.end_frame > seg.start_frame
}

fn source_clip_name(probe: &ProbeInfo) -> String {
    probe
        .source_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("source")
        .to_string()
}

fn clip_comment(seg: &Segment) -> String {
    let kind = movement_label(seg);
    let duration = (seg.end_seconds - seg.start_seconds).max(0.0);
    format!(
        "Video Tool: {kind} | score {:.2} | source frames {}-{} | duration {:.2}s",
        segment_quality_score(seg),
        seg.start_frame,
        seg.end_frame,
        duration
    )
}

fn movement_label(seg: &Segment) -> &'static str {
    match seg.kind {
        SegmentKind::Static => "static",
        SegmentKind::StaticSubject => "static subject",
        SegmentKind::SlowMotion => "slow motion",
        SegmentKind::GimbalMove => match seg.movement_type {
            MovementType::Zoom => "zoom",
            MovementType::Roll => "roll",
            MovementType::Complex => "complex camera move",
            MovementType::PanTilt => "pan/tilt",
            MovementType::Subject => "static subject",
            MovementType::SlowMotion => "slow motion",
        },
    }
}

fn write_clip_labels<W: Write>(w: &mut Writer<W>, seg: &Segment) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("labels")))
        .map_err(xml_err)?;
    write_text_elem(w, "label2", label_color(seg))?;
    w.write_event(Event::End(BytesEnd::new("labels")))
        .map_err(xml_err)?;
    Ok(())
}

fn label_color(seg: &Segment) -> &'static str {
    match seg.kind {
        SegmentKind::Static => "Cerulean",
        SegmentKind::StaticSubject => "Caribbean",
        SegmentKind::SlowMotion => "Iris",
        SegmentKind::GimbalMove => match seg.movement_type {
            MovementType::Zoom => "Mango",
            MovementType::Roll => "Lavender",
            MovementType::Complex => "Rose",
            MovementType::PanTilt => "Forest",
            MovementType::Subject => "Caribbean",
            MovementType::SlowMotion => "Iris",
        },
    }
}

fn select_sequence_probe<'a>(selected: &[(&'a ProbeInfo, &Segment)]) -> Option<&'a ProbeInfo> {
    let mut rates = BTreeMap::<(u32, bool), (usize, &ProbeInfo)>::new();
    for (probe, _) in selected {
        let entry = rates
            .entry((probe.timebase, probe.ntsc))
            .or_insert((0, probe));
        entry.0 += 1;
        if entry.1.slow_motion && !probe.slow_motion {
            entry.1 = probe;
        }
    }
    rates
        .into_values()
        .max_by(|(count_a, probe_a), (count_b, probe_b)| {
            count_a
                .cmp(count_b)
                .then_with(|| (probe_a.timebase <= 60).cmp(&(probe_b.timebase <= 60)))
                .then_with(|| (!probe_a.slow_motion).cmp(&!probe_b.slow_motion))
                .then_with(|| probe_b.timebase.cmp(&probe_a.timebase))
        })
        .map(|(_, probe)| probe)
}

fn write_file_ref<W: Write>(
    w: &mut Writer<W>,
    probe: &ProbeInfo,
    file_id: &str,
    include_audio: bool,
) -> AppResult<()> {
    let mut file = BytesStart::new("file");
    file.push_attribute(("id", file_id));

    w.write_event(Event::Start(file)).map_err(xml_err)?;

    let name = probe
        .source_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("source");
    write_text_elem(w, "name", name)?;
    write_text_elem(w, "pathurl", &path_to_url(&probe.source_path))?;
    write_rate(w, probe.timebase, probe.ntsc)?;
    write_text_elem(w, "duration", &probe.duration_frames.to_string())?;

    // <media>/<video>/<samplecharacteristics> so Premiere can resolve the
    // source without a Media Offline warning or Translation Report entry.
    w.write_event(Event::Start(BytesStart::new("media")))
        .map_err(xml_err)?;
    w.write_event(Event::Start(BytesStart::new("video")))
        .map_err(xml_err)?;
    write_text_elem(w, "duration", &probe.duration_frames.to_string())?;
    write_samplecharacteristics(w, probe.timebase, probe.ntsc, probe.width, probe.height)?;
    w.write_event(Event::End(BytesEnd::new("video")))
        .map_err(xml_err)?;
    if include_audio && let Some(audio) = &probe.audio {
        w.write_event(Event::Start(BytesStart::new("audio")))
            .map_err(xml_err)?;
        write_audio_samplecharacteristics(w, audio.sample_rate, audio.bit_depth)?;
        write_text_elem(w, "channelcount", &audio.channels.to_string())?;
        w.write_event(Event::End(BytesEnd::new("audio")))
            .map_err(xml_err)?;
    }
    w.write_event(Event::End(BytesEnd::new("media")))
        .map_err(xml_err)?;

    w.write_event(Event::End(BytesEnd::new("file")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_clip_comments<W: Write>(w: &mut Writer<W>, segment: &Segment) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("comments")))
        .map_err(xml_err)?;
    write_text_elem(w, "clipcommenta", &clip_comment(segment))?;
    w.write_event(Event::End(BytesEnd::new("comments")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_audio_samplecharacteristics<W: Write>(
    w: &mut Writer<W>,
    sample_rate: u32,
    bit_depth: u32,
) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("samplecharacteristics")))
        .map_err(xml_err)?;
    write_text_elem(w, "depth", &bit_depth.to_string())?;
    write_text_elem(w, "samplerate", &sample_rate.to_string())?;
    w.write_event(Event::End(BytesEnd::new("samplecharacteristics")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_samplecharacteristics<W: Write>(
    w: &mut Writer<W>,
    timebase: u32,
    ntsc: bool,
    width: u32,
    height: u32,
) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("samplecharacteristics")))
        .map_err(xml_err)?;
    write_rate(w, timebase, ntsc)?;
    write_text_elem(w, "width", &width.to_string())?;
    write_text_elem(w, "height", &height.to_string())?;
    write_text_elem(w, "anamorphic", "FALSE")?;
    write_text_elem(w, "pixelaspectratio", "square")?;
    write_text_elem(w, "fielddominance", "none")?;
    w.write_event(Event::End(BytesEnd::new("samplecharacteristics")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_rate<W: Write>(w: &mut Writer<W>, timebase: u32, ntsc: bool) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new("rate")))
        .map_err(xml_err)?;
    write_text_elem(w, "timebase", &timebase.to_string())?;
    write_text_elem(w, "ntsc", if ntsc { "TRUE" } else { "FALSE" })?;
    w.write_event(Event::End(BytesEnd::new("rate")))
        .map_err(xml_err)?;
    Ok(())
}

fn write_text_elem<W: Write>(w: &mut Writer<W>, name: &str, value: &str) -> AppResult<()> {
    w.write_event(Event::Start(BytesStart::new(name)))
        .map_err(xml_err)?;
    w.write_event(Event::Text(BytesText::new(value)))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new(name)))
        .map_err(xml_err)?;
    Ok(())
}

fn path_to_url(path: &Path) -> String {
    let mut p = path.to_string_lossy().replace('\\', "/");
    if let Some(stripped) = p.strip_prefix("//?/UNC/") {
        p = format!("//{stripped}");
    }
    if let Some(stripped) = p.strip_prefix("//?/") {
        p = stripped.to_string();
    }
    if let Some(unc) = p.strip_prefix("//") {
        let mut parts = unc.split('/').filter(|part| !part.is_empty());
        let Some(server) = parts.next() else {
            return "file:///".to_string();
        };
        let encoded = parts
            .map(|part| urlencoding::encode(part).into_owned())
            .collect::<Vec<_>>()
            .join("/");
        return format!("file://{server}/{encoded}");
    }
    let is_posix_absolute = p.starts_with('/') && !p.starts_with("//");
    if is_posix_absolute {
        p = p.trim_start_matches('/').to_string();
    }
    let parts: Vec<&str> = p.split('/').filter(|part| !part.is_empty()).collect();
    let mut encoded: Vec<String> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        if i == 0 && part.ends_with(':') {
            encoded.push((*part).to_string()); // drive letter, keep colon
        } else {
            encoded.push(urlencoding::encode(part).into_owned());
        }
    }
    format!("file://localhost/{}", encoded.join("/"))
}

fn validate_generated_xml(
    xml: &[u8],
    expected_video_clips: usize,
    expected_audio_clips: usize,
) -> AppResult<()> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(true);
    let mut root_seen = false;
    let mut video_clip_items = 0usize;
    let mut audio_clip_items = 0usize;
    let mut path_urls = 0usize;
    let mut link_refs = 0usize;
    let mut element_stack = Vec::<Vec<u8>>::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) => {
                let name = event.name().as_ref().to_vec();
                match name.as_slice() {
                    b"xmeml" => root_seen = true,
                    b"clipitem" => {
                        let is_audio = event.attributes().flatten().any(|attribute| {
                            attribute.key.as_ref() == b"id"
                                && attribute.value.as_ref().starts_with(b"audio-clipitem-")
                        });
                        if is_audio {
                            audio_clip_items += 1;
                        } else {
                            video_clip_items += 1;
                        }
                    }
                    b"pathurl" => path_urls += 1,
                    b"linkclipref" => link_refs += 1,
                    b"channelcount"
                        if element_stack.last().map(Vec::as_slice) != Some(b"audio") =>
                    {
                        return Err(AppError::Message(
                            "generated Premiere XML placed channelcount outside audio".to_string(),
                        ));
                    }
                    _ => {}
                }
                element_stack.push(name);
            }
            Ok(Event::End(_)) => {
                element_stack.pop();
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(error) => {
                return Err(AppError::Message(format!(
                    "generated Premiere XML is not well formed: {error}"
                )));
            }
        }
    }
    if !root_seen
        || video_clip_items != expected_video_clips
        || audio_clip_items != expected_audio_clips
        || path_urls != expected_video_clips
        || link_refs != expected_audio_clips.saturating_mul(4)
    {
        return Err(AppError::Message(format!(
            "generated Premiere XML failed validation (video={video_clip_items}/{expected_video_clips}, audio={audio_clip_items}/{expected_audio_clips}, files={path_urls}/{expected_video_clips}, links={link_refs}/{})",
            expected_audio_clips.saturating_mul(4)
        )));
    }
    Ok(())
}

fn xml_err(e: quick_xml::Error) -> AppError {
    AppError::Message(e.to_string())
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::SegmentKind;

    fn sample_probe(name: &str) -> ProbeInfo {
        ProbeInfo {
            source_path: PathBuf::from(format!("C:/vids/{name}")),
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

    fn sample_segment(kind: SegmentKind, start_f: u64, end_f: u64) -> Segment {
        Segment {
            source_path: PathBuf::from("C:/vids/a.mov"),
            start_frame: start_f,
            end_frame: end_f,
            start_seconds: start_f as f64 / 25.0,
            end_seconds: end_f as f64 / 25.0,
            kind,
            label_id: kind.label_id(),
            motion_score: 0.0,
            zoom_score: 0.0,
            movement_type: MovementType::PanTilt,
            motion_confidence: 0.88,
            motion_smoothness: 0.88,
            person_confidence: None,
            window_count: 1,
            cinematic_score: 0.0,
        }
    }

    fn sample_segment_for(name: &str, kind: SegmentKind, start_f: u64, end_f: u64) -> Segment {
        let mut segment = sample_segment(kind, start_f, end_f);
        segment.source_path = PathBuf::from(format!("C:/vids/{name}"));
        segment
    }

    #[test]
    fn url_encoding() {
        let p = PathBuf::from(r"C:\My Videos\clip 1.mov");
        let u = path_to_url(&p);
        assert!(u.starts_with("file://localhost/"));
        assert!(u.contains("C:/"));
        assert!(u.contains("My%20Videos"));
        assert!(u.contains("clip%201.mov"));
    }

    #[test]
    fn posix_absolute_paths_do_not_gain_an_extra_root_component() {
        let u = path_to_url(Path::new("/mnt/media/clip 1.mov"));
        assert_eq!(u, "file://localhost/mnt/media/clip%201.mov");
    }

    #[test]
    fn unc_paths_keep_the_server_as_the_file_uri_authority() {
        let u = path_to_url(Path::new(r"\\edit-server\wedding media\clip 1.mov"));
        assert_eq!(u, "file://edit-server/wedding%20media/clip%201.mov");
    }

    #[test]
    fn ntsc_sequence_duration_uses_the_rational_rate() {
        let seg = sample_segment(SegmentKind::GimbalMove, 0, 15000);
        assert_eq!(segment_duration_frames(&seg, 30, true), 17982);
    }

    #[test]
    fn export_writes_expected_premiere_structure() {
        let tmp = std::env::temp_dir().join("video_tool_xml_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("A Cam 001.mov");
        let segs = vec![
            sample_segment_for("A Cam 001.mov", SegmentKind::GimbalMove, 0, 25),
            sample_segment_for("A Cam 001.mov", SegmentKind::StaticSubject, 25, 50),
        ];
        let out = export_all(&[(probe, segs)], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        // Sequence format block present with correct rate and pixel aspect.
        assert!(xml.contains("<project id=\"project-1\">"));
        assert!(xml.contains("<name>Video Tool Selects</name>"));
        assert!(xml.contains("<bin id=\"bin-1\">"));
        assert!(xml.contains("<name>Detected Movements</name>"));
        assert!(xml.contains("<name>VT_Selects</name>"));
        assert!(xml.contains("<format>"));
        assert!(xml.contains("<pixelaspectratio>square</pixelaspectratio>"));
        assert!(xml.contains("<timebase>25</timebase>"));
        assert!(xml.contains("<fielddominance>none</fielddominance>"));

        // Only the highest-quality candidate is exported.
        assert_eq!(xml.matches("<clipitem id=").count(), 1);
        assert!(xml.contains("<in>25</in>"));
        assert!(xml.contains("<out>50</out>"));
        assert!(xml.contains("<start>0</start>"));
        assert!(xml.contains("<end>25</end>"));
        // The selected clipitem plus the shared source file record keep the source name.
        assert_eq!(xml.matches("<name>A Cam 001.mov</name>").count(), 2);
        assert!(!xml.contains("_M01_"));
        assert!(!xml.contains("_P02_"));
        assert!(xml.contains("<clipcommenta>Video Tool: static subject | score "));
        assert!(xml.contains("| source frames 25-50 | duration 1.00s</clipcommenta>"));
        assert!(xml.contains("<labels>"));
        assert!(xml.contains("<label2>Caribbean</label2>"));
        assert!(xml.contains("<clipitem id=\"clipitem-1\">"));
        assert_eq!(
            xml.matches("<masterclipid>masterclip-1</masterclipid>")
                .count(),
            1
        );
        assert!(!xml.contains("<masterclipid>masterclip-2</masterclipid>"));

        // The single selected clip emits one complete source file record.
        assert_eq!(xml.matches("<pathurl>").count(), 1);
        assert!(xml.contains("<file id=\"file-1\">"));
        assert!(!xml.contains("<audio>"));
    }

    #[test]
    fn optional_audio_export_writes_linked_source_audio() {
        let tmp = std::env::temp_dir().join("video_tool_xml_audio_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let mut probe = sample_probe("A Cam 001.mov");
        probe.audio = Some(crate::media::AudioInfo {
            stream_index: 1,
            channels: 2,
            sample_rate: 48_000,
            bit_depth: 16,
        });
        let segment = sample_segment_for("A Cam 001.mov", SegmentKind::GimbalMove, 25, 75);
        let outcome = export_all_with_options(
            &[(probe, vec![segment])],
            &tmp,
            ExportOptions {
                include_audio: true,
            },
        )
        .unwrap();
        let xml = std::fs::read_to_string(outcome.path).unwrap();

        assert_eq!(outcome.stats.audio_segments, 1);
        assert!(xml.contains("<clipitem id=\"audio-clipitem-1\">"));
        assert!(xml.contains("<linkclipref>audio-clipitem-1</linkclipref>"));
        assert!(xml.contains("<mediatype>audio</mediatype>"));
        assert!(xml.contains("<samplerate>48000</samplerate>"));
        assert!(xml.contains("<channelcount>2</channelcount>"));
        assert_eq!(xml.matches("<pathurl>").count(), 1);
    }

    #[test]
    fn validation_rejects_channelcount_inside_sample_characteristics() {
        let malformed = br#"<xmeml><audio><samplecharacteristics><channelcount>2</channelcount></samplecharacteristics></audio></xmeml>"#;
        let error = validate_generated_xml(malformed, 0, 0).unwrap_err();
        assert!(error.to_string().contains("channelcount outside audio"));
    }

    #[test]
    fn export_keeps_one_best_selection_from_every_source() {
        let tmp = std::env::temp_dir().join("video_tool_xml_per_source_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let entries = vec![
            (
                sample_probe("A.mov"),
                vec![
                    sample_segment_for("A.mov", SegmentKind::GimbalMove, 0, 25),
                    sample_segment_for("A.mov", SegmentKind::StaticSubject, 25, 50),
                ],
            ),
            (
                sample_probe("B.mov"),
                vec![
                    sample_segment_for("B.mov", SegmentKind::GimbalMove, 0, 40),
                    sample_segment_for("B.mov", SegmentKind::StaticSubject, 50, 75),
                ],
            ),
        ];

        assert_eq!(selection_count(&entries).unwrap(), 2);
        let out = export_all(&entries, &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert_eq!(xml.matches("<clipitem id=").count(), 2);
        assert_eq!(xml.matches("<pathurl>").count(), 2);
        assert!(xml.contains("<name>A.mov</name>"));
        assert!(xml.contains("<name>B.mov</name>"));
        assert!(xml.contains("<clipitem id=\"clipitem-1\">"));
        assert!(xml.contains("<clipitem id=\"clipitem-2\">"));
    }

    #[test]
    fn duplicate_source_entries_are_deduplicated() {
        let tmp = std::env::temp_dir().join("video_tool_xml_duplicate_source_test");
        std::fs::create_dir_all(&tmp).unwrap();
        let entries = vec![
            (
                sample_probe("A.mov"),
                vec![sample_segment_for("A.mov", SegmentKind::GimbalMove, 0, 25)],
            ),
            (
                sample_probe("A.mov"),
                vec![sample_segment_for(
                    "A.mov",
                    SegmentKind::StaticSubject,
                    25,
                    75,
                )],
            ),
        ];

        assert_eq!(selection_count(&entries).unwrap(), 1);
        let xml = std::fs::read_to_string(export_all(&entries, &tmp).unwrap()).unwrap();
        assert_eq!(xml.matches("<clipitem id=").count(), 1);
        assert_eq!(xml.matches("<pathurl>").count(), 1);
        assert!(xml.contains("<in>25</in>"));
    }

    #[test]
    fn sequence_uses_the_most_common_source_rate() {
        let tmp = std::env::temp_dir().join("video_tool_xml_sequence_rate_test");
        std::fs::create_dir_all(&tmp).unwrap();
        let a = sample_probe("A.mov");
        let b = sample_probe("B.mov");
        let mut slow = sample_probe("C.mov");
        slow.timebase = 100;
        slow.fps_num = 100;
        slow.duration_frames = 400;
        slow.slow_motion = true;
        let entries = vec![
            (
                a,
                vec![sample_segment_for("A.mov", SegmentKind::GimbalMove, 0, 25)],
            ),
            (
                b,
                vec![sample_segment_for("B.mov", SegmentKind::GimbalMove, 0, 25)],
            ),
            (
                slow,
                vec![sample_segment_for("C.mov", SegmentKind::SlowMotion, 0, 100)],
            ),
        ];

        let xml = std::fs::read_to_string(export_all(&entries, &tmp).unwrap()).unwrap();
        let sequence_prefix = xml.split("<media>").next().unwrap_or_default();
        assert!(sequence_prefix.contains("<timebase>25</timebase>"));
    }

    #[test]
    fn export_chooses_one_best_label() {
        let tmp = std::env::temp_dir().join("video_tool_xml_color_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("Original Clip Name.mov");
        let mut pan = sample_segment_for("Original Clip Name.mov", SegmentKind::GimbalMove, 0, 10);
        let mut zoom =
            sample_segment_for("Original Clip Name.mov", SegmentKind::GimbalMove, 10, 20);
        let mut roll =
            sample_segment_for("Original Clip Name.mov", SegmentKind::GimbalMove, 20, 30);
        let mut complex =
            sample_segment_for("Original Clip Name.mov", SegmentKind::GimbalMove, 30, 40);
        let subject =
            sample_segment_for("Original Clip Name.mov", SegmentKind::StaticSubject, 40, 50);
        let slow = sample_segment_for("Original Clip Name.mov", SegmentKind::SlowMotion, 50, 60);

        pan.movement_type = MovementType::PanTilt;
        zoom.movement_type = MovementType::Zoom;
        roll.movement_type = MovementType::Roll;
        complex.movement_type = MovementType::Complex;

        let out = export_all(
            &[(probe, vec![pan, zoom, roll, complex, subject, slow])],
            &tmp,
        )
        .unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert_eq!(xml.matches("<clipitem id=").count(), 1);
        assert_eq!(
            xml.matches("<name>Original Clip Name.mov</name>").count(),
            2
        );
        assert!(!xml.contains("Original Clip Name_"));
        assert!(xml.contains("<label2>Iris</label2>"));
        assert!(!xml.contains("<label2>Forest</label2>"));
        assert!(!xml.contains("<label2>Mango</label2>"));
    }

    #[test]
    fn static_clip_uses_explicit_cerulean_label() {
        let tmp = std::env::temp_dir().join("video_tool_xml_static_default_label_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("Static Clip.mov");
        let seg = sample_segment_for("Static Clip.mov", SegmentKind::Static, 0, 100);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(xml.contains("<clipitem id=\"clipitem-1\">"));
        assert!(xml.contains("Video Tool: static"));
        assert!(xml.contains("<labels>"));
        assert!(xml.contains("<label2>Cerulean</label2>"));
    }

    #[test]
    fn segment_kind_takes_precedence_when_choosing_color() {
        let mut slow = sample_segment(SegmentKind::SlowMotion, 0, 25);
        slow.movement_type = MovementType::Subject;
        assert_eq!(label_color(&slow), "Iris");
        assert_eq!(movement_label(&slow), "slow motion");
    }

    #[test]
    fn export_clamps_source_out_to_media_duration() {
        let tmp = std::env::temp_dir().join("video_tool_xml_clamp_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let mut probe = sample_probe("tail.mov");
        probe.duration_frames = 100;
        probe.duration_seconds = 4.0;
        let seg = sample_segment_for("tail.mov", SegmentKind::GimbalMove, 95, 130);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(xml.contains("<in>95</in>"));
        assert!(xml.contains("<out>100</out>"));
    }

    #[test]
    fn export_rejects_zero_length_segments_without_overwriting_previous_xml() {
        let tmp = std::env::temp_dir().join("video_tool_xml_zero_test");
        std::fs::create_dir_all(&tmp).unwrap();
        let out = tmp.join("analysis.premiere.xml");
        std::fs::write(&out, b"previous-good-xml").unwrap();

        let probe = sample_probe("zero.mov");
        let seg = sample_segment_for("zero.mov", SegmentKind::GimbalMove, 25, 25);

        let error = export_all(&[(probe, vec![seg])], &tmp).unwrap_err();
        assert!(error.to_string().contains("incomplete XML"));
        assert_eq!(std::fs::read(&out).unwrap(), b"previous-good-xml");
    }

    #[test]
    fn export_rejects_segments_that_start_after_media_end() {
        let tmp = std::env::temp_dir().join("video_tool_xml_after_end_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let mut probe = sample_probe("after.mov");
        probe.duration_frames = 100;
        let seg = sample_segment_for("after.mov", SegmentKind::GimbalMove, 120, 140);

        assert!(export_all(&[(probe, vec![seg])], &tmp).is_err());
    }
}
