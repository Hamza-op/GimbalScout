use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use quick_xml::Writer;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};

use crate::error::{AppError, AppResult};
use crate::media::ProbeInfo;
use crate::timeline::{MovementType, Segment, SegmentKind, segment_quality_score};

struct SequenceExport<'a> {
    entries: &'a [(ProbeInfo, Vec<Segment>)],
    selected: &'a [(&'a ProbeInfo, &'a Segment)],
    timebase: u32,
    ntsc: bool,
    width: u32,
    height: u32,
    total_frames: u64,
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

    // Assign one stable file-id per unique source path so multiple segments
    // from the same clip share a single <file> record.
    let mut file_ids: HashMap<PathBuf, String> = HashMap::new();
    let mut master_ids: HashMap<PathBuf, String> = HashMap::new();
    let mut next_file_index = 1usize;
    let mut next_master_index = 1usize;
    for (probe, _) in export.entries {
        file_ids
            .entry(probe.source_path.clone())
            .or_insert_with(|| {
                let id = format!("file-{next_file_index}");
                next_file_index += 1;
                id
            });
        master_ids
            .entry(probe.source_path.clone())
            .or_insert_with(|| {
                let id = format!("masterclip-{next_master_index}");
                next_master_index += 1;
                id
            });
    }
    let mut emitted_files: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut timeline_cursor = 0u64;
    for (zero_based_index, (probe, seg)) in export.selected.iter().enumerate() {
        let clip_index = zero_based_index + 1;
        let file_id = file_ids
            .get(&probe.source_path)
            .expect("file id inserted above")
            .clone();
        let master_id = master_ids
            .get(&probe.source_path)
            .expect("master id inserted above")
            .clone();
        let seq_start = timeline_cursor;
        let seq_end = seq_start + segment_duration_frames(seg, export.timebase, export.ntsc);
        timeline_cursor = seq_end;

        let is_first = emitted_files.insert(file_id.clone());
        write_clipitem(
            w, probe, seg, clip_index, &file_id, &master_id, is_first, seq_start, seq_end,
        )?;
    }

    w.write_event(Event::End(BytesEnd::new("track")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("video")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("media")))
        .map_err(xml_err)?;
    w.write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(xml_err)?;
    Ok(())
}

/// Write one Premiere-friendly XML project containing only the highest-scoring
/// valid select across all analyzed sources.
pub fn export_all(entries: &[(ProbeInfo, Vec<Segment>)], out_dir: &Path) -> AppResult<PathBuf> {
    let out_path = out_dir.join("analysis.premiere.xml");

    let mut selected = Vec::new();
    for (probe, segments) in entries {
        for seg in segments {
            if valid_source_trim(probe, seg)
                && segment_duration_frames(seg, probe.timebase, probe.ntsc) > 0
            {
                selected.push((probe, seg));
            }
        }
    }
    selected.sort_by(|(probe_a, seg_a), (probe_b, seg_b)| {
        segment_quality_score(seg_b)
            .partial_cmp(&segment_quality_score(seg_a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| probe_a.source_path.cmp(&probe_b.source_path))
            .then_with(|| {
                seg_a
                    .start_seconds
                    .partial_cmp(&seg_b.start_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    selected.truncate(1);

    let seed = selected
        .first()
        .map(|(probe, _)| *probe)
        .or_else(|| select_sequence_probe(entries));
    let (seq_timebase, seq_ntsc) = seed.map(|p| (p.timebase, p.ntsc)).unwrap_or((25, false));
    let (seq_width, seq_height) = seed.map(|p| (p.width, p.height)).unwrap_or((1920, 1080));
    let total_frames: u64 = selected
        .iter()
        .map(|(_, seg)| segment_duration_frames(seg, seq_timebase, seq_ntsc))
        .sum();

    let file = File::create(&out_path).map_err(|e| AppError::Io {
        path: out_path.clone(),
        source: e,
    })?;
    let mut w = Writer::new_with_indent(BufWriter::new(file), b' ', 2);

    w.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(xml_err)?;
    w.write_event(Event::Text(BytesText::new("\n")))
        .map_err(xml_err)?;

    let mut xmeml = BytesStart::new("xmeml");
    xmeml.push_attribute(("version", "4"));
    w.write_event(Event::Start(xmeml)).map_err(xml_err)?;
    let export = SequenceExport {
        entries,
        selected: &selected,
        timebase: seq_timebase,
        ntsc: seq_ntsc,
        width: seq_width,
        height: seq_height,
        total_frames,
    };
    write_project(&mut w, &export)?;
    w.write_event(Event::End(BytesEnd::new("xmeml")))
        .map_err(xml_err)?;
    w.write_event(Event::Text(BytesText::new("\n")))
        .map_err(xml_err)?;

    Ok(out_path)
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
    file_is_first: bool,
    seq_start: u64,
    seq_end: u64,
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

    write_file_ref(w, probe, file_id, file_is_first)?;
    write_clip_labels(w, seg)?;
    write_text_elem(w, "comments", &clip_comment(seg))?;

    w.write_event(Event::End(BytesEnd::new("clipitem")))
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
    seg.start_frame < probe.duration_frames && seg.end_frame > seg.start_frame
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
    match (seg.kind, seg.movement_type) {
        (SegmentKind::Static, _) => "static",
        (SegmentKind::StaticSubject, _) | (_, MovementType::Subject) => "static subject",
        (SegmentKind::SlowMotion, _) | (_, MovementType::SlowMotion) => "slow motion",
        (_, MovementType::Zoom) => "zoom",
        (_, MovementType::Roll) => "roll",
        (_, MovementType::Complex) => "complex camera move",
        (_, MovementType::PanTilt) => "pan/tilt",
    }
}

fn write_clip_labels<W: Write>(w: &mut Writer<W>, seg: &Segment) -> AppResult<()> {
    if seg.kind == SegmentKind::Static {
        return Ok(());
    }

    w.write_event(Event::Start(BytesStart::new("labels")))
        .map_err(xml_err)?;
    write_text_elem(w, "label2", label_color(seg))?;
    w.write_event(Event::End(BytesEnd::new("labels")))
        .map_err(xml_err)?;
    Ok(())
}

fn label_color(seg: &Segment) -> &'static str {
    match (seg.kind, seg.movement_type) {
        (SegmentKind::StaticSubject, _) | (_, MovementType::Subject) => "Caribbean",
        (SegmentKind::SlowMotion, _) | (_, MovementType::SlowMotion) => "Iris",
        (_, MovementType::Zoom) => "Mango",
        (_, MovementType::Roll) => "Lavender",
        (_, MovementType::Complex) => "Rose",
        (_, MovementType::PanTilt) => "Forest",
    }
}

fn select_sequence_probe(entries: &[(ProbeInfo, Vec<Segment>)]) -> Option<&ProbeInfo> {
    entries
        .iter()
        .filter(|(_, segs)| !segs.is_empty())
        .map(|(probe, _)| probe)
        .min_by_key(|probe| {
            let low_rate_penalty = if probe.timebase <= 60 { 0 } else { 1 };
            let slow_bonus = if probe.slow_motion { 0 } else { 1 };
            (low_rate_penalty, slow_bonus, probe.timebase)
        })
}

fn write_file_ref<W: Write>(
    w: &mut Writer<W>,
    probe: &ProbeInfo,
    file_id: &str,
    is_first: bool,
) -> AppResult<()> {
    let mut file = BytesStart::new("file");
    file.push_attribute(("id", file_id));

    // Subsequent references to the same source use a self-closing <file id="…"/>.
    if !is_first {
        w.write_event(Event::Empty(file)).map_err(xml_err)?;
        return Ok(());
    }

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
    w.write_event(Event::End(BytesEnd::new("media")))
        .map_err(xml_err)?;

    w.write_event(Event::End(BytesEnd::new("file")))
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
    if let Some(stripped) = p.strip_prefix("//?/") {
        p = stripped.to_string();
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
            sample_segment(SegmentKind::GimbalMove, 0, 25),
            sample_segment(SegmentKind::StaticSubject, 25, 50),
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
        assert!(xml.contains("<comments>Video Tool: static subject | score "));
        assert!(xml.contains("| source frames 25-50 | duration 1.00s</comments>"));
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
    }

    #[test]
    fn export_chooses_one_best_label() {
        let tmp = std::env::temp_dir().join("video_tool_xml_color_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("Original Clip Name.mov");
        let mut pan = sample_segment(SegmentKind::GimbalMove, 0, 10);
        let mut zoom = sample_segment(SegmentKind::GimbalMove, 10, 20);
        let mut roll = sample_segment(SegmentKind::GimbalMove, 20, 30);
        let mut complex = sample_segment(SegmentKind::GimbalMove, 30, 40);
        let subject = sample_segment(SegmentKind::StaticSubject, 40, 50);
        let slow = sample_segment(SegmentKind::SlowMotion, 50, 60);

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
    fn static_clip_uses_premiere_default_label() {
        let tmp = std::env::temp_dir().join("video_tool_xml_static_default_label_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("Static Clip.mov");
        let seg = sample_segment(SegmentKind::Static, 0, 100);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(xml.contains("<clipitem id=\"clipitem-1\">"));
        assert!(xml.contains("Video Tool: static"));
        assert!(!xml.contains("<labels>"));
        assert!(!xml.contains("<label2>"));
    }

    #[test]
    fn export_clamps_source_out_to_media_duration() {
        let tmp = std::env::temp_dir().join("video_tool_xml_clamp_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let mut probe = sample_probe("tail.mov");
        probe.duration_frames = 100;
        probe.duration_seconds = 4.0;
        let seg = sample_segment(SegmentKind::GimbalMove, 95, 130);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(xml.contains("<in>95</in>"));
        assert!(xml.contains("<out>100</out>"));
    }

    #[test]
    fn export_skips_zero_length_segments() {
        let tmp = std::env::temp_dir().join("video_tool_xml_zero_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let probe = sample_probe("zero.mov");
        let seg = sample_segment(SegmentKind::GimbalMove, 25, 25);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(!xml.contains("<clipitem id=\"clipitem-1\">"));
        assert!(xml.contains("<duration>0</duration>"));
    }

    #[test]
    fn export_skips_segments_that_start_after_media_end() {
        let tmp = std::env::temp_dir().join("video_tool_xml_after_end_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let mut probe = sample_probe("after.mov");
        probe.duration_frames = 100;
        let seg = sample_segment(SegmentKind::GimbalMove, 120, 140);

        let out = export_all(&[(probe, vec![seg])], &tmp).unwrap();
        let xml = std::fs::read_to_string(&out).unwrap();

        assert!(!xml.contains("<clipitem id=\"clipitem-1\">"));
    }
}
