# Engineering review

This review covers the analysis pipeline, model integration, timeline
selection, cache/discovery layers, Premiere XML export, persistence, release
configuration, and AMD/Intel CPU portability.

## Architecture assessment

The existing motion architecture remains appropriate for this application. A
small grayscale frame is sampled at fixed grid points, textured patches are
tracked, and a robust affine fit separates global camera motion from local
subject motion. The affine model captures pan, tilt, zoom, roll, and modest
shear without the instability and parallax overfitting of a full homography.
Deterministic seed selection and bounded RANSAC also make its cost predictable
on CPUs.

The compact bundled YOLO ONNX model was retained. Replacing it without a
labeled, representative footage set would trade a known working model for an
unmeasured one. The integration now handles both raw detector tensors and
six-column post-NMS tensors correctly, and the bundled model has an opt-in
inference smoke test. A future model replacement should be gated by precision,
recall, latency, and memory measurements on real wedding footage.

## High-impact changes

### Analysis and performance

- Analysis windows now use a half-window stride, matching the documented
  behavior and reducing missed movement or subject events at window boundaries.
- Movement-only mode decodes directly to a maximum height of 144 pixels instead
  of decoding at 720 pixels and immediately discarding most pixels. For 16:9
  footage this reduces the raw grayscale pipe from roughly 922 KB to 37 KB per
  sampled frame, about 25 times less data.
- YOLO letterbox coordinate maps are computed once per source geometry and
  reused for subsequent inference. Per-pixel floating-point coordinate mapping
  is no longer repeated for every detector sample.
- Detector input sizes are checked for overflow and truncation before indexing.
- Six-column post-NMS output now treats column 5 as a class id and column 4 as
  confidence. Previously a valid person class id of zero could zero out the
  score.
- Existing worker/thread budgeting was retained. It derives parallelism from
  logical CPU availability, bounds file workers, and divides the budget between
  FFmpeg and ONNX Runtime to avoid severe oversubscription.
- A single-pass thumbnail quality estimator now measures exposure balance,
  clipped highlights/shadows, tonal separation, and edge detail using integer
  accumulators. It reuses the existing 144px motion thumbnail, so no additional
  decode stream or full-resolution frame is required.
- When no motion or subject event survives, the best technically usable window
  becomes a clearly colored static fallback. The old behavior exported the
  entire source clip, which could place tens of seconds of rejected footage on
  the selects timeline.
- Long merged runs are focused on their highest-quality original window and
  capped at the editor-selected duration (8 seconds by default). Short events
  receive up to one second of source context on each side.
- Camera-model match quality is measured only on robust affine inliers. Moving
  foreground subjects still reduce spatial support once, but no longer suppress
  a valid background pan/tilt a second time through their rejected patch SAD.
- Window movement labels accumulate confidence-weighted temporal evidence, so a
  single false zoom/roll estimate cannot outvote a sustained pan or tilt.
- Automatic thresholds retain the normal conservative floor for static/mixed
  footage, lower it for uniformly slow coherent moves, and keep the existing
  upper cap for fast continuous movement.

### Cache and discovery

- Analysis cache schema is now version 17. Sidecars retain raw overlapping
  analysis windows rather than already-trimmed editorial output. Changing the
  select duration or audio-export choice now rebuilds XML from cache without
  re-decoding footage or rerunning YOLO. Older sidecars are intentionally
  ignored once so they can be rebuilt with image-quality evidence and audio
  metadata.
- Cache entries are structurally validated before writing and after reading.
  Empty results, invalid time spans, impossible probe values, mismatched source
  paths, malformed sidecars, and stale existing files are rejected.
- Recovery export ignores `discovery-cache.json`, deduplicates repeated source
  entries deterministically, and preserves valid entries for media that is
  temporarily offline.
- Model identity now uses a streaming full-content SHA-256 digest. The cache is
  reused if an identical model moves to another path and invalidated if model
  bytes change even when size and timestamps do not.
- Execution-provider details were removed from the semantic fingerprint because
  CPU, DirectML, CUDA, and other providers should produce the same selections.
- Nested media discovery no longer trusts a parent's cached `all_matches` list.
  It walks cached directory metadata at every level, avoiding `read_dir` for
  unchanged directories while still detecting a deep file added when the root
  timestamp also changes. Writing `analysis.premiere.xml` can change that root
  timestamp, so this was a direct cause of potentially missing clips.
- Discovery cache schema 3 removes recursively duplicated aggregate path lists.
  Each directory stores only its direct matches and child directories, keeping
  cache size linear in the number of files and directories.

### Premiere XML and clip coloring

- XML is generated completely in memory, parsed back for well-formedness and
  expected clip/file counts, then committed using durable atomic replacement.
- Export refuses an empty or incomplete source set instead of silently omitting
  clips and overwriting a valid previous XML.
- Duplicate source entries are collapsed and the best segment is selected
  globally for each unique source path.
- The sequence rate is selected from the most common source rate rather than the
  first lexically sorted filename. Sane low-rate/non-slow-motion tie breakers
  remain for mixed footage.
- Every select receives an explicit Premiere label. Segment kind takes
  precedence over legacy movement metadata, preventing stale cached metadata
  from recoloring slow-motion or static-subject clips incorrectly.
- UNC paths now use a standard `file://server/share/...` URI.
- Source audio probing records stream index, sample rate, bit depth, and channel
  count. Audio export remains off by default, but the opt-in path writes a
  sequence audio track, source audio characteristics, reciprocal video/audio
  links, and one shared file record per source.
- XML validation now counts video and audio clipitems independently while still
  requiring exactly one source path record per selected file.
- Export summaries report timeline duration and the movement, subject,
  slow-motion, fallback, cache, failure, and linked-audio counts.

Color mapping:

| Segment | Label |
| --- | --- |
| Static | Cerulean |
| Static subject | Caribbean |
| Slow motion | Iris |
| Pan/tilt | Forest |
| Zoom | Mango |
| Roll | Lavender |
| Complex move | Rose |
| Preserved original (>90 seconds) | Yellow |

### Persistence and extraction safety

- Settings, discovery metadata, analysis sidecars, and Premiere XML now share a
  same-directory atomic writer. It uses unique temporary files, flushes file
  contents, replaces the destination in one operation, and cleans up after
  errors. Windows replacement uses `MoveFileExW` with replace and write-through
  flags; Unix also syncs the parent directory.
- Embedded assets use a full-content hash. The digest is cached in-process so
  the large FFmpeg archive is not hashed separately for `ffmpeg` and `ffprobe`.
- Extraction locks are RAII guards, so all error paths release them. Locks left
  by a terminated process are treated as stale after ten minutes.
- The latest successful export summary is persisted. Results survives restarts,
  can recover an older XML created before summary persistence, and provides
  Open XML, Show in folder, and Copy path actions.

## CPU portability decision

The project contains no host-native compiler flags, manual AVX dispatch, or
vendor-specific AMD/Intel code. The default ONNX execution-provider chain always
ends with CPU, and movement-only builds compile without ONNX using
`--no-default-features`. Release artifacts currently target x86-64 Windows and
Linux; 32-bit x86 and ARM are outside the current release matrix.

This means supported AMD and Intel x64 CPUs execute the same code. GPU providers
are optional acceleration features and are not part of correctness or cache
identity.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo clippy --no-default-features --all-targets -- -D warnings`
- Default suite: 80 passed, 2 opt-in tests
- No-default-features suite: 77 passed, 1 opt-in FFmpeg test
- Bundled YOLO model inference smoke test: passed on CPU
- FFmpeg synthetic end-to-end camera-motion test: passed
- Compiler target inspection: no `RUSTFLAGS` override or `target-cpu=native`

Real-folder verification on `D:\Media\take`:

- 20/20 source clips exported with no failures.
- The selects timeline fell from about 349 seconds to 145.16 seconds; no select
  exceeds the configured 8-second cap.
- A repeat run loaded all 20 sidecars and regenerated XML in a few seconds.
- Optional audio produced 20 reciprocal linked audio clipitems at 48 kHz stereo
  with only 20 source path records; the normal video-only export was then
  restored.

## Intentional constraints

- Variable-frame-rate media remains rejected for exact trims. Supporting it
  correctly requires retaining packet timestamps; pretending it is constant
  rate would produce inaccurate Premiere in/out points.
- Model quality cannot be compared responsibly without labeled real footage.
  The current model stays in place until a reproducible evaluation dataset and
  acceptance thresholds exist.
