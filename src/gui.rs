use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use eframe::egui;
use rfd::FileDialog;

use crate::config;
use crate::engine::{self, AnalyzeArgs, ProgressMsg, RunSummary};
use crate::error::{AppError, AppResult};
use crate::settings::PersistedSettings;

// ──────────────────────────────────────────────
//  Color Palette – Motion Lab
// ──────────────────────────────────────────────
const BG_DEEP: egui::Color32 = egui::Color32::from_rgb(11, 13, 14);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(19, 22, 24);
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(25, 29, 31);
const BG_INPUT: egui::Color32 = egui::Color32::from_rgb(15, 17, 19);
const BG_SOFT: egui::Color32 = egui::Color32::from_rgb(31, 35, 36);
const BORDER_SUBTLE: egui::Color32 = egui::Color32::from_rgb(54, 61, 62);
const BORDER_GLOW: egui::Color32 = egui::Color32::from_rgb(242, 137, 68);

const ACCENT_TEAL: egui::Color32 = egui::Color32::from_rgb(80, 205, 185);
const ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(242, 137, 68);
const ACCENT_AMBER: egui::Color32 = egui::Color32::from_rgb(249, 208, 120);

const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(243, 236, 223);
const TEXT_SECONDARY: egui::Color32 = egui::Color32::from_rgb(204, 194, 178);
const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(156, 147, 133);

const SUCCESS: egui::Color32 = egui::Color32::from_rgb(110, 212, 132);
const DANGER: egui::Color32 = egui::Color32::from_rgb(255, 102, 76);

pub fn run_gui() -> AppResult<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1080.0, 820.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Video Tool",
        options,
        Box::new(|cc| {
            apply_theme(&cc.egui_ctx);
            Ok(Box::new(VideoToolApp::new()))
        }),
    )
    .map_err(|e| AppError::Message(format!("failed to start GUI: {e}")))
}

fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(10.0);
    style.spacing.button_padding = egui::vec2(12.0, 6.0);
    style.spacing.text_edit_width = 260.0;

    let rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.noninteractive.rounding = rounding;
    style.visuals.widgets.inactive.rounding = rounding;
    style.visuals.widgets.hovered.rounding = rounding;
    style.visuals.widgets.active.rounding = rounding;
    style.visuals.widgets.open.rounding = rounding;
    style.visuals.window_rounding = egui::Rounding::same(6.0);
    style.visuals.menu_rounding = egui::Rounding::same(6.0);

    style.visuals.dark_mode = true;
    style.visuals.panel_fill = BG_PANEL;
    style.visuals.window_fill = BG_PANEL;
    style.visuals.extreme_bg_color = BG_INPUT;
    style.visuals.faint_bg_color = BG_SOFT;
    style.visuals.override_text_color = Some(TEXT_PRIMARY);
    style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);

    style.visuals.widgets.noninteractive.bg_fill = BG_CARD;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_SECONDARY);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);

    style.visuals.widgets.inactive.bg_fill = BG_INPUT;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);

    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(43, 39, 34);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, ACCENT_ORANGE);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5, ACCENT_ORANGE);

    style.visuals.widgets.active.bg_fill = ACCENT_ORANGE;
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, BORDER_GLOW);

    style.visuals.selection.bg_fill = egui::Color32::from_rgba_premultiplied(242, 137, 68, 70);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT_AMBER);
    style.visuals.hyperlink_color = ACCENT_AMBER;

    ctx.set_style(style);
}

// ──────────────────────────────────────────────
//  Progress state tracked in the GUI
// ──────────────────────────────────────────────

#[derive(Clone, Default)]
struct ProgressState {
    total_files: usize,
    completed_files: usize,
    discovery_complete: bool,
    /// Full paths currently being processed by workers.
    active_files: Vec<PathBuf>,
    /// Human-readable label for the current setup phase (before files start).
    preparing_phase: Option<String>,
}

impl ProgressState {
    fn fraction(&self) -> f32 {
        if self.total_files == 0 {
            0.0
        } else if self.discovery_complete {
            self.completed_files as f32 / self.total_files as f32
        } else {
            let in_flight = self.completed_files + self.active_files.len();
            in_flight.min(self.total_files) as f32 / self.total_files as f32
        }
    }

    fn label(&self) -> String {
        if self.total_files == 0 {
            "Discovering files…".to_string()
        } else if !self.discovery_complete {
            format!(
                "Scanning and processing… {} found, {} done",
                self.total_files, self.completed_files
            )
        } else {
            format!(
                "Processing file {} of {}",
                (self.completed_files + 1).min(self.total_files),
                self.total_files
            )
        }
    }
}

struct VideoToolApp {
    form: AnalyzeForm,
    status: StatusState,
    running: bool,
    result_receiver: Option<Receiver<Result<RunSummary, String>>>,
    progress_receiver: Option<Receiver<ProgressMsg>>,
    start_time: Option<Instant>,
    last_summary: Option<RunSummary>,
    progress: ProgressState,
    cancel_flag: Option<Arc<AtomicBool>>,
    /// Persisted settings loaded on startup; passed to the engine and saved
    /// after every successful run.
    persisted_settings: Option<PersistedSettings>,
    /// Setup tools background state.
    setup_state: SetupState,
    setup_result_rx: Option<Receiver<Result<String, String>>>,
    setup_progress_rx: Option<Receiver<String>>,
}

#[derive(Clone, Default)]
enum SetupState {
    #[default]
    Idle,
    Running(String),
    Done(String),
    Failed(String),
}

#[derive(Clone)]
enum StatusState {
    Ready,
    Running(String),
    Success(String),
    Error(String),
}

impl VideoToolApp {
    fn new() -> Self {
        // Load persisted settings; fall back to defaults on error.
        let persisted = match PersistedSettings::load() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to load settings: {e}");
                None
            }
        };

        let form = if let Some(ref s) = persisted {
            AnalyzeForm::from_settings(s)
        } else {
            AnalyzeForm::default()
        };

        Self {
            form,
            status: StatusState::Ready,
            running: false,
            result_receiver: None,
            progress_receiver: None,
            start_time: None,
            last_summary: None,
            progress: ProgressState::default(),
            cancel_flag: None,
            persisted_settings: persisted,
            setup_state: SetupState::Idle,
            setup_result_rx: None,
            setup_progress_rx: None,
        }
    }
}

impl eframe::App for VideoToolApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker(ctx);
        self.poll_setup(ctx);
        self.paint_background(ctx);

        self.render_header(ctx);
        self.render_status_bar(ctx);
        self.render_main(ctx);
    }
}

impl VideoToolApp {
    fn paint_background(&self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let painter = ctx.layer_painter(egui::LayerId::background());
        painter.rect_filled(screen, 0.0, BG_DEEP);

        let grid = egui::Color32::from_rgba_premultiplied(255, 255, 255, 5);
        let mut x = screen.left();
        while x < screen.right() {
            painter.line_segment(
                [egui::pos2(x, screen.top()), egui::pos2(x, screen.bottom())],
                egui::Stroke::new(1.0, grid),
            );
            x += 56.0;
        }
        let mut y = screen.top();
        while y < screen.bottom() {
            painter.line_segment(
                [egui::pos2(screen.left(), y), egui::pos2(screen.right(), y)],
                egui::Stroke::new(1.0, grid),
            );
            y += 56.0;
        }
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(screen.left(), screen.top()),
                egui::vec2(screen.width(), 3.0),
            ),
            0.0,
            ACCENT_ORANGE,
        );
    }

    // ── Top header bar ──────────────────────────
    fn render_header(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header_panel")
            .frame(egui::Frame {
                fill: egui::Color32::from_rgba_premultiplied(19, 22, 24, 245),
                inner_margin: egui::Margin::symmetric(18.0, 10.0),
                stroke: egui::Stroke::new(1.0, BORDER_SUBTLE),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("VIDEO TOOL")
                            .size(18.0)
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                    ui.add_space(10.0);
                    render_signal_badge(ui, "Premiere XML", ACCENT_ORANGE);
                    render_signal_badge(ui, self.form.sensitivity_label(), ACCENT_TEAL);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        render_badge(
                            ui,
                            if self.form.enable_yolo {
                                "YOLO On"
                            } else {
                                "YOLO Off"
                            },
                        );
                        ui.add_space(4.0);
                        render_badge(ui, "Motion Fit");
                    });
                });
            });
    }

    // ── Bottom status bar ───────────────────────
    fn render_status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .frame(egui::Frame {
                fill: BG_PANEL,
                inner_margin: egui::Margin::symmetric(14.0, 6.0),
                stroke: egui::Stroke::new(1.0, BORDER_SUBTLE),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| match &self.status {
                    StatusState::Ready => {
                        ui.label(
                            egui::RichText::new("●")
                                .size(10.0)
                                .color(egui::Color32::from_rgb(196, 184, 164)),
                        );
                        ui.label(
                            egui::RichText::new("Ready")
                                .size(12.0)
                                .color(TEXT_SECONDARY),
                        );
                    }
                    StatusState::Running(msg) => {
                        ui.spinner();
                        ui.label(egui::RichText::new(msg).size(12.0).color(ACCENT_AMBER));
                        if let Some(start) = self.start_time {
                            let elapsed = start.elapsed().as_secs();
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new(format!("⏱ {elapsed}s"))
                                            .size(12.0)
                                            .color(TEXT_SECONDARY),
                                    );
                                },
                            );
                        }
                    }
                    StatusState::Success(msg) => {
                        ui.label(egui::RichText::new("✓").size(13.0).color(SUCCESS));
                        ui.label(egui::RichText::new(msg).size(12.0).color(SUCCESS));
                    }
                    StatusState::Error(msg) => {
                        ui.label(egui::RichText::new("✗").size(13.0).color(DANGER));
                        ui.label(egui::RichText::new(msg).size(12.0).color(DANGER));
                    }
                });
            });
    }

    // ── Main scrollable content ─────────────────
    fn render_main(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: BG_DEEP,
                inner_margin: egui::Margin::symmetric(14.0, 12.0),
                ..Default::default()
            })
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        ui.columns(2, |columns| {
                            columns[0].add_enabled_ui(!self.running, |ui| {
                                self.card_input(ui);
                                ui.add_space(10.0);
                                self.action_bar(ui);
                            });
                            columns[0].add_space(10.0);
                            self.render_telemetry_column(&mut columns[0]);

                            columns[1].add_enabled_ui(!self.running, |ui| {
                                self.card_advanced(ui);
                            });
                        });
                    });
            });
    }

    fn render_telemetry_column(&self, ui: &mut egui::Ui) {
        if self.running {
            self.render_progress(ui);
        } else if let Some(summary) = &self.last_summary {
            render_summary_card(ui, summary);
        } else {
            render_card(ui, "OUT", "Output", |ui| {
                ui.columns(3, |columns| {
                    dashboard_stat(&mut columns[0], "Status", "Ready", TEXT_SECONDARY);
                    dashboard_stat(&mut columns[1], "Segments", "0", ACCENT_AMBER);
                    dashboard_stat(&mut columns[2], "Failed", "0", SUCCESS);
                });
            });
        }
    }

    // ── Card: Input folder + extensions ─────────
    fn card_input(&mut self, ui: &mut egui::Ui) {
        render_card(ui, "SRC", "Source", |ui| {
            path_row(ui, "Folder", &mut self.form.input, BrowseKind::Folder, true);

            param_row(ui, "Extensions", |ui| {
                let re = ui.add(
                    egui::TextEdit::singleline(&mut self.form.extensions)
                        .desired_width(220.0)
                        .hint_text("mov,mp4,mxf"),
                );
                re.on_hover_text("Comma-separated file extensions to scan");
            });
        });
    }

    // ── Card: Advanced settings (collapsed) ─────
    fn card_advanced(&mut self, ui: &mut egui::Ui) {
        render_card(ui, "CTL", "Deep Controls", |ui| {
            section_header(ui, "Detection");
            control_strip(ui, |ui| {
                compact_label(ui, "Detector");
                toggle_chip(ui, "YOLO", &mut self.form.enable_yolo);
                ui.add_space(14.0);
                ui.add_enabled_ui(self.form.enable_yolo, |ui| {
                    compact_label(ui, "Person");
                    ui.add_sized(
                        [64.0, 26.0],
                        egui::DragValue::new(&mut self.form.person_confidence)
                            .speed(0.01)
                            .range(0.0..=1.0)
                            .max_decimals(2),
                    );
                });
            });

            section_header(ui, "Motion");
            control_strip(ui, |ui| {
                compact_label(ui, "Sensitivity");
                sensitivity_button(ui, "Subtle", &mut self.form.motion_threshold, 1.4);
                sensitivity_button(ui, "Balanced", &mut self.form.motion_threshold, 1.8);
                sensitivity_button(ui, "Strict", &mut self.form.motion_threshold, 3.2);
            });
            control_strip(ui, |ui| {
                compact_label(ui, "Threshold");
                ui.add_sized(
                    [64.0, 26.0],
                    egui::DragValue::new(&mut self.form.motion_threshold)
                        .speed(0.05)
                        .range(0.2..=16.0)
                        .max_decimals(2),
                );
                ui.add_space(18.0);
                compact_label(ui, "Window");
                ui.add_sized(
                    [70.0, 26.0],
                    egui::DragValue::new(&mut self.form.window_seconds)
                        .speed(0.1)
                        .range(0.25..=30.0)
                        .suffix(" s")
                        .max_decimals(2),
                );
            });

            section_header(ui, "Sampling");
            control_strip(ui, |ui| {
                compact_label(ui, "Height");
                ui.add_sized(
                    [76.0, 26.0],
                    egui::DragValue::new(&mut self.form.analysis_height)
                        .speed(1.0)
                        .range(2..=2160)
                        .suffix(" px"),
                );
                ui.add_space(18.0);
                compact_label(ui, "FPS");
                ui.add_sized(
                    [64.0, 26.0],
                    egui::DragValue::new(&mut self.form.analysis_fps)
                        .speed(0.25)
                        .range(0.25..=60.0)
                        .max_decimals(2),
                );
            });
            control_strip(ui, |ui| {
                compact_label(ui, "Workers");
                let worker_input = ui.add_sized(
                    [78.0, 26.0],
                    egui::TextEdit::singleline(&mut self.form.max_files)
                        .desired_width(78.0)
                        .hint_text("auto"),
                );
                worker_input.on_hover_text(format!(
                    "How many files to analyze in parallel. Leave blank for auto ({})",
                    default_worker_count()
                ));
                ui.add_space(18.0);
                ui.checkbox(&mut self.form.verbose, "Verbose");
            });

            section_header(ui, "Tools");
            path_row(
                ui,
                "FFmpeg",
                &mut self.form.ffmpeg_bin,
                BrowseKind::File,
                false,
            );
            path_row(
                ui,
                "FFprobe",
                &mut self.form.ffprobe_bin,
                BrowseKind::File,
                false,
            );
            ui.add_enabled_ui(self.form.enable_yolo, |ui| {
                path_row(
                    ui,
                    "YOLO",
                    &mut self.form.yolo_model,
                    BrowseKind::File,
                    false,
                );
            });

            ui.add_space(6.0);
            self.render_setup_button(ui);
        });
    }

    // ── Action buttons bar ──────────────────────
    fn action_bar(&mut self, ui: &mut egui::Ui) {
        render_card(ui, "RUN", "Launch", |ui| {
            let has_input = !self.form.input.trim().is_empty();
            let btn_text = if self.running {
                "STOP ANALYSIS"
            } else {
                "START ANALYSIS"
            };

            let btn_color = if self.running {
                egui::Color32::WHITE
            } else if !has_input {
                TEXT_SECONDARY
            } else {
                egui::Color32::WHITE
            };
            let btn_fill = if self.running {
                DANGER
            } else if !has_input {
                BG_SOFT
            } else {
                ACCENT_ORANGE
            };
            let btn_stroke = if self.running {
                egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 148, 132))
            } else if !has_input {
                egui::Stroke::new(1.0, egui::Color32::from_rgb(84, 92, 94))
            } else {
                egui::Stroke::new(1.0, BORDER_GLOW)
            };

            let btn = egui::Button::new(egui::RichText::new(btn_text).size(15.0).color(btn_color))
                .fill(btn_fill)
                .rounding(egui::Rounding::same(8.0))
                .stroke(btn_stroke)
                .min_size(egui::vec2(ui.available_width(), 38.0));

            let enabled = self.running || has_input;
            let response = ui.add_enabled(enabled, btn);
            if response.clicked() {
                if self.running {
                    self.stop_job();
                } else {
                    self.start_job();
                }
            }
            if !has_input && !self.running {
                response.on_hover_text("Select an input folder first");
            }

            if self.running {
                ui.add_space(8.0);
                let fraction = self.progress.fraction();
                let pct = (fraction * 100.0) as u32;
                let bar = egui::ProgressBar::new(fraction)
                    .text(
                        egui::RichText::new(format!("{} · {}%", self.progress.label(), pct))
                            .size(12.5)
                            .color(TEXT_PRIMARY),
                    )
                    .fill(ACCENT_ORANGE)
                    .rounding(egui::Rounding::same(8.0));

                ui.add_sized([ui.available_width(), 28.0], bar);
            }
        });
    }

    // ── Live progress panel ─────────────────────
    fn render_progress(&self, ui: &mut egui::Ui) {
        render_card(ui, "03", "Live Telemetry", |ui| {
            ui.horizontal(|ui| {
                dashboard_stat(
                    ui,
                    "Found",
                    &self.progress.total_files.to_string(),
                    ACCENT_AMBER,
                );
                dashboard_stat(
                    ui,
                    "Done",
                    &self.progress.completed_files.to_string(),
                    ACCENT_TEAL,
                );
                dashboard_stat(
                    ui,
                    "Active",
                    &self.progress.active_files.len().to_string(),
                    ACCENT_ORANGE,
                );
            });
            ui.add_space(10.0);

            egui::Frame::none()
                .fill(BG_SOFT)
                .rounding(egui::Rounding::same(8.0))
                .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
                .inner_margin(egui::Margin::same(14.0))
                .show(ui, |ui| {
                    if self.progress.active_files.is_empty() {
                        let phase_label = self
                            .progress
                            .preparing_phase
                            .as_deref()
                            .unwrap_or("Discovering and preparing files…");
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                egui::RichText::new(phase_label)
                                    .size(12.0)
                                    .color(ACCENT_AMBER),
                            );
                        });
                    } else {
                        ui.label(
                            egui::RichText::new("Currently processing:")
                                .size(11.0)
                                .color(TEXT_MUTED),
                        );
                        ui.add_space(4.0);

                        for path in &self.progress.active_files {
                            let name = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown");
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("  ⟳").size(12.0).color(ACCENT_ORANGE),
                                );
                                ui.label(
                                    egui::RichText::new(name)
                                        .size(12.0)
                                        .color(TEXT_SECONDARY)
                                        .monospace(),
                                );
                            })
                            .response
                            .on_hover_text(path.display().to_string());
                        }
                    }
                });
        });
    }

    // ── Job management ──────────────────────────
    fn start_job(&mut self) {
        // Snapshot current form state into persisted preferences before
        // kicking off the analysis so the settings file stays current.
        self.save_preferences();

        match self.form.to_args(self.persisted_settings.clone()) {
            Ok(mut args) => {
                let cancel_flag = Arc::new(AtomicBool::new(false));
                args.cancel_flag = Arc::clone(&cancel_flag);
                let (result_tx, result_rx) = mpsc::channel();
                let (progress_tx, progress_rx) = mpsc::channel();
                self.result_receiver = Some(result_rx);
                self.progress_receiver = Some(progress_rx);
                self.running = true;
                self.cancel_flag = Some(cancel_flag);
                self.start_time = Some(Instant::now());
                self.last_summary = None;
                self.progress = ProgressState::default();
                self.status = StatusState::Running("Analyzing files…".to_string());

                std::thread::spawn(move || {
                    let result =
                        engine::run_analyze(args, Some(progress_tx)).map_err(|e| e.to_string());
                    let _ = result_tx.send(result);
                });
            }
            Err(err) => {
                self.status = StatusState::Error(err.to_string());
            }
        }
    }

    fn stop_job(&mut self) {
        if let Some(cancel_flag) = &self.cancel_flag {
            cancel_flag.store(true, Ordering::Relaxed);
            self.status = StatusState::Running("Stopping analysis…".to_string());
        }
    }

    /// Write current form values back into PersistedSettings and flush to disk.
    fn save_preferences(&mut self) {
        let settings = self
            .persisted_settings
            .get_or_insert_with(PersistedSettings::default);

        settings.preferences.last_input = self.form.input.clone();
        settings.preferences.extensions = self.form.extensions.clone();
        settings.preferences.analysis_height = self.form.analysis_height;
        settings.preferences.analysis_fps = self.form.analysis_fps;
        settings.preferences.window_seconds = self.form.window_seconds;
        settings.preferences.motion_threshold = self.form.motion_threshold;
        settings.preferences.person_confidence = self.form.person_confidence;
        settings.preferences.enable_yolo = self.form.enable_yolo;
        settings.preferences.verbose = self.form.verbose;
        settings.preferences.ffmpeg_override = self.form.ffmpeg_bin.clone();
        settings.preferences.ffprobe_override = self.form.ffprobe_bin.clone();
        settings.preferences.yolo_override = self.form.yolo_model.clone();

        if let Err(e) = settings.save() {
            tracing::warn!("Failed to save settings: {e}");
        }
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        if !self.running {
            return;
        }

        ctx.request_repaint_after(Duration::from_millis(100));

        // Drain all available progress messages without blocking.
        if let Some(prx) = &self.progress_receiver {
            loop {
                match prx.try_recv() {
                    Ok(msg) => match msg {
                        ProgressMsg::Preparing { phase } => {
                            self.progress.preparing_phase = Some(phase.clone());
                            self.status = StatusState::Running(phase);
                        }
                        ProgressMsg::Discovered { total } => {
                            self.progress.total_files = total;
                            if self.progress.active_files.is_empty() {
                                self.status = StatusState::Running(format!(
                                    "Scanning input folder… ({total} files found so far)"
                                ));
                            }
                        }
                        ProgressMsg::DiscoveryFinished { total } => {
                            self.progress.total_files = total;
                            self.progress.discovery_complete = true;
                            self.progress.preparing_phase = None;
                            self.status = StatusState::Running(if total == 0 {
                                "No matching files found.".to_string()
                            } else {
                                format!("Found {total} files — finishing analysis…")
                            });
                        }
                        ProgressMsg::FileStarted { path, .. } => {
                            self.progress.active_files.push(path);
                            self.status = StatusState::Running(self.progress.label());
                        }
                        ProgressMsg::FileFinished { path, .. } => {
                            if let Some(pos) =
                                self.progress.active_files.iter().position(|p| p == &path)
                            {
                                self.progress.active_files.swap_remove(pos);
                            }
                            self.progress.completed_files += 1;
                            self.status = StatusState::Running(self.progress.label());
                        }
                    },
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        }

        // Check for the final result.
        if let Some(receiver) = &self.result_receiver {
            match receiver.try_recv() {
                Ok(Ok(summary)) => {
                    let elapsed = self.start_time.map(|s| s.elapsed().as_secs()).unwrap_or(0);
                    let xml_name = summary
                        .output_path
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("analysis.premiere.xml")
                        .to_string();
                    self.running = false;
                    self.cancel_flag = None;
                    self.result_receiver = None;
                    self.progress_receiver = None;
                    self.status = if summary.failed_files > 0 {
                        StatusState::Success(format!(
                            "Partial export in {elapsed}s — {} analyzed, {} failed, {} segments → {xml_name}",
                            summary.files_analyzed, summary.failed_files, summary.exported_segments,
                        ))
                    } else {
                        StatusState::Success(format!(
                            "Done in {elapsed}s — {} files, {} segments → {xml_name}",
                            summary.files_analyzed, summary.exported_segments,
                        ))
                    };
                    self.last_summary = Some(summary);
                }
                Ok(Err(err)) => {
                    self.running = false;
                    self.cancel_flag = None;
                    self.result_receiver = None;
                    self.progress_receiver = None;
                    self.status = if err.contains("analysis cancelled") {
                        StatusState::Error("Analysis cancelled.".to_string())
                    } else {
                        StatusState::Error(format!("Analysis failed: {err}"))
                    };
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.running = false;
                    self.cancel_flag = None;
                    self.result_receiver = None;
                    self.progress_receiver = None;
                    self.status =
                        StatusState::Error("Worker disconnected unexpectedly.".to_string());
                }
            }
        }
    }

    // ── Setup tools ─────────────────────────────
    fn start_setup(&mut self) {
        let ffmpeg_override = optional_path(&self.form.ffmpeg_bin);
        let ffprobe_override = optional_path(&self.form.ffprobe_bin);
        let yolo_override = optional_path(&self.form.yolo_model);
        let enable_yolo = self.form.enable_yolo;
        let mut settings = self.persisted_settings.clone().unwrap_or_default();

        let (result_tx, result_rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        self.setup_result_rx = Some(result_rx);
        self.setup_progress_rx = Some(progress_rx);
        self.setup_state = SetupState::Running("Starting…".to_string());

        std::thread::spawn(move || {
            let ptx = progress_tx;
            let result = config::setup_tools(
                ffmpeg_override,
                ffprobe_override,
                yolo_override,
                enable_yolo,
                &mut settings,
                |msg| {
                    let _ = ptx.send(msg.to_string());
                },
            );
            match result {
                Ok(r) => {
                    let mut summary = format!(
                        "ffmpeg: {}\nffprobe: {}",
                        r.ffmpeg.display(),
                        r.ffprobe.display()
                    );
                    if let Some(ref yolo) = r.yolo_model {
                        summary.push_str(&format!("\nYOLO: {}", yolo.display()));
                    }
                    let _ = result_tx.send(Ok(summary));
                }
                Err(e) => {
                    let _ = result_tx.send(Err(e.to_string()));
                }
            }
        });
    }

    fn poll_setup(&mut self, ctx: &egui::Context) {
        if self.setup_result_rx.is_none() {
            return;
        }

        ctx.request_repaint_after(Duration::from_millis(80));

        // Drain progress messages.
        if let Some(prx) = &self.setup_progress_rx {
            loop {
                match prx.try_recv() {
                    Ok(msg) => {
                        self.setup_state = SetupState::Running(msg);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        }

        // Check for completion.
        if let Some(rx) = &self.setup_result_rx {
            match rx.try_recv() {
                Ok(Ok(summary)) => {
                    // Reload persisted settings so analysis picks up the cached paths.
                    if let Ok(Some(s)) = PersistedSettings::load() {
                        self.persisted_settings = Some(s);
                    }
                    self.setup_state = SetupState::Done(summary);
                    self.setup_result_rx = None;
                    self.setup_progress_rx = None;
                }
                Ok(Err(err)) => {
                    self.setup_state = SetupState::Failed(err);
                    self.setup_result_rx = None;
                    self.setup_progress_rx = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.setup_state = SetupState::Failed("Setup worker disconnected.".to_string());
                    self.setup_result_rx = None;
                    self.setup_progress_rx = None;
                }
            }
        }
    }

    fn render_setup_button(&mut self, ui: &mut egui::Ui) {
        let is_running = matches!(self.setup_state, SetupState::Running(_));
        let can_start = !is_running && !self.running;

        match &self.setup_state {
            SetupState::Idle => {}
            SetupState::Running(phase) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new(phase).size(11.0).color(ACCENT_AMBER));
                });
                ui.add_space(4.0);
            }
            SetupState::Done(summary) => {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(18, 28, 22))
                    .rounding(egui::Rounding::same(6.0))
                    .stroke(egui::Stroke::new(1.0, SUCCESS))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("✓").size(12.0).color(SUCCESS));
                            ui.label(
                                egui::RichText::new("Tools ready")
                                    .size(11.0)
                                    .color(SUCCESS)
                                    .strong(),
                            );
                        });
                        for line in summary.lines() {
                            ui.label(
                                egui::RichText::new(line)
                                    .size(10.0)
                                    .color(TEXT_MUTED)
                                    .monospace(),
                            );
                        }
                    });
                ui.add_space(4.0);
            }
            SetupState::Failed(err) => {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(30, 18, 16))
                    .rounding(egui::Rounding::same(6.0))
                    .stroke(egui::Stroke::new(1.0, DANGER))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(egui::RichText::new("✗").size(12.0).color(DANGER));
                            ui.label(egui::RichText::new(err).size(11.0).color(DANGER));
                        });
                    });
                ui.add_space(4.0);
            }
        }

        let btn_label = if is_running {
            "SETTING UP…"
        } else {
            "SETUP TOOLS"
        };
        let btn_fill = if is_running {
            BG_SOFT
        } else {
            egui::Color32::from_rgb(30, 38, 36)
        };
        let btn_stroke = if is_running {
            egui::Stroke::new(1.0, BORDER_SUBTLE)
        } else {
            egui::Stroke::new(1.0, ACCENT_TEAL)
        };
        let btn = egui::Button::new(
            egui::RichText::new(btn_label)
                .size(12.0)
                .color(if is_running { TEXT_MUTED } else { ACCENT_TEAL })
                .strong(),
        )
        .fill(btn_fill)
        .rounding(egui::Rounding::same(7.0))
        .stroke(btn_stroke)
        .min_size(egui::vec2(ui.available_width(), 30.0));

        let response = ui
            .add_enabled(can_start, btn)
            .on_hover_text("Pre-extract and validate tools now so analysis starts instantly");
        if response.clicked() {
            self.start_setup();
        }
    }
}

// ──────────────────────────────────────────────
//  AnalyzeForm
//  The output path is not exposed in the GUI: the generated XML and caches are
//  written next to the selected input folder.
// ──────────────────────────────────────────────

#[derive(Clone)]
struct AnalyzeForm {
    input: String,
    ffmpeg_bin: String,
    ffprobe_bin: String,
    yolo_model: String,
    analysis_height: u32,
    analysis_fps: f32,
    window_seconds: f32,
    motion_threshold: f32,
    person_confidence: f32,
    enable_yolo: bool,
    max_files: String,
    extensions: String,
    verbose: bool,
}

impl Default for AnalyzeForm {
    fn default() -> Self {
        Self {
            input: String::new(),
            ffmpeg_bin: String::new(),
            ffprobe_bin: String::new(),
            yolo_model: String::new(),
            analysis_height: 360,
            analysis_fps: 12.0,
            window_seconds: 1.0,
            motion_threshold: 1.8,
            person_confidence: 0.42,
            enable_yolo: true,
            max_files: String::new(),
            extensions: "mov,mp4,mxf".to_string(),
            verbose: false,
        }
    }
}

impl AnalyzeForm {
    /// Construct form fields from previously-persisted settings.
    fn from_settings(s: &PersistedSettings) -> Self {
        let p = &s.preferences;
        Self {
            input: p.last_input.clone(),
            ffmpeg_bin: p.ffmpeg_override.clone(),
            ffprobe_bin: p.ffprobe_override.clone(),
            yolo_model: p.yolo_override.clone(),
            analysis_height: p.analysis_height,
            analysis_fps: p.analysis_fps,
            window_seconds: p.window_seconds,
            motion_threshold: p.motion_threshold,
            person_confidence: p.person_confidence,
            enable_yolo: p.enable_yolo,
            max_files: String::new(),
            extensions: p.extensions.clone(),
            verbose: p.verbose,
        }
    }
    fn to_args(&self, persisted: Option<PersistedSettings>) -> AppResult<AnalyzeArgs> {
        let input = self.input.trim();
        if input.is_empty() {
            return Err(AppError::Unsupported(
                "Input folder is required.".to_string(),
            ));
        }

        let max_files = if self.max_files.trim().is_empty() {
            None
        } else {
            let parsed = self.max_files.trim().parse::<usize>().map_err(|_| {
                AppError::Unsupported(
                    "Workers must be a positive integer, or left blank for auto.".to_string(),
                )
            })?;
            if parsed == 0 {
                return Err(AppError::Unsupported(
                    "Workers must be greater than 0, or left blank for auto.".to_string(),
                ));
            }
            Some(parsed)
        };

        let input_path = PathBuf::from(input);
        Ok(AnalyzeArgs {
            input: input_path.clone(),
            output: input_path,
            yolo_model: optional_path(&self.yolo_model),
            enable_yolo: self.enable_yolo,
            ffmpeg_bin: optional_path(&self.ffmpeg_bin),
            ffprobe_bin: optional_path(&self.ffprobe_bin),
            analysis_height: self.analysis_height,
            analysis_fps: self.analysis_fps,
            window_seconds: self.window_seconds,
            motion_threshold: self.motion_threshold,
            person_confidence: self.person_confidence,
            max_files,
            extensions: self.extensions.trim().to_string(),
            verbose: self.verbose,
            yolo_intra_threads: None,
            ffmpeg_threads: None,
            buf_frames: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            persisted_settings: persisted,
        })
    }

    fn sensitivity_label(&self) -> &'static str {
        if (self.motion_threshold - 1.4).abs() < 0.05 {
            "Subtle"
        } else if (self.motion_threshold - 1.8).abs() < 0.05 {
            "Balanced"
        } else if (self.motion_threshold - 3.2).abs() < 0.05 {
            "Strict"
        } else {
            "Custom"
        }
    }
}

fn optional_path(value: &str) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().div_ceil(2).clamp(1, 8))
        .unwrap_or(4)
}

// ──────────────────────────────────────────────
//  Reusable UI Components
// ──────────────────────────────────────────────

fn render_card(ui: &mut egui::Ui, icon: &str, title: &str, content: impl FnOnce(&mut egui::Ui)) {
    let outer_width = ui.available_width();

    egui::Frame::none()
        .fill(BG_PANEL)
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(egui::Margin::same(12.0))
        .show(ui, |ui| {
            ui.set_min_width((outer_width - 24.0).max(220.0));

            ui.horizontal(|ui| {
                render_signal_badge(ui, icon, ACCENT_ORANGE);
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(title)
                        .size(14.0)
                        .color(TEXT_PRIMARY)
                        .strong(),
                );
            });

            let rect = ui.available_rect_before_wrap();
            let line_y = ui.cursor().min.y + 2.0;
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), line_y),
                    egui::pos2(rect.left() + 96.0, line_y),
                ],
                egui::Stroke::new(1.0, ACCENT_ORANGE),
            );
            ui.add_space(8.0);

            content(ui);
        });
}

fn render_badge(ui: &mut egui::Ui, text: &str) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(36, 40, 42))
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(74, 82, 84)))
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(10.0).color(TEXT_PRIMARY));
        });
}

fn render_summary_card(ui: &mut egui::Ui, summary: &RunSummary) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(20, 30, 24))
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(1.0, SUCCESS))
        .inner_margin(egui::Margin::same(20.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                render_signal_badge(ui, "OK", SUCCESS);
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Results")
                        .size(16.0)
                        .color(SUCCESS)
                        .strong(),
                );
            });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                stat_pill(
                    ui,
                    "Scanned",
                    &summary.files_scanned.to_string(),
                    TEXT_SECONDARY,
                );
                ui.add_space(8.0);
                stat_pill(
                    ui,
                    "Analyzed",
                    &summary.files_analyzed.to_string(),
                    ACCENT_ORANGE,
                );
                ui.add_space(8.0);
                stat_pill(
                    ui,
                    "Segments",
                    &summary.exported_segments.to_string(),
                    ACCENT_AMBER,
                );
                if summary.cached_files > 0 {
                    ui.add_space(8.0);
                    stat_pill(ui, "Cached", &summary.cached_files.to_string(), ACCENT_TEAL);
                }
                if summary.failed_files > 0 {
                    ui.add_space(8.0);
                    stat_pill(ui, "Failed", &summary.failed_files.to_string(), DANGER);
                }
            });

            if let Some(path) = &summary.output_path {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("📄").size(13.0).color(TEXT_MUTED));
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(path.display().to_string())
                            .size(11.5)
                            .color(TEXT_SECONDARY)
                            .monospace(),
                    );
                });
            }
        });
}

fn stat_pill(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    egui::Frame::none()
        .fill(BG_CARD)
        .rounding(egui::Rounding::same(8.0))
        .inner_margin(egui::Margin::symmetric(10.0, 7.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(label).size(10.0).color(TEXT_MUTED));
                ui.label(egui::RichText::new(value).size(16.0).color(color).strong());
            });
        });
}

fn dashboard_stat(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    egui::Frame::none()
        .fill(BG_CARD)
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(egui::Margin::symmetric(10.0, 7.0))
        .show(ui, |ui| {
            ui.set_min_width((ui.available_width() - 2.0).max(72.0));
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(label).size(10.0).color(TEXT_MUTED));
                ui.label(egui::RichText::new(value).size(14.0).color(color).strong());
            });
        });
}

fn render_signal_badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgba_premultiplied(
            color.r(),
            color.g(),
            color.b(),
            70,
        ))
        .rounding(egui::Rounding::same(6.0))
        .stroke(egui::Stroke::new(1.0, color))
        .inner_margin(egui::Margin::symmetric(8.0, 3.0))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(10.0)
                    .color(TEXT_PRIMARY)
                    .strong(),
            );
        });
}

#[derive(Clone, Copy)]
enum BrowseKind {
    Folder,
    File,
}

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    browse_kind: BrowseKind,
    required: bool,
) {
    ui.horizontal(|ui| {
        let label_color = if required {
            TEXT_PRIMARY
        } else {
            TEXT_SECONDARY
        };
        let w = 92.0;
        ui.allocate_ui(egui::vec2(w, 18.0), |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(label).size(12.0).color(label_color));
            });
        });

        let text_edit = egui::TextEdit::singleline(value)
            .desired_width((ui.available_width() - 84.0).max(140.0))
            .hint_text(if required { "Required" } else { "Auto-detect" });
        ui.add(text_edit);

        let browse_btn =
            egui::Button::new(egui::RichText::new("Browse").size(11.5).color(TEXT_PRIMARY))
                .fill(egui::Color32::from_rgb(42, 37, 32))
                .rounding(egui::Rounding::same(7.0))
                .stroke(egui::Stroke::new(1.0, ACCENT_ORANGE))
                .min_size(egui::vec2(68.0, 25.0));

        let response = ui.add(browse_btn);
        let response = match browse_kind {
            BrowseKind::Folder => response.on_hover_text("Choose source folder"),
            BrowseKind::File => response.on_hover_text("Choose tool executable or model file"),
        };

        if response.clicked() {
            let dialog = FileDialog::new();
            match browse_kind {
                BrowseKind::Folder => {
                    if let Some(path) = dialog.pick_folder() {
                        *value = path.display().to_string();
                    }
                }
                BrowseKind::File => {
                    if let Some(path) = dialog.pick_file() {
                        *value = path.display().to_string();
                    }
                }
            }
        }
    });
}

fn param_row(ui: &mut egui::Ui, label: &str, widget: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        param_label(ui, label);
        widget(ui);
    });
}

fn param_label(ui: &mut egui::Ui, label: &str) {
    let w = 92.0;
    ui.allocate_ui(egui::vec2(w, 18.0), |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(label).size(12.0).color(TEXT_SECONDARY));
        });
    });
}

fn compact_label(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).size(11.0).color(TEXT_SECONDARY));
}

fn control_strip(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) {
    let width = ui.available_width();
    egui::Frame::none()
        .fill(BG_INPUT)
        .rounding(egui::Rounding::same(7.0))
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(egui::Margin::symmetric(10.0, 7.0))
        .show(ui, |ui| {
            ui.set_min_width((width - 20.0).max(220.0));
            ui.horizontal_wrapped(content);
        });
    ui.add_space(6.0);
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(11.0)
                .color(ACCENT_AMBER)
                .strong(),
        );
        let rect = ui.available_rect_before_wrap();
        let y = ui.cursor().min.y + 8.0;
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, BORDER_SUBTLE),
        );
    });
    ui.add_space(2.0);
}

fn toggle_chip(ui: &mut egui::Ui, label: &str, value: &mut bool) -> egui::Response {
    let text = if *value {
        format!("{label} ON")
    } else {
        format!("{label} OFF")
    };
    let fill = if *value {
        ACCENT_TEAL
    } else {
        egui::Color32::from_rgb(35, 34, 33)
    };
    let color = if *value { BG_DEEP } else { ACCENT_AMBER };
    let stroke = if *value {
        egui::Stroke::new(1.0, ACCENT_TEAL)
    } else {
        egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 76, 57))
    };
    let response = ui.add(
        egui::Button::new(egui::RichText::new(text).size(11.5).color(color).strong())
            .fill(fill)
            .rounding(egui::Rounding::same(6.0))
            .stroke(stroke)
            .min_size(egui::vec2(92.0, 27.0)),
    );
    if response.clicked() {
        *value = !*value;
    }
    response
}

fn sensitivity_button(
    ui: &mut egui::Ui,
    label: &str,
    target: &mut f32,
    value: f32,
) -> egui::Response {
    let selected = (*target - value).abs() < 0.05;
    let fill = if selected {
        ACCENT_ORANGE
    } else {
        egui::Color32::from_rgb(32, 36, 36)
    };
    let text = if selected {
        egui::Color32::WHITE
    } else {
        TEXT_SECONDARY
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, BORDER_GLOW)
    } else {
        egui::Stroke::new(1.0, BORDER_SUBTLE)
    };
    let response = ui.add(
        egui::Button::new(egui::RichText::new(label).size(12.0).color(text))
            .fill(fill)
            .rounding(egui::Rounding::same(8.0))
            .stroke(stroke)
            .min_size(egui::vec2(76.0, 26.0)),
    );
    if response.clicked() {
        *target = value;
    }
    response
}
