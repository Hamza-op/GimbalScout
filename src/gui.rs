use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use eframe::egui;
use rfd::FileDialog;

use crate::config;
use crate::engine::{self, AnalyzeArgs, ProgressMsg, RunSummary};
use crate::error::{AppError, AppResult};
use crate::media;
use crate::settings::{PersistedExportSummary, PersistedSettings};

// ──────────────────────────────────────────────
//  Color Palette – Motion Lab
// ──────────────────────────────────────────────
const BG_DEEP: egui::Color32 = egui::Color32::from_rgb(11, 13, 14);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(19, 22, 24);
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(21, 24, 25);
const BG_INPUT: egui::Color32 = egui::Color32::from_rgb(15, 17, 19);
const BG_SOFT: egui::Color32 = egui::Color32::from_rgb(28, 32, 33);
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
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([860.0, 620.0]),
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

    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(8.0);
    style.spacing.button_padding = egui::vec2(9.0, 4.0);
    style.spacing.text_edit_width = 240.0;
    style.spacing.interact_size.y = 24.0;

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
    style.visuals.window_stroke = egui::Stroke::NONE;

    style.visuals.widgets.noninteractive.bg_fill = BG_CARD;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, TEXT_SECONDARY);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;

    style.visuals.widgets.inactive.bg_fill = BG_INPUT;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0_f32, TEXT_PRIMARY);
    style.visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(44, 50, 51));

    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(43, 39, 34);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5_f32, ACCENT_ORANGE);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5_f32, ACCENT_ORANGE);

    style.visuals.widgets.active.bg_fill = ACCENT_ORANGE;
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, BORDER_GLOW);

    style.visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(242, 137, 68, 70);
    style.visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT_AMBER);
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
    page: UiPage,
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
    acceleration: config::AccelerationInfo,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiPage {
    Analyze,
    Results,
    Settings,
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

#[derive(Clone, Copy)]
enum SummaryAction {
    OpenXml,
    ShowInFolder,
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

        let last_summary = restore_last_summary(persisted.as_ref());
        Self {
            page: UiPage::Analyze,
            form,
            status: StatusState::Ready,
            running: false,
            result_receiver: None,
            progress_receiver: None,
            start_time: None,
            last_summary,
            progress: ProgressState::default(),
            cancel_flag: None,
            persisted_settings: persisted,
            setup_state: SetupState::Idle,
            setup_result_rx: None,
            setup_progress_rx: None,
            acceleration: config::acceleration_info(),
        }
    }
}

impl eframe::App for VideoToolApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker(ctx);
        self.poll_setup(ctx);
        self.paint_background(ctx);

        self.render_sidebar(ctx);
        self.render_main(ctx);
    }
}

impl VideoToolApp {
    fn paint_background(&self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let painter = ctx.layer_painter(egui::LayerId::background());
        painter.rect_filled(screen, 0.0, BG_DEEP);

        // Soft horizontal accent stripe at the top: orange in the centre,
        // fading to transparent at both edges. Approximated with a few
        // overlaid rectangles since egui's painter has no native gradient.
        let stripe_rect = egui::Rect::from_min_size(
            egui::pos2(screen.left(), screen.top()),
            egui::vec2(screen.width(), 2.0),
        );
        painter.rect_filled(stripe_rect, 0.0, ACCENT_ORANGE);

        // Subtle vignette glow under the stripe.
        let glow_rect = egui::Rect::from_min_size(
            egui::pos2(screen.left(), screen.top() + 2.0),
            egui::vec2(screen.width(), 28.0),
        );
        painter.rect_filled(
            glow_rect,
            0.0,
            egui::Color32::from_rgba_unmultiplied(242, 137, 68, 14),
        );
    }

    // ── Navigation rail ─────────────────────────
    fn render_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("navigation")
            .exact_width(184.0)
            .resizable(false)
            .frame(egui::Frame {
                fill: egui::Color32::from_rgb(15, 18, 20),
                inner_margin: egui::Margin::symmetric(14.0, 18.0),
                stroke: egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(38, 43, 45)),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    render_brand_mark(ui);
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("VIDEO TOOL")
                            .size(15.0)
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                });

                ui.add_space(34.0);
                if navigation_button(ui, "Analyze", self.page == UiPage::Analyze).clicked() {
                    self.page = UiPage::Analyze;
                }
                ui.add_space(6.0);
                if navigation_button(ui, "Results", self.page == UiPage::Results).clicked() {
                    self.page = UiPage::Results;
                }
                ui.add_space(6.0);
                if navigation_button(ui, "Settings", self.page == UiPage::Settings).clicked() {
                    self.page = UiPage::Settings;
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    render_sidebar_status(ui, &self.status, self.start_time);
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new("Premiere XML")
                            .size(10.5)
                            .color(TEXT_MUTED),
                    );
                });
            });
    }

    // ── Main scrollable content ─────────────────
    fn render_main(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: BG_DEEP,
                inner_margin: egui::Margin::symmetric(22.0, 18.0),
                ..Default::default()
            })
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        let content_width =
                            (ui.clip_rect().right() - ui.cursor().left()).max(400.0);
                        ui.set_width(content_width);
                        ui.set_max_width(content_width);
                        match self.page {
                            UiPage::Analyze => self.render_analyze_page(ui),
                            UiPage::Results => self.render_results_page(ui),
                            UiPage::Settings => self.render_settings_page(ui),
                        }
                    });
            });
    }

    fn render_analyze_page(&mut self, ui: &mut egui::Ui) {
        page_header(
            ui,
            "Analyze footage",
            "Find the strongest camera move in every clip and export clean Premiere selections.",
        );
        ui.add_space(18.0);

        ui.add_enabled_ui(!self.running, |ui| self.card_input(ui));
        ui.add_space(12.0);

        if ui.available_width() < 900.0 {
            ui.add_enabled_ui(!self.running, |ui| self.card_advanced(ui));
            ui.add_space(12.0);
            self.render_launch_column(ui);
        } else {
            let gap = 12.0;
            let available = (ui.max_rect().right() - ui.cursor().left()).max(0.0);
            let left_width = ((available - gap) * 0.60).max(420.0);
            let right_width = (available - gap - left_width).max(300.0);
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(left_width, 0.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_max_width(left_width);
                        ui.add_enabled_ui(!self.running, |ui| self.card_advanced(ui));
                    },
                );
                ui.add_space(gap);
                ui.allocate_ui_with_layout(
                    egui::vec2(right_width, 0.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_max_width(right_width);
                        self.render_launch_column(ui);
                    },
                );
            });
        }

        ui.add_space(12.0);
        self.render_recent_export(ui);
    }

    fn render_launch_column(&mut self, ui: &mut egui::Ui) {
        self.action_bar(ui);
        ui.add_space(10.0);

        if self.running {
            self.render_progress(ui);
        } else if let Some(summary) = self.last_summary.clone() {
            if let Some(action) = render_summary_card(ui, &summary) {
                self.handle_summary_action(action, &summary);
            }
        } else {
            render_card(ui, "Export summary", |ui| {
                ui.horizontal(|ui| {
                    render_document_icon(ui, ACCENT_TEAL);
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("One best selection per clip")
                                .size(13.0)
                                .color(TEXT_PRIMARY)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new(
                                "The highest-scoring move from every clip will be written to Premiere XML.",
                            )
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                        );
                    });
                });
            });
        }
    }

    fn render_results_page(&mut self, ui: &mut egui::Ui) {
        page_header(
            ui,
            "Results",
            "Review and reopen the latest Premiere XML export.",
        );
        ui.add_space(18.0);
        if let Some(summary) = self.last_summary.clone() {
            if let Some(action) = render_summary_card(ui, &summary) {
                self.handle_summary_action(action, &summary);
            }
        } else {
            render_empty_state(
                ui,
                "No export yet",
                "Run an analysis and the best selection will appear here.",
            );
        }
    }

    fn render_settings_page(&mut self, ui: &mut egui::Ui) {
        page_header(
            ui,
            "Analysis settings",
            "Tune detection and tool paths. Defaults are optimized for accuracy.",
        );
        ui.add_space(18.0);
        ui.add_enabled_ui(!self.running, |ui| self.card_advanced(ui));
    }

    fn render_recent_export(&self, ui: &mut egui::Ui) {
        render_card(ui, "Recent export", |ui| {
            if let Some(summary) = &self.last_summary {
                let filename = summary
                    .output_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
                    .unwrap_or("analysis.premiere.xml");
                ui.horizontal(|ui| {
                    render_document_icon(ui, SUCCESS);
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(filename)
                            .size(12.0)
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        render_signal_badge(ui, "Ready", SUCCESS);
                        render_badge(ui, "Best per clip");
                    });
                });
            } else {
                ui.horizontal(|ui| {
                    render_document_icon(ui, TEXT_MUTED);
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Your latest XML export will appear here.")
                            .size(11.5)
                            .color(TEXT_MUTED),
                    );
                });
            }
        });
    }

    // ── Card: Input folder + extensions ─────────
    fn card_input(&mut self, ui: &mut egui::Ui) {
        render_card(ui, "Source folder", |ui| {
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

    // ── Card: Editor mode presets ───────────────
    fn card_advanced(&mut self, ui: &mut egui::Ui) {
        render_card(ui, "Mode", |ui| {
            section_header(ui, "Pick one");
            for mode in EditorMode::ALL {
                mode_button(ui, mode, &mut self.form);
                ui.add_space(4.0);
            }

            ui.add_space(2.0);
            control_strip(ui, |ui| {
                compact_label(ui, "Current");
                render_signal_badge(ui, self.form.editor_mode_label(), ACCENT_ORANGE);
                render_signal_badge(ui, self.form.sampling_label().as_str(), ACCENT_TEAL);
                render_signal_badge(
                    ui,
                    if self.form.enable_yolo {
                        "Subject detection"
                    } else {
                        "Movement only"
                    },
                    if self.form.enable_yolo {
                        ACCENT_AMBER
                    } else {
                        TEXT_SECONDARY
                    },
                );
            });

            section_header(ui, "Export");
            control_strip(ui, |ui| {
                ui.checkbox(&mut self.form.include_audio, "Include source audio")
                    .on_hover_text("Adds linked production audio when the source clip contains it");
                ui.add_space(12.0);
                compact_label(ui, "Select length");
                ui.add_sized(
                    [72.0, 26.0],
                    egui::DragValue::new(&mut self.form.max_select_seconds)
                        .speed(0.5)
                        .range(2.0..=30.0)
                        .suffix(" s")
                        .max_decimals(1),
                )
                .on_hover_text(
                    "Maximum XML select duration; shorter detections receive edit handles",
                );
            });

            section_header(ui, "Performance");
            control_strip(ui, |ui| {
                compact_label(ui, "Acceleration");
                let badge_color = if self.acceleration.gpu_heavy {
                    SUCCESS
                } else {
                    ACCENT_TEAL
                };
                let label = self.acceleration.label();
                let detail = self.acceleration.detail();
                let resp = ui
                    .scope(|ui| {
                        render_signal_badge(ui, &label, badge_color);
                    })
                    .response;
                resp.on_hover_text(detail);
            });

            ui.add_space(4.0);
            egui::CollapsingHeader::new(
                egui::RichText::new("Advanced")
                    .size(11.5)
                    .color(ACCENT_AMBER)
                    .strong(),
            )
            .default_open(false)
            .show(ui, |ui| {
                section_header(ui, "Fine tune");
                control_strip(ui, |ui| {
                    compact_label(ui, "Motion");
                    let drag = egui::DragValue::new(&mut self.form.motion_threshold)
                        .speed(0.05)
                        .range(0.0..=16.0)
                        .max_decimals(2)
                        .custom_formatter(|n, _| {
                            if n == 0.0 {
                                "Auto".to_string()
                            } else {
                                format!("{n:.2}")
                            }
                        });
                    ui.add_sized([64.0, 26.0], drag);
                    ui.add_space(12.0);
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
                control_strip(ui, |ui| {
                    compact_label(ui, "Workers");
                    ui.add_sized(
                        [78.0, 26.0],
                        egui::TextEdit::singleline(&mut self.form.max_files)
                            .desired_width(78.0)
                            .hint_text("auto"),
                    )
                    .on_hover_text(format!(
                        "How many files to analyze in parallel. Leave blank for auto ({})",
                        default_worker_count()
                    ));
                    ui.add_space(12.0);
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
                ui.add_enabled_ui(self.form.enable_yolo && cfg!(feature = "yolo"), |ui| {
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
        });
    }

    // ── Action buttons bar ──────────────────────
    fn action_bar(&mut self, ui: &mut egui::Ui) {
        let title = if self.running {
            "Analysis in progress"
        } else {
            "Ready to analyze"
        };
        render_card(ui, title, |ui| {
            let has_input = !self.form.input.trim().is_empty();
            let btn_text = if self.running {
                "Stop analysis"
            } else {
                "Start analysis"
            };

            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("One best selection per clip")
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    render_signal_badge(ui, "Premiere XML", ACCENT_ORANGE);
                    render_signal_badge(ui, "Best per clip", ACCENT_TEAL);
                    if self.form.include_audio {
                        render_signal_badge(ui, "Linked audio", ACCENT_AMBER);
                    }
                });
            });
            ui.add_space(14.0);

            let btn_color = if !has_input && !self.running {
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
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(255, 148, 132))
            } else if !has_input {
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(84, 92, 94))
            } else {
                egui::Stroke::new(1.0_f32, BORDER_GLOW)
            };

            let btn = egui::Button::new(
                egui::RichText::new(btn_text)
                    .size(16.0)
                    .color(btn_color)
                    .strong(),
            )
            .fill(btn_fill)
            .rounding(egui::Rounding::same(9.0))
            .stroke(btn_stroke)
            .min_size(egui::vec2(ui.available_width(), 52.0));

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
        render_card(ui, "Live Telemetry", |ui| {
            ui.columns(3, |columns| {
                dashboard_stat(
                    &mut columns[0],
                    "Found",
                    &self.progress.total_files.to_string(),
                    ACCENT_AMBER,
                );
                dashboard_stat(
                    &mut columns[1],
                    "Done",
                    &self.progress.completed_files.to_string(),
                    ACCENT_TEAL,
                );
                dashboard_stat(
                    &mut columns[2],
                    "Active",
                    &self.progress.active_files.len().to_string(),
                    ACCENT_ORANGE,
                );
            });
            ui.add_space(10.0);

            egui::Frame::none()
                .fill(BG_SOFT)
                .rounding(egui::Rounding::same(8.0))
                .stroke(egui::Stroke::new(1.0_f32, BORDER_SUBTLE))
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
        settings.preferences.enable_yolo = self.form.enable_yolo && cfg!(feature = "yolo");
        settings.preferences.include_audio = self.form.include_audio;
        settings.preferences.max_select_seconds = self.form.max_select_seconds;
        settings.preferences.verbose = self.form.verbose;
        settings.preferences.ffmpeg_override = self.form.ffmpeg_bin.clone();
        settings.preferences.ffprobe_override = self.form.ffprobe_bin.clone();
        settings.preferences.yolo_override = self.form.yolo_model.clone();

        if let Err(e) = settings.save() {
            tracing::warn!("Failed to save settings: {e}");
        }
    }

    fn persist_export_summary(&mut self, summary: &RunSummary) {
        let Some(output_path) = summary.output_path.as_ref() else {
            return;
        };
        let settings = self
            .persisted_settings
            .get_or_insert_with(PersistedSettings::default);
        settings.last_export = Some(PersistedExportSummary {
            files_scanned: summary.files_scanned,
            files_analyzed: summary.files_analyzed,
            cached_files: summary.cached_files,
            exported_segments: summary.exported_segments,
            selected_duration_seconds: summary.selected_duration_seconds,
            movement_segments: summary.movement_segments,
            subject_segments: summary.subject_segments,
            slow_motion_segments: summary.slow_motion_segments,
            static_segments: summary.static_segments,
            audio_segments: summary.audio_segments,
            failed_paths: summary
                .failed_paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            output_path: output_path.to_string_lossy().into_owned(),
        });
        if let Err(error) = settings.save() {
            tracing::warn!("Failed to save latest export summary: {error}");
        }
    }

    fn handle_summary_action(&mut self, action: SummaryAction, summary: &RunSummary) {
        let Some(path) = summary.output_path.as_deref() else {
            return;
        };
        let result = match action {
            SummaryAction::OpenXml => open_output_path(path),
            SummaryAction::ShowInFolder => reveal_output_path(path),
        };
        if let Err(error) = result {
            self.status = StatusState::Error(format!("Could not open export: {error}"));
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
                            "Partial export in {elapsed}s — {} analyzed, {} failed, {} selections → {xml_name}",
                            summary.files_analyzed, summary.failed_files, summary.exported_segments,
                        ))
                    } else {
                        StatusState::Success(format!(
                            "Done in {elapsed}s — {} files, {} selections → {xml_name}",
                            summary.files_analyzed, summary.exported_segments,
                        ))
                    };
                    self.persist_export_summary(&summary);
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
        let enable_yolo = self.form.enable_yolo && cfg!(feature = "yolo");
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
                    let tools_dir = r
                        .ffmpeg
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "app data".to_string());
                    let probe_dir = r
                        .ffprobe
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| tools_dir.clone());
                    let mut summary = if probe_dir == tools_dir {
                        format!("FFmpeg + FFprobe cached in {tools_dir}")
                    } else {
                        format!("FFmpeg cached in {tools_dir}\nFFprobe cached in {probe_dir}")
                    };
                    if let Some(ref yolo) = r.yolo_model
                        && let Some(dir) = yolo.parent()
                    {
                        summary.push_str(&format!("\nYOLO cached in {}", dir.display()));
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
                    .fill(egui::Color32::from_rgb(16, 30, 24))
                    .rounding(egui::Rounding::same(6.0))
                    .inner_margin(egui::Margin::symmetric(10.0, 8.0))
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
                        for line in summary.lines().take(2) {
                            let response = ui.add(
                                egui::Label::new(
                                    egui::RichText::new(short_path_line(line))
                                        .size(10.0)
                                        .color(TEXT_MUTED)
                                        .monospace(),
                                )
                                .truncate(),
                            );
                            response.on_hover_text(line);
                        }
                    });
                ui.add_space(4.0);
            }
            SetupState::Failed(err) => {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(30, 18, 16))
                    .rounding(egui::Rounding::same(6.0))
                    .inner_margin(egui::Margin::symmetric(10.0, 8.0))
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
            egui::Color32::from_rgb(24, 42, 39)
        };
        let btn = egui::Button::new(
            egui::RichText::new(btn_label)
                .size(12.0)
                .color(if is_running { TEXT_MUTED } else { ACCENT_TEAL })
                .strong(),
        )
        .fill(btn_fill)
        .rounding(egui::Rounding::same(7.0))
        .stroke(egui::Stroke::NONE)
        .min_size(egui::vec2(ui.available_width(), 30.0));

        let response = ui
            .add_enabled(can_start, btn)
            .on_hover_text("Pre-extract and validate tools now so analysis starts instantly");
        if response.clicked() {
            self.start_setup();
        }
    }
}

fn short_path_line(line: &str) -> String {
    const MAX: usize = 76;
    if line.chars().count() <= MAX {
        return line.to_string();
    }
    let prefix: String = line.chars().take(24).collect();
    let suffix: String = line
        .chars()
        .rev()
        .take(42)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{prefix}…{suffix}")
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
    include_audio: bool,
    max_select_seconds: f32,
    max_files: String,
    extensions: String,
    verbose: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SamplingPreset {
    Low,
    Medium,
    High,
    Max,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EditorMode {
    Movement,
    SubjectSelects,
}

impl EditorMode {
    const ALL: [Self; 2] = [Self::Movement, Self::SubjectSelects];

    fn label(self) -> &'static str {
        match self {
            Self::Movement => "Movement",
            Self::SubjectSelects => "People + Motion",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Movement => "Camera movement only",
            Self::SubjectSelects => "Camera movement with subject detection",
        }
    }

    fn values(self) -> (SamplingPreset, f32, f32, bool, f32) {
        match self {
            Self::Movement => (SamplingPreset::High, 0.0, 1.0, false, 0.42),
            Self::SubjectSelects => (SamplingPreset::High, 0.0, 1.0, true, 0.42),
        }
    }

    fn apply(self, form: &mut AnalyzeForm) {
        let (sampling, motion_threshold, window_seconds, enable_yolo, person_confidence) =
            self.values();
        form.set_sampling_preset(sampling);
        form.motion_threshold = motion_threshold;
        form.window_seconds = window_seconds;
        form.enable_yolo = enable_yolo && cfg!(feature = "yolo");
        form.person_confidence = person_confidence;
        form.max_files.clear();
    }

    fn matches_form(self, form: &AnalyzeForm) -> bool {
        let (sampling, motion_threshold, window_seconds, enable_yolo, person_confidence) =
            self.values();
        form.sampling_preset() == Some(sampling)
            && (form.motion_threshold - motion_threshold).abs() < 0.05
            && (form.window_seconds - window_seconds).abs() < 0.05
            && form.enable_yolo == (enable_yolo && cfg!(feature = "yolo"))
            && (form.person_confidence - person_confidence).abs() < 0.05
    }
}

impl SamplingPreset {
    const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Max];

    fn values(self) -> (u32, f32) {
        match self {
            Self::Low => (360, 8.0),
            Self::Medium => (540, 12.0),
            Self::High => (720, 18.0),
            Self::Max => (720, 24.0),
        }
    }

    fn from_values(height: u32, fps: f32) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| {
            let (preset_height, preset_fps) = preset.values();
            height == preset_height && (fps - preset_fps).abs() < 0.05
        })
    }
}

impl Default for AnalyzeForm {
    fn default() -> Self {
        Self {
            input: String::new(),
            ffmpeg_bin: String::new(),
            ffprobe_bin: String::new(),
            yolo_model: String::new(),
            analysis_height: 720,
            analysis_fps: 18.0,
            window_seconds: 1.0,
            motion_threshold: 0.0,
            person_confidence: 0.42,
            enable_yolo: cfg!(feature = "yolo"),
            include_audio: false,
            max_select_seconds: 8.0,
            max_files: String::new(),
            extensions: media::DEFAULT_VIDEO_EXTENSIONS.to_string(),
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
            enable_yolo: p.enable_yolo && cfg!(feature = "yolo"),
            include_audio: p.include_audio,
            max_select_seconds: p.max_select_seconds,
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
            enable_yolo: self.enable_yolo && cfg!(feature = "yolo"),
            include_audio: self.include_audio,
            max_select_seconds: self.max_select_seconds,
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

    fn sampling_preset(&self) -> Option<SamplingPreset> {
        SamplingPreset::from_values(self.analysis_height, self.analysis_fps)
    }

    fn set_sampling_preset(&mut self, preset: SamplingPreset) {
        let (height, fps) = preset.values();
        self.analysis_height = height;
        self.analysis_fps = fps;
    }

    fn sampling_label(&self) -> String {
        let effective_height = if self.enable_yolo {
            self.analysis_height
        } else {
            self.analysis_height.clamp(2, 144)
        };
        let purpose = if self.enable_yolo { "detect" } else { "motion" };
        match self.sampling_preset() {
            Some(preset) => {
                let (_, fps) = preset.values();
                format!("{effective_height} px {purpose} / {fps:.0} fps")
            }
            None => format!(
                "Custom: {effective_height} px {purpose} / {:.1} fps",
                self.analysis_fps
            ),
        }
    }

    fn editor_mode_label(&self) -> &'static str {
        EditorMode::ALL
            .into_iter()
            .find(|mode| mode.matches_form(self))
            .map(EditorMode::label)
            .unwrap_or("Custom")
    }
}

fn restore_last_summary(settings: Option<&PersistedSettings>) -> Option<RunSummary> {
    let settings = settings?;
    if let Some(summary) = &settings.last_export {
        return Some(RunSummary {
            files_scanned: summary.files_scanned,
            files_analyzed: summary.files_analyzed,
            cached_files: summary.cached_files,
            exported_segments: summary.exported_segments,
            selected_duration_seconds: summary.selected_duration_seconds,
            movement_segments: summary.movement_segments,
            subject_segments: summary.subject_segments,
            slow_motion_segments: summary.slow_motion_segments,
            static_segments: summary.static_segments,
            audio_segments: summary.audio_segments,
            failed_files: summary.failed_paths.len(),
            failed_paths: summary.failed_paths.iter().map(PathBuf::from).collect(),
            output_path: Some(PathBuf::from(&summary.output_path)),
        });
    }

    // One-time recovery for exports created before summaries were persisted.
    let legacy_path =
        PathBuf::from(settings.preferences.last_input.trim()).join("analysis.premiere.xml");
    legacy_path.is_file().then_some(RunSummary {
        output_path: Some(legacy_path),
        ..RunSummary::default()
    })
}

fn open_output_path(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        Command::new("rundll32.exe")
            .args(["url.dll,FileProtocolHandler"])
            .arg(path)
            .spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(path).spawn()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(path).spawn()?;
    }
    Ok(())
}

fn reveal_output_path(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer.exe")
            .arg(format!("/select,{}", path.display()))
            .spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg("-R").arg(path).spawn()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(path.parent().unwrap_or(path))
            .spawn()?;
    }
    Ok(())
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

fn page_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.label(
        egui::RichText::new(title)
            .size(27.0)
            .color(TEXT_PRIMARY)
            .strong(),
    );
    ui.add_space(3.0);
    ui.label(
        egui::RichText::new(subtitle)
            .size(12.5)
            .color(TEXT_SECONDARY),
    );
}

fn navigation_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let width = ui.available_width();
    let fill = if selected {
        egui::Color32::from_rgb(34, 35, 34)
    } else {
        egui::Color32::TRANSPARENT
    };

    let response = egui::Frame::none()
        .fill(fill)
        .rounding(egui::Rounding::same(8.0))
        .stroke(if selected {
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(48, 49, 47))
        } else {
            egui::Stroke::NONE
        })
        .inner_margin(egui::Margin::symmetric(12.0, 11.0))
        .show(ui, |ui| {
            ui.set_min_width((width - 24.0).max(100.0));
            ui.horizontal(|ui| {
                let color = if selected { ACCENT_ORANGE } else { TEXT_MUTED };
                let (icon_rect, _) =
                    ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                ui.painter().circle_stroke(
                    icon_rect.center(),
                    6.0,
                    egui::Stroke::new(1.5_f32, color),
                );
                ui.painter().hline(
                    icon_rect.x_range(),
                    icon_rect.center().y,
                    egui::Stroke::new(1.0_f32, color),
                );
                ui.add_space(7.0);
                ui.label(
                    egui::RichText::new(label)
                        .size(13.0)
                        .color(if selected {
                            ACCENT_ORANGE
                        } else {
                            TEXT_SECONDARY
                        })
                        .strong(),
                );
            });
        })
        .response
        .interact(egui::Sense::click());

    if selected {
        let marker = egui::Rect::from_min_size(
            egui::pos2(response.rect.left(), response.rect.top() + 8.0),
            egui::vec2(2.0, response.rect.height() - 16.0),
        );
        ui.painter()
            .rect_filled(marker, egui::Rounding::same(1.0), ACCENT_ORANGE);
    }

    response
}

fn render_sidebar_status(ui: &mut egui::Ui, status: &StatusState, start_time: Option<Instant>) {
    let (label, color, detail) = match status {
        StatusState::Ready => ("Ready".to_string(), ACCENT_TEAL, "Ready".to_string()),
        StatusState::Running(message) => {
            let elapsed = start_time
                .map(|start| start.elapsed().as_secs())
                .unwrap_or(0);
            (
                format!("Analyzing · {elapsed}s"),
                ACCENT_AMBER,
                message.clone(),
            )
        }
        StatusState::Success(message) => ("Export ready".to_string(), SUCCESS, message.clone()),
        StatusState::Error(message) => ("Needs attention".to_string(), DANGER, message.clone()),
    };

    let width = ui.available_width();
    let response = egui::Frame::none()
        .fill(egui::Color32::from_rgb(21, 25, 27))
        .rounding(egui::Rounding::same(7.0))
        .stroke(egui::Stroke::new(1.0_f32, BORDER_SUBTLE))
        .inner_margin(egui::Margin::symmetric(10.0, 9.0))
        .show(ui, |ui| {
            ui.set_min_width((width - 20.0).max(100.0));
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("●").size(9.0).color(color));
                ui.label(egui::RichText::new(label).size(11.5).color(color));
            });
        })
        .response;
    response.on_hover_text(detail);
}

fn render_document_icon(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(28.0, 32.0), egui::Sense::hover());
    let body = rect.shrink2(egui::vec2(4.0, 3.0));
    ui.painter().rect_stroke(
        body,
        egui::Rounding::same(3.0),
        egui::Stroke::new(1.4_f32, color),
    );
    for offset in [12.0, 17.0, 22.0] {
        ui.painter().hline(
            (body.left() + 5.0)..=(body.right() - 5.0),
            body.top() + offset,
            egui::Stroke::new(1.0_f32, color),
        );
    }
}

fn render_empty_state(ui: &mut egui::Ui, title: &str, body: &str) {
    egui::Frame::none()
        .fill(BG_PANEL)
        .rounding(egui::Rounding::same(10.0))
        .stroke(egui::Stroke::new(1.0_f32, BORDER_SUBTLE))
        .inner_margin(egui::Margin::same(28.0))
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                render_document_icon(ui, TEXT_MUTED);
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(title)
                        .size(16.0)
                        .color(TEXT_PRIMARY)
                        .strong(),
                );
                ui.label(egui::RichText::new(body).size(11.5).color(TEXT_MUTED));
            });
        });
}

fn render_card(ui: &mut egui::Ui, title: &str, content: impl FnOnce(&mut egui::Ui)) {
    let viewport_width = (ui.max_rect().right() - ui.cursor().left()).max(200.0);
    let outer_width = ui.available_width().min(viewport_width);

    egui::Frame::none()
        .fill(BG_PANEL)
        .rounding(egui::Rounding::same(10.0))
        .stroke(egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgb(44, 49, 51),
        ))
        .inner_margin(egui::Margin::symmetric(16.0, 13.0))
        .show(ui, |ui| {
            let content_width = (outer_width - 32.0).max(200.0);
            ui.set_width(content_width);
            ui.set_max_width(content_width);
            ui.label(
                egui::RichText::new(title)
                    .size(13.5)
                    .color(TEXT_PRIMARY)
                    .strong(),
            );
            ui.add_space(10.0);
            content(ui);
        });
}

fn render_badge(ui: &mut egui::Ui, text: &str) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(36, 40, 42))
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgb(74, 82, 84),
        ))
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(10.0).color(TEXT_PRIMARY));
        });
}

fn render_summary_card(ui: &mut egui::Ui, summary: &RunSummary) -> Option<SummaryAction> {
    let mut action = None;
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(20, 30, 24))
        .rounding(egui::Rounding::same(8.0))
        .stroke(egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgb(43, 78, 54),
        ))
        .inner_margin(egui::Margin::same(16.0))
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

            let has_run_stats = summary.files_scanned > 0
                || summary.files_analyzed > 0
                || summary.exported_segments > 0;
            if has_run_stats {
                ui.horizontal_wrapped(|ui| {
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
                        "Selections",
                        &summary.exported_segments.to_string(),
                        ACCENT_AMBER,
                    );
                    if summary.selected_duration_seconds > 0.0 {
                        ui.add_space(8.0);
                        stat_pill(
                            ui,
                            "Timeline",
                            &format_duration(summary.selected_duration_seconds),
                            ACCENT_TEAL,
                        );
                    }
                    if summary.cached_files > 0 {
                        ui.add_space(8.0);
                        stat_pill(ui, "Cached", &summary.cached_files.to_string(), ACCENT_TEAL);
                    }
                    if summary.failed_files > 0 {
                        ui.add_space(8.0);
                        stat_pill(ui, "Failed", &summary.failed_files.to_string(), DANGER);
                    }
                });
            } else if summary.output_path.is_some() {
                ui.label(
                    egui::RichText::new(
                        "Existing XML recovered. Run analysis once to populate detailed statistics.",
                    )
                    .size(11.5)
                    .color(TEXT_SECONDARY),
                );
            }

            if summary.exported_segments > 0 {
                ui.add_space(10.0);
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(24, 48, 35))
                    .rounding(egui::Rounding::same(6.0))
                    .stroke(egui::Stroke::new(
                        1.0_f32,
                        egui::Color32::from_rgb(54, 122, 77),
                    ))
                    .inner_margin(egui::Margin::symmetric(10.0, 7.0))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} best selection{} exported",
                                    summary.exported_segments,
                                    if summary.exported_segments == 1 {
                                        ""
                                    } else {
                                        "s"
                                    },
                                ))
                                .size(11.5)
                                .color(TEXT_PRIMARY)
                                .strong(),
                            );
                            ui.label(
                                egui::RichText::new("— AI-assisted; review before the final cut")
                                    .size(11.0)
                                    .color(TEXT_SECONDARY),
                            );
                        });
                    });
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if summary.movement_segments > 0 {
                        render_badge(ui, &format!("{} movement", summary.movement_segments));
                    }
                    if summary.subject_segments > 0 {
                        render_badge(ui, &format!("{} subject", summary.subject_segments));
                    }
                    if summary.slow_motion_segments > 0 {
                        render_badge(ui, &format!("{} slow motion", summary.slow_motion_segments));
                    }
                    if summary.static_segments > 0 {
                        render_badge(ui, &format!("{} fallback", summary.static_segments));
                    }
                    render_signal_badge(
                        ui,
                        if summary.audio_segments > 0 {
                            "Source audio linked"
                        } else {
                            "Video only"
                        },
                        if summary.audio_segments > 0 {
                            ACCENT_AMBER
                        } else {
                            TEXT_MUTED
                        },
                    );
                });
            }

            if !summary.failed_paths.is_empty() {
                ui.add_space(10.0);
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(38, 23, 20))
                    .rounding(egui::Rounding::same(6.0))
                    .stroke(egui::Stroke::new(1.0_f32, DANGER))
                    .inner_margin(egui::Margin::symmetric(10.0, 7.0))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Could not analyze after retry:")
                                .size(11.0)
                                .color(DANGER)
                                .strong(),
                        );
                        for path in &summary.failed_paths {
                            let name = path
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or("unknown video");
                            ui.label(
                                egui::RichText::new(name)
                                    .size(10.5)
                                    .color(TEXT_SECONDARY)
                                    .monospace(),
                            )
                            .on_hover_text(path.display().to_string());
                        }
                    });
            }

            if let Some(path) = &summary.output_path {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    render_signal_badge(ui, "XML", TEXT_MUTED);
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(path.display().to_string())
                            .size(11.5)
                            .color(TEXT_SECONDARY)
                            .monospace(),
                    );
                });
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Open XML").clicked() {
                        action = Some(SummaryAction::OpenXml);
                    }
                    if ui.button("Show in folder").clicked() {
                        action = Some(SummaryAction::ShowInFolder);
                    }
                    if ui.button("Copy path").clicked() {
                        ui.ctx().copy_text(path.display().to_string());
                    }
                });
            }
        });
    action
}

fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds.max(0.0).round() as u64;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    if minutes > 0 {
        format!("{minutes}:{seconds:02}")
    } else {
        format!("{seconds}s")
    }
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
        .inner_margin(egui::Margin::symmetric(10.0, 7.0))
        .show(ui, |ui| {
            ui.set_min_width((ui.available_width() - 2.0).max(72.0));
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(label).size(10.0).color(TEXT_MUTED));
                ui.label(egui::RichText::new(value).size(14.0).color(color).strong());
            });
        });
}

/// Compact brand mark: two stacked accent bars suggesting timeline tracks.
fn render_brand_mark(ui: &mut egui::Ui) {
    let size = egui::vec2(20.0, 20.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter();
    let inset = 2.0;
    let bar_h = 4.0;
    let gap = 2.0;
    let top = egui::Rect::from_min_size(
        egui::pos2(rect.left() + inset, rect.center().y - bar_h - gap / 2.0),
        egui::vec2(rect.width() - inset * 2.0, bar_h),
    );
    let bot = egui::Rect::from_min_size(
        egui::pos2(rect.left() + inset, rect.center().y + gap / 2.0),
        egui::vec2((rect.width() - inset * 2.0) * 0.55, bar_h),
    );
    painter.rect_filled(top, egui::Rounding::same(1.5), ACCENT_ORANGE);
    painter.rect_filled(bot, egui::Rounding::same(1.5), ACCENT_TEAL);
    // Decorative dot at the right of the bottom bar.
    painter.circle_filled(
        egui::pos2(
            rect.right() - inset - 2.0,
            rect.center().y + gap / 2.0 + bar_h / 2.0,
        ),
        2.0,
        ACCENT_AMBER,
    );
}

fn render_signal_badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    // `from_rgba_unmultiplied` gives a true translucent tint; the previous
    // `from_rgba_premultiplied(r, g, b, 70)` was invalid because premultiplied
    // alpha requires r, g, b ≤ a — bright accents like ACCENT_AMBER overflowed
    // to a solid block which made the text inside disappear.
    let fill = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 48);
    let text_color = readable_on(color);
    egui::Frame::none()
        .fill(fill)
        .rounding(egui::Rounding::same(5.0))
        .stroke(egui::Stroke::new(1.0_f32, color))
        .inner_margin(egui::Margin::symmetric(8.0, 3.0))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(10.0)
                    .color(text_color)
                    .strong(),
            );
        });
}

/// Pick a foreground colour that stays legible on top of `bg`.
/// Uses perceived luminance so light accents (amber, teal, success) get
/// a near-black label while dark accents keep the cream `TEXT_PRIMARY`.
fn readable_on(bg: egui::Color32) -> egui::Color32 {
    let r = bg.r() as f32;
    let g = bg.g() as f32;
    let b = bg.b() as f32;
    let lum = 0.299 * r + 0.587 * g + 0.114 * b;
    if lum > 165.0 {
        egui::Color32::from_rgb(18, 14, 8)
    } else {
        TEXT_PRIMARY
    }
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
        let w = 78.0;
        let h = ui.spacing().interact_size.y;
        ui.allocate_ui_with_layout(
            egui::vec2(w, h),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(label).size(12.0).color(label_color));
            },
        );

        let text_width = (ui.available_width() - 78.0).max(120.0);
        let text_edit = egui::TextEdit::singleline(value).hint_text(if required {
            "Required"
        } else {
            "Auto-detect"
        });
        ui.add_sized([text_width, h], text_edit);

        let browse_btn =
            egui::Button::new(egui::RichText::new("Browse").size(11.5).color(TEXT_PRIMARY))
                .fill(egui::Color32::from_rgb(42, 37, 32))
                .rounding(egui::Rounding::same(6.0))
                .stroke(egui::Stroke::new(1.0_f32, ACCENT_ORANGE))
                .min_size(egui::vec2(64.0, h));

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
    let w = 78.0;
    let h = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(
        egui::vec2(w, h),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(label).size(12.0).color(TEXT_SECONDARY));
        },
    );
}

fn compact_label(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).size(11.0).color(TEXT_SECONDARY));
}

fn control_strip(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) {
    let width = ui.available_width();
    egui::Frame::none()
        .fill(egui::Color32::from_rgba_premultiplied(
            BG_INPUT.r(),
            BG_INPUT.g(),
            BG_INPUT.b(),
            160,
        ))
        .rounding(egui::Rounding::same(6.0))
        .inner_margin(egui::Margin::symmetric(8.0, 5.0))
        .show(ui, |ui| {
            ui.set_min_width((width - 16.0).max(200.0));
            ui.horizontal_wrapped(content);
        });
    ui.add_space(4.0);
}

fn mode_button(ui: &mut egui::Ui, mode: EditorMode, form: &mut AnalyzeForm) -> egui::Response {
    let selected = mode.matches_form(form);
    let fill = if selected {
        egui::Color32::from_rgb(52, 35, 25)
    } else {
        egui::Color32::from_rgb(25, 29, 31)
    };
    let stroke = if selected {
        egui::Stroke::new(1.25_f32, BORDER_GLOW)
    } else {
        egui::Stroke::new(1.0_f32, BORDER_SUBTLE)
    };
    let response = egui::Frame::none()
        .fill(fill)
        .rounding(egui::Rounding::same(9.0))
        .stroke(stroke)
        .inner_margin(egui::Margin::symmetric(14.0, 12.0))
        .show(ui, |ui| {
            ui.set_min_width((ui.available_width() - 28.0).max(200.0));
            ui.set_min_height(52.0);
            ui.horizontal(|ui| {
                render_mode_icon(ui, mode, if selected { ACCENT_ORANGE } else { ACCENT_TEAL });
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.add_space(3.0);
                    ui.label(
                        egui::RichText::new(mode.label())
                            .size(14.0)
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(mode.description())
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    render_selection_radio(ui, selected);
                    if selected {
                        ui.add_space(8.0);
                        render_signal_badge(ui, "ACTIVE", ACCENT_AMBER);
                    }
                });
            });
        })
        .response
        .interact(egui::Sense::click());

    if response.clicked() {
        mode.apply(form);
    }
    response
}

fn render_mode_icon(ui: &mut egui::Ui, mode: EditorMode, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(44.0, 44.0), egui::Sense::hover());
    ui.painter().rect_filled(
        rect,
        egui::Rounding::same(7.0),
        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 22),
    );
    ui.painter().rect_stroke(
        rect,
        egui::Rounding::same(7.0),
        egui::Stroke::new(1.0_f32, color),
    );
    match mode {
        EditorMode::Movement => {
            ui.painter()
                .circle_stroke(rect.center(), 7.0, egui::Stroke::new(1.5_f32, color));
            ui.painter().line_segment(
                [
                    rect.center() + egui::vec2(5.0, -5.0),
                    rect.center() + egui::vec2(11.0, -11.0),
                ],
                egui::Stroke::new(1.5_f32, color),
            );
        }
        EditorMode::SubjectSelects => {
            ui.painter().circle_stroke(
                rect.center() - egui::vec2(0.0, 5.0),
                5.0,
                egui::Stroke::new(1.5_f32, color),
            );
            ui.painter().line_segment(
                [
                    rect.center() + egui::vec2(-9.0, 11.0),
                    rect.center() + egui::vec2(-6.0, 5.0),
                ],
                egui::Stroke::new(1.5_f32, color),
            );
            ui.painter().line_segment(
                [
                    rect.center() + egui::vec2(-6.0, 5.0),
                    rect.center() + egui::vec2(6.0, 5.0),
                ],
                egui::Stroke::new(1.5_f32, color),
            );
            ui.painter().line_segment(
                [
                    rect.center() + egui::vec2(6.0, 5.0),
                    rect.center() + egui::vec2(9.0, 11.0),
                ],
                egui::Stroke::new(1.5_f32, color),
            );
        }
    }
}

fn render_selection_radio(ui: &mut egui::Ui, selected: bool) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
    let color = if selected { ACCENT_ORANGE } else { TEXT_MUTED };
    ui.painter()
        .circle_stroke(rect.center(), 8.0, egui::Stroke::new(1.5_f32, color));
    if selected {
        ui.painter().circle_filled(rect.center(), 4.0, color);
    }
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.add_space(5.0);
    ui.label(
        egui::RichText::new(label)
            .size(11.0)
            .color(ACCENT_AMBER)
            .strong(),
    );
    ui.add_space(2.0);
}
