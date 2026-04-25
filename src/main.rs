#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analyzer;
mod cache;
mod config;
mod engine;
mod error;
mod gui;
mod media;
mod settings;
mod timeline;
mod xml_exporter;

fn main() {
    engine::init_tracing(false);

    if let Err(err) = gui::run_gui() {
        eprintln!("{err}");
        std::process::exit(err.exit_code());
    }
}
