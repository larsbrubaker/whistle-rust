//! # whistle-demo-native
//!
//! The whistle-rust demo on the desktop: the same whistle-demo UI as the browser's, in a native
//! window (agg-gui-shell). The model runs on an engine thread (`engine.rs`), so the window never
//! blocks; the library splits each pass across a few worker threads (`whistle::set_threads`,
//! `args::default_threads`). The model is a path from the command line, the checkout's
//! `models/whistle.cact`, the cache, or a checked download (`model.rs`). Files come from the OS
//! dialog (rfd), a drop on the window, the OS's open request or the command line.
//!
//! Live audio enters through the capture seam (`capture.rs`). No microphone source is wired in
//! yet, so Record shows as unavailable with a note; `--mic-from-wav` plays a file through the
//! same live path. `bridge.rs` connects the UI, the engine and the capture; `shell_host.rs` is
//! the shell's side.
//!
//! Run from `demo/`: `cargo run -p whistle-demo-native --release` (see `--help`).

mod args;
mod bridge;
mod capture;
mod engine;
mod model;
mod shell_host;
#[cfg(test)]
mod test_support;

use std::process::ExitCode;
use std::sync::Arc;

use agg_gui_shell::ShellConfig;
use whistle_demo::{build_app_on, install_fonts, DemoState, Platform, DESKTOP_TITLE};

use crate::args::{default_threads, Args, USAGE};
use crate::bridge::{Bridge, Startup, Wake};
use crate::capture::{Capture, WavCapture};
use crate::model::ModelSource;
use crate::shell_host::DemoShellHost;

/// Why Record is unavailable in this build.
const MIC_NOTE: &str = "live microphone input is not in the desktop build yet";

/// Painted frames before `--screenshot` captures, once the transcript is shown: enough for the
/// layout the new text asks for and a settled paint.
const SCREENSHOT_SETTLE_FRAMES: u32 = 8;

fn main() -> ExitCode {
    let argv: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let args = match Args::parse(argv.iter().cloned()) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("whistle-demo-native: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.help {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    whistle::set_threads(args.threads.unwrap_or_else(|| default_threads(cores)));

    let capture: Option<Box<dyn Capture>> = match &args.mic_from_wav {
        Some(path) => match std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| WavCapture::from_wav(&bytes, true))
        {
            Ok(capture) => Some(Box::new(capture)),
            Err(e) => {
                eprintln!("whistle-demo-native: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let mut config = ShellConfig::new(DESKTOP_TITLE)
        .with_logical_size(960.0, 720.0)
        .with_min_logical_size(560.0, 420.0)
        .with_device_label("whistle-demo-native");
    if let Some(path) = &args.screenshot {
        config = config.with_screenshot(path, SCREENSHOT_SETTLE_FRAMES);
    }

    let mut capture = Some(capture);
    let result = agg_gui_shell::run(config, |init| {
        let font = install_fonts(init.device_scale());
        let state = DemoState::new();
        // The engine and capture threads wake this (the UI) thread's loop through its queue.
        let queue = agg_gui::ui_thread::current_queue();
        let wake: Wake = Arc::new(move || queue.signal_async_state_change());
        let bridge = Bridge::start(
            state.clone(),
            ModelSource::new(args.model.clone()),
            capture.take().flatten(),
            MIC_NOTE,
            wake,
        );
        if let Some(path) = &args.wav {
            bridge.at_startup(Startup::Open(path.clone()));
        }
        if args.record {
            bridge.at_startup(Startup::Record);
        }
        let app = build_app_on(state, bridge.clone(), font, &Platform::desktop());
        let host = DemoShellHost::new(bridge, args.screenshot.is_some(), argv.clone());
        Ok((app, host))
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("whistle-demo-native: {e}");
            ExitCode::FAILURE
        }
    }
}
