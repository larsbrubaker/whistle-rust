//! The desktop demo's `ShellHost` (agg-gui-shell's hooks): each idle pass of the event loop
//! applies what the engine and the capture reported (`bridge.rs`) and asks for a layout, so the
//! UI's `Rebuilder`s show it; WAV files the OS asks the app to open (macOS Finder, Dock) are
//! transcribed like a drop. macOS also hands over the app's own command-line arguments that
//! way; `main.rs` already acted on those, so they are skipped. For `--screenshot` it holds the
//! loop, once the window has painted, until a transcript or an error is shown, so the shell's
//! capture (a fixed number of painted frames later) has it.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use agg_gui::App;
use agg_gui_shell::{ShellControl, ShellHost};

use crate::bridge::{Bridge, Startup};

/// The longest the screenshot run holds the loop. Under the shell's capture budget (15 s
/// without a painted frame), so a slow run captures what it has rather than failing.
const HOLD: Duration = Duration::from_secs(12);

pub struct DemoShellHost {
    bridge: Rc<Bridge>,
    /// `--screenshot`: hold for the first outcome (cleared once held).
    hold_for_outcome: bool,
    /// The command line's arguments, which macOS also reports as files to open.
    argv: Vec<OsString>,
}

impl DemoShellHost {
    pub fn new(bridge: Rc<Bridge>, hold_for_outcome: bool, argv: Vec<OsString>) -> DemoShellHost {
        DemoShellHost {
            bridge,
            hold_for_outcome,
            argv,
        }
    }
}

/// `path` made absolute, for comparing paths given in different forms.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The first of `paths` that is an existing `.wav` file and not one of `argv`.
fn first_wav(paths: Vec<PathBuf>, argv: &[OsString]) -> Option<PathBuf> {
    let given: Vec<PathBuf> = argv.iter().map(|a| absolute(Path::new(a))).collect();
    paths.into_iter().find(|p| {
        let wav = p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"));
        wav && p.is_file() && !given.contains(&absolute(p))
    })
}

impl ShellHost for DemoShellHost {
    fn on_idle(&mut self, _app: &mut App, _control: &mut ShellControl<'_>) {
        let mut changed = self.bridge.drain();
        if std::mem::take(&mut self.hold_for_outcome) {
            if !self.bridge.wait_for_outcomes(1, HOLD) {
                eprintln!(
                    "whistle-demo-native: no transcript within {} s; capturing what is shown",
                    HOLD.as_secs()
                );
            }
            changed = true;
        }
        if changed {
            agg_gui::animation::request_layout();
        }
    }

    fn on_open_files(&mut self, _app: &mut App, paths: Vec<PathBuf>) {
        if let Some(path) = first_wav(paths, &self.argv) {
            self.bridge.at_startup(Startup::Open(path));
            agg_gui::animation::request_layout();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opened_files_skip_the_command_line_and_anything_but_a_wav() {
        let scratch = crate::test_support::Scratch::new("open-files");
        let dir = scratch.path();
        let (wav, png, other) = (dir.join("a.wav"), dir.join("shot.png"), dir.join("b.WAV"));
        for f in [&wav, &png, &other] {
            std::fs::write(f, b"x").unwrap();
        }
        let missing = dir.join("gone.wav");
        // macOS hands the app its own arguments as files to open: they were handled already.
        let argv = vec![
            wav.clone().into_os_string(),
            "--screenshot".into(),
            png.clone().into(),
        ];
        assert_eq!(first_wav(vec![wav.clone(), png.clone()], &argv), None);
        assert_eq!(
            first_wav(vec![png, missing, other.clone()], &argv),
            Some(other.clone())
        );
        assert_eq!(first_wav(vec![wav.clone()], &[]), Some(wav));
    }
}
