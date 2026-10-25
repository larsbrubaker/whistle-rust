//! The desktop demo's command line (read by `main.rs`), and the default thread count for the
//! library's projections. Hand-parsed: a handful of flags does not need a parser crate.

use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "\
whistle-demo-native: live speech-to-text with whistle-rust, on the desktop

usage: whistle-demo-native [options] [FILE.wav]

  FILE.wav, --wav FILE   transcribe this WAV once the model is ready
  --model FILE           load whistle.cact from FILE (only from there)
                         default: the checkout's models/whistle.cact, then the cache,
                         then a download of the pinned file (checked by size and SHA-256)
  --threads N            worker threads for the model (default: one per core but one, 1 to 4)
  --mic-from-wav FILE    play FILE through the live-audio path in real time, in place of the
                         microphone
  --record               start recording once the model is ready (from the microphone, or
                         from --mic-from-wav)
  --screenshot FILE      once a transcript (or an error) is shown, save the window as a PNG
                         and exit
  -h, --help             this text";

/// What the command line asked for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub help: bool,
    pub model: Option<PathBuf>,
    pub threads: Option<usize>,
    pub wav: Option<PathBuf>,
    pub mic_from_wav: Option<PathBuf>,
    pub record: bool,
    pub screenshot: Option<PathBuf>,
}

impl Args {
    /// Parse the arguments after the program name; an error says what is wrong.
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Args, String> {
        let mut out = Args::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let mut value = |flag: &str| {
                args.next()
                    .map(PathBuf::from)
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match arg.to_str() {
                Some("-h" | "--help") => out.help = true,
                Some("--model") => out.model = Some(value("--model")?),
                Some("--wav") => out.wav = Some(value("--wav")?),
                Some("--mic-from-wav") => out.mic_from_wav = Some(value("--mic-from-wav")?),
                Some("--screenshot") => out.screenshot = Some(value("--screenshot")?),
                Some("--record") => out.record = true,
                Some("--threads") => {
                    let n = value("--threads")?;
                    let n = n.to_str().and_then(|s| s.parse::<usize>().ok());
                    match n {
                        Some(n) if (1..=whistle::MAX_THREADS).contains(&n) => out.threads = Some(n),
                        _ => {
                            return Err(format!(
                                "--threads takes a number from 1 to {}",
                                whistle::MAX_THREADS
                            ))
                        }
                    }
                }
                Some(flag) if flag.starts_with('-') => {
                    return Err(format!("unknown option {flag}"))
                }
                // A bare path: what "open with" on Windows and Linux passes.
                _ if out.wav.is_none() => out.wav = Some(PathBuf::from(arg)),
                _ => return Err("only one WAV file at a time".into()),
            }
        }
        Ok(out)
    }
}

/// Worker threads for the model on a machine with `cores` logical cores: one per core but one
/// (so the UI thread keeps a core while a pass runs), at least 1, at most 4, the cap of the
/// library's own default (`whistle::threads`).
pub fn default_threads(cores: usize) -> usize {
    cores.saturating_sub(1).clamp(1, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        Args::parse(args.iter().map(OsString::from))
    }

    #[test]
    fn flags_and_a_bare_path_parse() {
        let a = parse(&["--model", "m.cact", "--threads", "2", "clip.wav"]).unwrap();
        assert_eq!(a.model, Some(PathBuf::from("m.cact")));
        assert_eq!(a.threads, Some(2));
        assert_eq!(a.wav, Some(PathBuf::from("clip.wav")));
        let a = parse(&[
            "--mic-from-wav",
            "c.wav",
            "--record",
            "--screenshot",
            "s.png",
        ])
        .unwrap();
        assert!(a.record);
        assert_eq!(a.screenshot, Some(PathBuf::from("s.png")));
        // Without --mic-from-wav, --record records from the microphone.
        assert!(parse(&["--record"]).unwrap().record);
        assert_eq!(parse(&[]).unwrap(), Args::default());
    }

    #[test]
    fn bad_command_lines_say_what_is_wrong() {
        assert_eq!(parse(&["--model"]).unwrap_err(), "--model needs a value");
        assert!(parse(&["--threads", "0"]).unwrap_err().contains("1 to"));
        assert!(parse(&["--threads", "x"]).is_err());
        assert_eq!(parse(&["--bogus"]).unwrap_err(), "unknown option --bogus");
        assert!(parse(&["a.wav", "b.wav"]).is_err());
    }

    #[test]
    fn the_default_leaves_a_core_and_stops_at_four() {
        assert_eq!(default_threads(1), 1);
        assert_eq!(default_threads(2), 1);
        assert_eq!(default_threads(4), 3);
        assert_eq!(default_threads(5), 4);
        assert_eq!(default_threads(16), 4);
    }
}
