// Wall-clock benchmark of the library against WAV files: loads the model once (timed), then
// transcribes each clip N times and prints one JSON line per clip. It is the "ours" half of
// `scripts/bench.sh`, which times the closed engine CLI on the same clips.
//
//   cargo run --release -p whistle-rust --example bench -- \
//       [--model PATH] [--runs N] [--beams B] [--threads T] clip.wav...
//
// Std only. Each clip gets one untimed warm-up call (page-ins, first-use thread spawns), then N
// timed calls. `encoder_ms` is the median of features + encoder on the whole clip, timed
// separately from the calls (the library does not expose a per-stage breakdown); the call itself
// may trim silence first, so on clips with leading or trailing silence it is an upper bound.

use std::time::Instant;

use whistle::audio::{features, read_wav};
use whistle::model::Encoder;
use whistle::{Options, Whistle};

struct Args {
    model: String,
    runs: usize,
    beams: usize,
    threads: usize,
    clips: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: "models/whistle.cact".to_string(),
        runs: 10,
        beams: whistle::DEFAULT_BEAMS,
        threads: whistle::threads(),
        clips: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        let number = |s: String, name: &str| {
            s.parse::<usize>()
                .map_err(|_| format!("{name} must be a whole number"))
        };
        match arg.as_str() {
            "--model" => args.model = value("--model")?,
            "--runs" => args.runs = number(value("--runs")?, "--runs")?,
            "--beams" => args.beams = number(value("--beams")?, "--beams")?,
            "--threads" => args.threads = number(value("--threads")?, "--threads")?,
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => args.clips.push(arg),
        }
    }
    if args.runs == 0 || args.clips.is_empty() {
        return Err(
            "usage: bench [--model PATH] [--runs N] [--beams B] [--threads T] clip.wav..."
                .to_string(),
        );
    }
    Ok(args)
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    whistle::set_threads(args.threads);

    // The load, timed the way the engine's is: read the file and bind it.
    let start = Instant::now();
    let bytes = std::fs::read(&args.model).map_err(|e| format!("{}: {e}", args.model))?;
    let model = Whistle::from_cact(&bytes).map_err(|e| e.to_string())?;
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    drop(bytes);
    println!(
        "{{\"kind\":\"load\",\"beams\":{},\"threads\":{},\"load_ms\":{load_ms:.1}}}",
        args.beams,
        whistle::threads()
    );

    let options = Options {
        beams: args.beams,
        ..Options::default()
    };
    for path in &args.clips {
        let wav = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let audio = read_wav(&wav).map_err(|e| format!("{path}: {e}"))?;
        let seconds = audio.samples.len() as f64 / f64::from(audio.sample_rate);

        let mut text = model
            .transcribe_wav(&wav, &options)
            .map_err(|e| format!("{path}: {e}"))?
            .text;
        let mut calls = Vec::with_capacity(args.runs);
        let mut encoders = Vec::with_capacity(args.runs);
        for _ in 0..args.runs {
            let start = Instant::now();
            text = model
                .transcribe_wav(&wav, &options)
                .map_err(|e| format!("{path}: {e}"))?
                .text;
            calls.push(start.elapsed().as_secs_f64() * 1e3);

            let start = Instant::now();
            let feats = features(&audio.samples);
            Encoder::new(model.weights())
                .forward(&feats)
                .map_err(|e| format!("{path}: {e}"))?;
            encoders.push(start.elapsed().as_secs_f64() * 1e3);
        }
        calls.sort_by(f64::total_cmp);
        encoders.sort_by(f64::total_cmp);
        let name = std::path::Path::new(path)
            .file_stem()
            .map_or(path.clone(), |s| s.to_string_lossy().into_owned());
        println!(
            "{{\"kind\":\"clip\",\"clip\":{},\"seconds\":{seconds:.2},\"beams\":{},\"threads\":{},\
             \"runs\":{},\"median_ms\":{:.1},\"min_ms\":{:.1},\"encoder_ms\":{:.1},\"text\":{}}}",
            json_string(&name),
            args.beams,
            whistle::threads(),
            args.runs,
            median(&calls),
            calls[0],
            median(&encoders),
            json_string(&text)
        );
    }
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("bench: {message}");
        std::process::exit(1);
    }
}
