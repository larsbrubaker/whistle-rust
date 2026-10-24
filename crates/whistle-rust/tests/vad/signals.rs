// Inputs for the vad tests: deterministic synthetic non-speech (a fixed-seed xorshift PRNG, no
// crate), level scaling in dBFS, and the speech clips in test-data/clips/ with the span the
// engine's own word timings (test-data/engine/<clip>.json, used only as data) put the words in.

#![allow(dead_code)] // each test file uses its own subset

use std::f32::consts::PI;
use std::ops::Range;

use crate::support::{read_json, repo_root};

pub const RATE: usize = 16000;

/// xorshift64*: a fixed-seed generator so every run builds the same noise.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (x as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub fn rms(samples: &[f32]) -> f32 {
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len().max(1) as f64).sqrt() as f32
}

/// Scales `samples` to an RMS of `dbfs` (0 dBFS is a full-scale square wave).
pub fn at_level(mut samples: Vec<f32>, dbfs: f32) -> Vec<f32> {
    let gain = 10f32.powf(dbfs / 20.0) / rms(&samples);
    samples.iter_mut().for_each(|s| *s *= gain);
    samples
}

pub fn white(seconds: f32, dbfs: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let n = (seconds * RATE as f32) as usize;
    at_level((0..n).map(|_| rng.uniform()).collect(), dbfs)
}

/// Roughly 1/f noise: white noise plus one-pole low-passes at octave-spaced corners, each
/// weighted so the sum falls about 3 dB per octave between 40 Hz and 5 kHz.
pub fn pink(seconds: f32, dbfs: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let corners = [40.0f32, 160.0, 640.0, 2560.0];
    let poles: Vec<f32> = corners
        .iter()
        .map(|c| (-2.0 * PI * c / RATE as f32).exp())
        .collect();
    let mut state = [0.0f32; 4];
    let n = (seconds * RATE as f32) as usize;
    let out = (0..n)
        .map(|_| {
            let w = rng.uniform();
            let mut sum = 0.25 * w;
            for (k, (s, &p)) in state.iter_mut().zip(&poles).enumerate() {
                *s = p * *s + (1.0 - p) * w;
                // Lower corners get more gain so their band carries as much power per octave.
                sum += *s * 2f32.powi(4 - k as i32);
            }
            sum
        })
        .collect();
    at_level(out, dbfs)
}

/// Brown-ish fan noise: white noise through a leaky integrator (corner about 25 Hz), so power
/// falls about 6 dB per octave across the audible band.
pub fn brown(seconds: f32, dbfs: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut s = 0.0f32;
    let n = (seconds * RATE as f32) as usize;
    at_level(
        (0..n)
            .map(|_| {
                s = 0.99 * s + rng.uniform();
                s
            })
            .collect(),
        dbfs,
    )
}

/// Mains hum: `fundamental` with harmonics 2..=10 at amplitude 1/k.
pub fn hum(fundamental: f32, seconds: f32, dbfs: f32) -> Vec<f32> {
    let n = (seconds * RATE as f32) as usize;
    let out = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            (1..=10)
                .map(|k| (2.0 * PI * fundamental * k as f32 * t).sin() / k as f32)
                .sum()
        })
        .collect();
    at_level(out, dbfs)
}

pub fn tone(hz: f32, seconds: f32, dbfs: f32) -> Vec<f32> {
    let n = (seconds * RATE as f32) as usize;
    let out = (0..n)
        .map(|i| (2.0 * PI * hz * i as f32 / RATE as f32).sin())
        .collect();
    at_level(out, dbfs)
}

/// One full-scale-ish impulse in the middle of `seconds` of digital silence.
pub fn click(seconds: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; (seconds * RATE as f32) as usize];
    let mid = out.len() / 2;
    out[mid] = 0.9;
    out[mid + 1] = -0.6;
    out
}

/// A speech clip at 16 kHz and the sample span from its first word's start to its last word's
/// end in the engine's transcript.
pub struct SpeechClip {
    pub name: String,
    pub samples: Vec<f32>,
    pub words: Range<usize>,
}

pub fn speech_clips() -> Vec<SpeechClip> {
    let root = repo_root();
    let mut names: Vec<String> = std::fs::read_dir(root.join("test-data/clips"))
        .expect("test-data/clips")
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|f| f.strip_suffix(".wav").map(str::to_string))
        .filter(|n| n != "silence")
        .collect();
    names.sort();
    assert!(
        names.len() >= 8,
        "expected the eight speech clips, found {names:?}"
    );
    names
        .into_iter()
        .map(|name| {
            let samples = load_clip(&name);
            let engine = read_json(&root.join(format!("test-data/engine/{name}.json")));
            let words = engine["words"].as_array().expect("engine words");
            let at = |w: &serde_json::Value, key: &str| {
                (w[key].as_f64().expect("word timing") * RATE as f64).round() as usize
            };
            let span = at(&words[0], "start")..at(&words[words.len() - 1], "end");
            SpeechClip {
                name,
                samples,
                words: span,
            }
        })
        .collect()
}

pub fn load_clip(name: &str) -> Vec<f32> {
    let path = repo_root().join(format!("test-data/clips/{name}.wav"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let audio = whistle::audio::read_wav(&bytes).expect("clip decodes");
    assert_eq!(audio.sample_rate, whistle::audio::SAMPLE_RATE, "{name}");
    audio.samples
}

/// Samples of `span` covered by `segments`.
pub fn covered(segments: &[Range<usize>], span: &Range<usize>) -> usize {
    segments
        .iter()
        .map(|s| s.end.min(span.end).saturating_sub(s.start.max(span.start)))
        .sum()
}

/// The non-speech cases every test runs, by name.
pub fn non_speech_cases() -> Vec<(String, Vec<f32>)> {
    let mut cases = vec![
        ("empty".to_string(), Vec::new()),
        ("silence.wav".to_string(), load_clip("silence")),
    ];
    for (k, dbfs) in [-60.0f32, -40.0, -20.0].into_iter().enumerate() {
        cases.push((
            format!("white {dbfs} dBFS"),
            white(3.0, dbfs, 11 + k as u64),
        ));
        cases.push((format!("pink {dbfs} dBFS"), pink(3.0, dbfs, 21 + k as u64)));
    }
    cases.push(("hum 50 Hz".to_string(), hum(50.0, 3.0, -30.0)));
    cases.push(("hum 60 Hz".to_string(), hum(60.0, 3.0, -30.0)));
    cases.push(("tone 1 kHz".to_string(), tone(1000.0, 3.0, -20.0)));
    cases.push(("click".to_string(), click(2.0)));
    cases.push(("brown fan".to_string(), brown(3.0, -30.0, 31)));
    cases
}
