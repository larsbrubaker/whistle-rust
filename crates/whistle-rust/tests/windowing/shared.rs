// Shared state for the windowing test area: the model bound once per test process, a clip's
// 16 kHz samples (test-data/clips/ or the held-out test-data/clips-extra/), long inputs built
// by concatenating clips with chosen pauses (with each clip's span in the result), the one-beam
// options the area uses (the search is not the point here), the reference decode of one
// window's samples on their own, and word recall against the clips' own transcripts.

use std::ops::Range;
use std::sync::OnceLock;

use whistle::audio::read_wav;
use whistle::{Options, Transcript, VadMode, Whistle, Window};

use crate::wer::normalize;

use crate::support::archive::model_bytes;
use crate::support::repo_root;

pub const RATE: usize = 16_000;

pub fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| Whistle::from_cact(&model_bytes()).expect("whistle.cact loads"))
}

/// The five English clips. Long inputs cycle through all five: the same one or two sentences
/// repeated many times in one 30 s window make the decoder skip repeats (measured while
/// designing this; docs/design/windowing.md), which no real recording does.
pub const ENGLISH: [&str; 5] = ["en_samantha", "en_daniel", "en_names", "en_petg", "en_cura"];

/// `clip`'s samples (every clip is 16 kHz mono), from test-data/clips/ or test-data/clips-extra/.
pub fn clip(name: &str) -> Vec<f32> {
    let mut path = repo_root().join(format!("test-data/clips/{name}.wav"));
    if !path.exists() {
        path = repo_root().join(format!("test-data/clips-extra/{name}.wav"));
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let audio = read_wav(&bytes).expect("clip decodes");
    assert_eq!(audio.sample_rate, RATE as u32, "{name}");
    audio.samples
}

/// `clips` in order, each followed by its pause in seconds (digital silence), and each clip's
/// span in the result.
pub fn concat(clips: &[(&str, f32)]) -> (Vec<f32>, Vec<Range<usize>>) {
    let mut out = Vec::new();
    let mut spans = Vec::new();
    for &(name, pause) in clips {
        let start = out.len();
        out.extend(clip(name));
        spans.push(start..out.len());
        out.resize(out.len() + (pause * RATE as f32) as usize, 0.0);
    }
    (out, spans)
}

/// English for `count` clips: the five English clips in turn with pauses of 0.5, 0.75 and
/// 1 s in turn (12 clips are about 45 s, 18 about 69 s).
pub fn long_english(count: usize) -> (Vec<f32>, Vec<Range<usize>>) {
    let pauses = [0.5, 0.75, 1.0];
    let clips: Vec<(&str, f32)> = (0..count)
        .map(|i| (ENGLISH[i % 5], pauses[i % 3]))
        .collect();
    concat(&clips)
}

pub fn greedy(vad: VadMode, word_timestamps: bool) -> Options {
    Options {
        vad,
        beams: 1,
        word_timestamps,
        ..Options::default()
    }
}

/// One window's samples decoded on their own as the window decode sees them: no detector
/// (the window is already the span to decode), the same other options.
pub fn alone(samples: &[f32], w: &Window, options: &Options) -> Transcript {
    let options = Options {
        vad: VadMode::Off,
        ..options.clone()
    };
    whistle()
        .transcribe(&samples[w.decode.clone()], &options)
        .expect("a window decodes")
}

pub fn seconds(sample: usize) -> f32 {
    sample as f32 / RATE as f32
}

/// `before` English clips with 0.5 s between them, a pause of `pause` seconds, then `after`
/// more: two speech runs that each fit in one window.
pub fn english_pause_english(
    before: usize,
    pause: f32,
    after: usize,
) -> (Vec<f32>, Vec<Range<usize>>) {
    let mut clips: Vec<(&str, f32)> = (0..before).map(|i| (ENGLISH[i % 5], 0.5)).collect();
    clips.last_mut().expect("a clip before the pause").1 = pause;
    clips.extend((0..after).map(|i| (ENGLISH[(before + i) % 5], 0.5)));
    concat(&clips)
}

/// The length, in words, of a longest common subsequence of `a` and `b`.
pub fn lcs(a: &[String], b: &[String]) -> usize {
    lcs_matches(a, b).iter().filter(|&&m| m).count()
}

/// For each word of `a`, whether a longest common subsequence of `a` and `b` uses it.
pub fn lcs_matches(a: &[String], b: &[String]) -> Vec<bool> {
    let (n, m) = (a.len(), b.len());
    let mut t = vec![0usize; (n + 1) * (m + 1)];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[i * (m + 1) + j] = if a[i] == b[j] {
                t[(i + 1) * (m + 1) + j + 1] + 1
            } else {
                t[(i + 1) * (m + 1) + j].max(t[i * (m + 1) + j + 1])
            };
        }
    }
    let mut matched = vec![false; n];
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            matched[i] = true;
            i += 1;
            j += 1;
        } else if t[(i + 1) * (m + 1) + j] >= t[i * (m + 1) + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    matched
}

/// Per clip, the share of its words that `text` keeps in order, against `clips` (spans of
/// `samples`) each decoded on its own with `options`: one longest common subsequence between
/// all the clips' words (`tests/decoding/wer.rs` normalization) and the text's, counted per
/// clip. The reference does not depend on the windows. The decode of a clip changes with its
/// neighbours, so this is no measure of single lost words: it guards against skipped clips,
/// which keep almost none of their words, not against a word lost inside a decoded clip.
pub fn clip_recalls(
    samples: &[f32],
    clips: &[Range<usize>],
    options: &Options,
    text: &str,
) -> Vec<f64> {
    let options = Options {
        word_timestamps: false,
        ..options.clone()
    };
    let per_clip: Vec<Vec<String>> = clips
        .iter()
        .map(|c| {
            let t = whistle()
                .transcribe(&samples[c.clone()], &options)
                .expect("a clip decodes");
            normalize(&t.text)
        })
        .collect();
    let reference: Vec<String> = per_clip.concat();
    let matched = lcs_matches(&reference, &normalize(text));
    let mut at = 0;
    per_clip
        .iter()
        .map(|words| {
            let kept = matched[at..at + words.len()].iter().filter(|&&m| m).count();
            at += words.len();
            kept as f64 / words.len().max(1) as f64
        })
        .collect()
}

/// The smallest of `recalls`, printed with all of them under `label`.
pub fn min_recall(label: &str, recalls: &[f64]) -> f64 {
    let shown: Vec<String> = recalls.iter().map(|r| format!("{r:.2}")).collect();
    println!("{label}: clip recalls [{}]", shown.join(", "));
    recalls.iter().copied().fold(1.0, f64::min)
}

/// Whether every clip span lies inside some window's decoded range.
pub fn every_clip_decoded(windows: &[Window], clips: &[Range<usize>]) -> bool {
    clips.iter().all(|c| {
        windows
            .iter()
            .any(|w| w.decode.start <= c.start && c.end <= w.decode.end)
    })
}
