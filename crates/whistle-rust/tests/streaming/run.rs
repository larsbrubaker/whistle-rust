// Feeding a stream for the streaming tests: the contract clips' names, the one-beam options the
// gate uses, and `feed`, which pushes audio in fixed chunks, then finishes, keeping every
// update, the transcript after each push, the seconds received at each update, the buffer's
// largest size and each update's compute time.

use std::time::{Duration, Instant};

use whistle::{Options, StreamUpdate, Transcript};

use crate::shared::{whistle, RATE};

/// The eight contract clips (the silence clip is tested on its own).
pub const CLIPS: [&str; 8] = [
    "de_anna",
    "en_daniel",
    "en_samantha",
    "es_monica",
    "fr_thomas",
    "it_alice",
    "nl_xander",
    "pl_zosia",
];

/// The live preset, one beam.
pub fn greedy() -> Options {
    Options::streaming()
}

/// Seconds received when each committed word committed, minus its end (finish's included).
pub fn latencies(fed: &Fed) -> Vec<f64> {
    let mut out = Vec::new();
    for (u, &received) in fed.updates.iter().zip(&fed.received) {
        for w in &u.committed {
            out.push(f64::from(received - w.end));
        }
    }
    out
}

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Adjacent word pairs of `text` (normalized) where the second repeats the first or starts
/// with it ("of of", "matter matterhackers"): what a word committed twice looks like.
pub fn doubled(text: &str) -> Vec<String> {
    let words = crate::wer::normalize(text);
    words
        .windows(2)
        .filter(|p| p[1] == p[0] || (p[0].len() >= 3 && p[1].starts_with(&p[0])))
        .map(|p| format!("{} {}", p[0], p[1]))
        .collect()
}

pub struct Fed {
    /// Every push's update, then finish's.
    pub updates: Vec<StreamUpdate>,
    /// Seconds received when each update was returned.
    pub received: Vec<f32>,
    /// `Stream::transcript` after each push (not after finish).
    pub transcripts: Vec<Transcript>,
    /// Everything committed, finish included: the updates' texts joined, their words, the last
    /// language reported (no tokens, `repeated` false).
    pub last: Transcript,
    pub max_buffered: usize,
    pub times: Vec<Duration>,
    /// Every update's `dropped`, summed.
    pub dropped: f32,
}

/// Pushes `samples` in chunks of `chunk` seconds, then finishes.
pub fn feed(samples: &[f32], chunk: f32, options: &Options) -> Fed {
    let mut stream = whistle().stream(options).expect("a stream opens");
    let size = ((chunk * RATE as f32) as usize).max(1);
    let mut fed = Fed {
        updates: Vec::new(),
        received: Vec::new(),
        transcripts: Vec::new(),
        last: stream.transcript(),
        max_buffered: 0,
        times: Vec::new(),
        dropped: 0.0,
    };
    let mut received = 0;
    for piece in samples.chunks(size) {
        let started = Instant::now();
        let update = stream.push(piece).expect("a push decodes");
        fed.times.push(started.elapsed());
        received += piece.len();
        fed.updates.push(update);
        fed.received.push(received as f32 / RATE as f32);
        fed.transcripts.push(stream.transcript());
        fed.max_buffered = fed.max_buffered.max(stream.buffered());
    }
    let started = Instant::now();
    let update = stream.finish().expect("finish decodes");
    fed.times.push(started.elapsed());
    fed.updates.push(update);
    fed.received.push(received as f32 / RATE as f32);
    fed.dropped = fed.updates.iter().map(|u| u.dropped).sum();
    fed.last = Transcript {
        text: joined(&fed),
        language: fed.updates.iter().rev().find_map(|u| u.language.clone()),
        tokens: Vec::new(),
        // The updates do not carry it; the transcripts after each push do.
        repeated: false,
        words: fed
            .updates
            .iter()
            .flat_map(|u| u.committed.clone())
            .collect(),
    };
    fed
}

/// Every update's committed text, joined with single spaces, or none before an update that
/// `continues` the word before it.
pub fn joined(fed: &Fed) -> String {
    let mut out = String::new();
    for u in fed.updates.iter().filter(|u| !u.text.is_empty()) {
        if !out.is_empty() && !u.continues {
            out.push(' ');
        }
        out.push_str(&u.text);
    }
    out
}

/// The character edit distance between `a` and `b`, normalized (lower case, letters and digits,
/// no spaces): how far a stream's committed text is from the full pass's, counting every
/// character lost or repeated.
pub fn char_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = crate::wer::normalize(a).concat().chars().collect();
    let b: Vec<char> = crate::wer::normalize(b).concat().chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut next = vec![i + 1; b.len() + 1];
        for (j, &y) in b.iter().enumerate() {
            next[j + 1] = (row[j] + usize::from(x != y))
                .min(row[j + 1] + 1)
                .min(next[j] + 1);
        }
        row = next;
    }
    row[b.len()]
}
