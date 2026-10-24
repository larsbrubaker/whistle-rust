// Unit tests for the word aligner (`align`): the median filter, DTW's monotonic path and the
// token spans it gives, the word grouping's edge cases (a trailing lone `▁`, text opening with a
// punctuation word), the cancel check before the alignment pass, and (ignored, the measurement
// behind `ALIGNMENT_HEADS`) a sweep that scores every (layer, head) against the engine's word
// times on the contract clips.

use std::borrow::Cow;
use std::sync::OnceLock;

use super::*;
use crate::error::Error;
use crate::model::HeadFilter;
use crate::transcribe::search::Ranking;
use crate::transcribe::{AlignmentPass, Cancel, Options, Stage, Whistle};

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the workspace root")
        .to_path_buf()
}

fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| {
        let bytes = std::fs::read(repo().join("models/whistle.cact")).expect("whistle.cact");
        Whistle::from_cact(&bytes).expect("whistle.cact loads")
    })
}

#[test]
fn median_filter_removes_a_one_frame_spike_and_keeps_steps() {
    let row = [0.0, 0.0, 9.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    assert_eq!(
        median_filter(&row, 3),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn dtw_follows_the_cheapest_monotonic_path() {
    // Token 0 is cheap on frames 0-1, token 1 on frames 2-3.
    let cost = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0];
    assert_eq!(dtw(&cost, 2, 4), vec![(0, 0), (0, 1), (1, 2), (1, 3)]);
}

#[test]
fn token_spans_cover_every_frame_in_order() {
    let m = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    assert_eq!(token_spans(&m, 3, 5), vec![(0, 2), (2, 4), (4, 5)]);
}

#[test]
fn a_trailing_lone_space_piece_joins_the_last_word() {
    let t = whistle().tokenizer();
    let space = (0..t.piece_count() as u32)
        .find(|&id| t.piece(id) == Some("▁"))
        .expect("a lone ▁ piece");
    let mut tokens = t.encode("Make the cube");
    tokens.push(space);
    let groups = word_groups(t, &tokens);
    assert_eq!(groups.len(), 3, "{groups:?}");
    assert_eq!(groups[2].1, tokens.len());
}

#[test]
fn text_opening_with_a_punctuation_word_joins_it_to_the_first_word() {
    let t = whistle().tokenizer();
    let tokens = t.encode("« Bonjour »");
    let words: Vec<String> = word_groups(t, &tokens)
        .iter()
        .map(|g| t.decode(&tokens[g.0..g.1]).trim().to_string())
        .collect();
    assert_eq!(words, ["« Bonjour", "»"]);
}

#[test]
fn a_cancel_before_the_alignment_pass_stops_the_call() {
    let wav = std::fs::read(repo().join("test-data/clips/nl_xander.wav")).unwrap();
    let audio = crate::audio::read_wav(&wav).unwrap();
    let cancel = Cancel::new();
    let options = Options {
        cancel: Some(cancel.clone()),
        beams: 1,
        word_timestamps: true,
        ..Options::default()
    };
    let mut last = None;
    let samples = Cow::Owned(audio.samples);
    let result = whistle().run(samples, audio.sample_rate, &options, &mut |s| {
        if s == Stage::Align {
            cancel.cancel();
        }
        last = Some(s);
    });
    assert_eq!(result.err(), Some(Error::Cancelled));
    assert_eq!(last, Some(Stage::Align));
}

type Pass = (String, AlignmentPass, Vec<f32>);

/// Each speech clip's alignment pass with every head (3 beams, the default trim) and the
/// engine's word starts.
fn passes(whistle: &Whistle) -> Vec<Pass> {
    let clips = [
        "de_anna",
        "en_daniel",
        "en_samantha",
        "es_monica",
        "fr_thomas",
    ];
    let clips = clips.iter().chain(&["it_alice", "nl_xander", "pl_zosia"]);
    let options = Options {
        word_timestamps: true,
        ..Options::default()
    };
    clips
        .map(|clip| {
            let wav = std::fs::read(repo().join(format!("test-data/clips/{clip}.wav"))).unwrap();
            let audio = crate::audio::read_wav(&wav).unwrap();
            let samples = Cow::Owned(audio.samples);
            let (_, pass) = whistle
                .run_full(
                    samples,
                    audio.sample_rate,
                    &options,
                    &mut |_| {},
                    Ranking::Mean,
                    HeadFilter::All,
                )
                .unwrap();
            let path = repo().join(format!("test-data/engine/{clip}.json"));
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let starts = v["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w["start"].as_f64().unwrap() as f32)
                .collect();
            (clip.to_string(), pass.unwrap(), starts)
        })
        .collect()
}

/// Mean and worst absolute start error in frames over every clip's words, for `heads`.
fn start_error(whistle: &Whistle, passes: &[Pass], heads: &[(usize, usize)]) -> (f32, f32) {
    let (mut sum, mut n, mut worst) = (0.0f32, 0usize, 0.0f32);
    for (_, p, engine) in passes {
        let a = Alignment {
            cross: &p.cross,
            logits: &p.logits,
            tokens: &p.tokens,
            text_ids: whistle.first_tag,
            offset: p.offset,
            duration: p.duration,
        };
        for (w, e) in words(&whistle.tokenizer, &a, heads).iter().zip(engine) {
            let d = (w.start - e).abs() / FRAME_SECONDS;
            sum += d;
            worst = worst.max(d);
            n += 1;
        }
    }
    (sum / n.max(1) as f32, worst)
}

/// How monotonic (share of text rows whose argmax frame does not go back) and sharp (mean peak
/// probability) one head's text rows are.
fn shape(passes: &[Pass], l: usize, h: usize) -> (f32, f32) {
    let (mut mono, mut peak, mut n) = (0usize, 0.0f32, 0usize);
    for (_, p, _) in passes {
        let mut last = 0;
        for i in 0..p.tokens.len() {
            let r = p.cross.row(l, h, TEXT_ROW + i).unwrap();
            let (arg, max) =
                r.iter().enumerate().fold(
                    (0, f32::MIN),
                    |b, (j, &v)| {
                        if v > b.1 {
                            (j, v)
                        } else {
                            b
                        }
                    },
                );
            mono += usize::from(arg >= last);
            last = arg;
            peak += max;
            n += 1;
        }
    }
    (mono as f32 / n as f32, peak / n as f32)
}

#[test]
#[ignore = "measurement: prints the per-head scores behind ALIGNMENT_HEADS (release, ~10 s)"]
fn alignment_head_sweep() {
    let whistle = whistle();
    let passes = passes(whistle);
    let mut scored = Vec::new();
    for &(l, h) in &passes[0].1.cross.heads {
        let (err, worst) = start_error(whistle, &passes, &[(l, h)]);
        let (mono, peak) = shape(&passes, l, h);
        println!("L{l} H{h}: start err {err:.2} fr (worst {worst:.1}), monotonic {mono:.2}, peak {peak:.2}");
        scored.push((err, (l, h)));
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    for k in 1..=8 {
        let set: Vec<_> = scored[..k].iter().map(|s| s.1).collect();
        let (err, worst) = start_error(whistle, &passes, &set);
        println!("top {k} {set:?}: start err {err:.2} fr (worst {worst:.1})");
    }
    let (err, worst) = start_error(whistle, &passes, ALIGNMENT_HEADS);
    println!("ALIGNMENT_HEADS {ALIGNMENT_HEADS:?}: start err {err:.2} fr (worst {worst:.1})");
}
