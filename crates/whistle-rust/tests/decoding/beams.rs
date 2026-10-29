// The beam search on the contract clips: it is deterministic; on the clips measured it found
// no transcript that the mean score it ranks by puts below greedy's (a measured result, not a
// guarantee: a beam can prune greedy's path); and its normalized word error rate against the
// true texts stays within the bound measured at one beam and at three.

use crate::clips::{beams, contract_clips, log_probability, transcribe, whole_clip, Clip};
use crate::wer::{normalize, raw_wer, wer};

/// The normalized corpus WER against the true texts measured 2026-10-24 (72 words; see
/// `ranking_word_error_rates` in `src/transcribe/tests.rs`), a bound so an improvement passes:
/// one beam 14 edits, three 12.
const PINNED: [(usize, f64); 2] = [(1, 14.0 / 72.0), (3, 12.0 / 72.0)];

#[test]
fn beam_search_is_deterministic() {
    let clips = contract_clips();
    let clip = clips.iter().find(|c| c.name == "pl_zosia").unwrap();
    assert_eq!(transcribe(clip, &beams(3)), transcribe(clip, &beams(3)));
}

/// Checks the mean score of `n` beams against greedy on `clips`, printing both.
fn scored_no_lower_than_greedy(clips: &[Clip], counts: &[usize]) {
    for clip in clips {
        let greedy = transcribe(clip, &whole_clip(1));
        let lang = greedy.language.clone().unwrap();
        let floor = log_probability(clip, &lang, &greedy.tokens);
        for &n in counts {
            let got = transcribe(clip, &whole_clip(n));
            assert_eq!(got.language, greedy.language, "{} beams {n}", clip.name);
            let score = log_probability(clip, &lang, &got.tokens);
            println!(
                "{} beams {n}: mean {score:.4} vs greedy {floor:.4}",
                clip.name
            );
            assert!(
                score >= floor - 1e-9,
                "{} beams {n}: {score} < greedy {floor}",
                clip.name
            );
        }
    }
}

#[test]
fn three_beams_scored_no_lower_than_greedy_on_the_clips_where_they_differ() {
    let clips: Vec<Clip> = contract_clips()
        .into_iter()
        .filter(|c| c.name == "pl_zosia" || c.name == "fr_thomas")
        .collect();
    scored_no_lower_than_greedy(&clips, &[3]);
}

#[test]
#[ignore = "measurement: every beam count against greedy"]
fn every_beam_count_scored_no_lower_than_greedy_on_every_clip() {
    scored_no_lower_than_greedy(&contract_clips(), &[2, 3, 5, 8]);
}

fn pinned_wer(n: usize, bound: f64) {
    let clips = contract_clips();
    let (mut edits, mut words) = (0.0, 0.0);
    for clip in &clips {
        let text = transcribe(clip, &beams(n)).text;
        let w = normalize(&clip.truth).len() as f64;
        edits += wer(&clip.truth, &text) * w;
        words += w;
        println!(
            "beams {n} {:12} {:.3} (raw {:.3}) {text}",
            clip.name,
            wer(&clip.truth, &text),
            raw_wer(&clip.truth, &text)
        );
    }
    let corpus = edits / words;
    println!("beams {n}: corpus WER {corpus:.4}");
    assert!(
        corpus <= bound + 1e-9,
        "beams {n}: WER {corpus:.4} > {bound:.4}"
    );
}

#[test]
fn one_beam_word_error_rate_stays_within_the_measured_bound() {
    pinned_wer(PINNED[0].0, PINNED[0].1);
}

#[test]
fn three_beam_word_error_rate_stays_within_the_measured_bound() {
    pinned_wer(PINNED[1].0, PINNED[1].1);
}
