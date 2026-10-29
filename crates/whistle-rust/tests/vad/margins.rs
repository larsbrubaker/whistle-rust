// How far each default threshold sits from flipping a case. `threshold_flip_points` is the
// measurement, its table on the test (ignored: `cargo test --test vad threshold_flip_points --
// --ignored --nocapture`); `defaults_hold_their_measured_margins` and
// `min_modulation_db_sits_between_fan_noise_and_noisy_speech` pin parts of it, so a deliberate
// change to the features does not flip a clip unnoticed.

use whistle::vad::{speech_segments_with, VadConfig};

use crate::signals::{non_speech_cases, speech_clips};
use crate::speech::{attenuated, span_failure, with_noise};

type Setter = fn(&mut VadConfig, f32);

/// A case: its name, its samples, and the engine's word span for speech (None for non-speech).
type Case = (String, Vec<f32>, Option<std::ops::Range<usize>>);

fn params() -> Vec<(&'static str, Setter, f32, f32)> {
    vec![
        ("margin_on_db", |c, v| c.margin_on_db = v, 0.0, 40.0),
        ("margin_off_db", |c, v| c.margin_off_db = v, 0.0, 30.0),
        (
            "min_modulation_db",
            |c, v| c.min_modulation_db = v,
            0.0,
            20.0,
        ),
        (
            "min_modulation_db (energy gate off)",
            |c, v| {
                c.min_modulation_db = v;
                c.margin_on_db = 0.0;
                c.margin_off_db = 0.0;
                c.min_modulation_span_s = 0.0;
            },
            0.0,
            20.0,
        ),
        (
            "margin_on_db (modulation gate off)",
            |c, v| {
                c.margin_on_db = v;
                c.min_modulation_db = 0.0;
            },
            0.0,
            40.0,
        ),
        (
            "min_modulation_span_s",
            |c, v| c.min_modulation_span_s = v,
            0.0,
            0.5,
        ),
        ("merge_gap_s", |c, v| c.merge_gap_s = v, 0.0, 1.0),
        (
            "absolute_floor_db",
            |c, v| c.absolute_floor_db = v,
            -130.0,
            -10.0,
        ),
    ]
}

/// Inputs for the sweep: (name, samples, Some(word span) for speech, None for non-speech).
fn cases() -> Vec<Case> {
    let mut out: Vec<_> = non_speech_cases()
        .into_iter()
        .map(|(n, s)| (n, s, None))
        .collect();
    for clip in speech_clips() {
        let w = Some(clip.words.clone());
        out.push((clip.name.clone(), clip.samples.clone(), w.clone()));
        out.push((
            format!("{} snr10", clip.name),
            with_noise(&clip.samples, 10.0, 7),
            w.clone(),
        ));
        out.push((
            format!("{} -30", clip.name),
            attenuated(&clip.samples, 30.0),
            w,
        ));
    }
    out
}

fn passes(c: &VadConfig, samples: &[f32], words: &Option<std::ops::Range<usize>>) -> bool {
    let seg = speech_segments_with(c, samples);
    match words {
        None => seg.is_empty(),
        Some(w) => span_failure(&seg, w).is_none(),
    }
}

fn all_pass(c: &VadConfig, cases: &[Case], speech: bool) -> Vec<String> {
    cases
        .iter()
        .filter(|(_, _, w)| w.is_some() == speech)
        .filter(|(_, s, w)| !passes(c, s, w))
        .map(|(n, _, _)| n.clone())
        .collect()
}

#[test]
fn defaults_hold_their_measured_margins() {
    let cases = cases();
    // Non-speech stays empty with the start margin 3 dB more permissive (it flips below 10.5 dB
    // with the modulation gate off), and speech is still found with it 5 dB stricter (it flips
    // above 23.5 dB) or with the absolute floor at -69 dB (the 30 dB quieter nl_xander flips above
    // -67 dB).
    let mut loose = VadConfig::default();
    loose.margin_on_db -= 3.0;
    loose.min_modulation_db = 0.0;
    assert_eq!(all_pass(&loose, &cases, false), Vec::<String>::new());
    let mut strict = VadConfig::default();
    strict.margin_on_db += 5.0;
    assert_eq!(all_pass(&strict, &cases, true), Vec::<String>::new());
    let high_floor = VadConfig {
        absolute_floor_db: -69.0,
        ..VadConfig::default()
    };
    assert_eq!(all_pass(&high_floor, &cases, true), Vec::<String>::new());
}

#[test]
fn min_modulation_db_sits_between_fan_noise_and_noisy_speech() {
    // The thinnest margin: with the energy margins off, fan noise is rejected only from 2.0 dB;
    // speech at 10 dB SNR is found only up to 3.0 dB. The default 2.5 sits 0.5 dB from each.
    let cases = cases();
    let loose = VadConfig {
        min_modulation_db: 2.0,
        margin_on_db: 0.0,
        margin_off_db: 0.0,
        min_modulation_span_s: 0.0,
        ..VadConfig::default()
    };
    assert_eq!(all_pass(&loose, &cases, false), Vec::<String>::new());
    let strict = VadConfig {
        min_modulation_db: 3.0,
        ..VadConfig::default()
    };
    assert_eq!(all_pass(&strict, &cases, true), Vec::<String>::new());
}

/// Sweeps one threshold at a time from the defaults and prints every value at which a case
/// flips. Measured with the defaults as shipped (each sweep's step in brackets):
///
/// | threshold | non-speech is rejected | speech is found | default |
/// | --- | --- | --- | --- |
/// | `margin_on_db` [0.5] | from 10.5 with modulation off (brown fan); at any value with it on | up to 23.5 (es_monica, 10 dB SNR) | 15 |
/// | `margin_off_db` [0.375] | at any value | up to 5.25 (nl_xander, 10 dB SNR) | 3 |
/// | `min_modulation_db` [0.25] | from 2.0 with the energy margins off (brown fan); at any value with them on | up to 3.0 (en_samantha, 10 dB SNR) | 2.5 |
/// | `absolute_floor_db` [1.5] | at any value | up to −67 (nl_xander, 30 dB quieter) | −75 |
/// | `min_modulation_span_s` [0.006] | at any value | up to 0.38 (it_alice, 30 dB quieter) | 0.1 |
/// | `merge_gap_s` [0.0125] | at any value | at any value in 0–1 s | 0.3 |
///
/// Non-speech is rejected by two independent gates (rise and modulation), so a row whose
/// non-speech column names a flip point measured it with the other gate switched off. The
/// thinnest margin is modulation alone: fan noise is rejected from 2.0 dB and speech at 10 dB
/// SNR found up to 3.0 dB, so the default sits midway and the rise gate carries most of the
/// work.
#[test]
#[ignore]
fn threshold_flip_points() {
    let cases = cases();
    for (pname, set, lo, hi) in params() {
        println!("== {pname}");
        for (name, samples, words) in &cases {
            let steps = 80;
            let mut ok = Vec::new();
            for i in 0..=steps {
                let v = lo + (hi - lo) * i as f32 / steps as f32;
                let mut c = VadConfig::default();
                set(&mut c, v);
                ok.push((v, passes(&c, samples, words)));
            }
            // Every value at which the outcome changes, as "pass→fail@v" or "fail→pass@v".
            let flips: Vec<String> = ok
                .windows(2)
                .filter(|w| w[0].1 != w[1].1)
                .map(|w| {
                    let dir = if w[0].1 { "pass→fail" } else { "fail→pass" };
                    format!("{dir}@{}", w[1].0)
                })
                .collect();
            let first = if ok[0].1 { "pass" } else { "fail" };
            println!("{name:24} {first} at {lo}, flips {flips:?}");
        }
    }
}
