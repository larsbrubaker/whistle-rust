// Keyword biasing: on the extra clips (names, places, product words) biasing finds at least the
// measured number of keywords at one beam and at three, including every keyword the engine
// finds with its own keyword option; on the eight contract clips, irrelevant keywords leave the
// transcript exactly as it was (the port's greedy snapshot at one beam, test-data/snapshots/,
// and the engine's text at three). The bonus sweep and the counts without biasing are an
// ignored measurement.
//
// A keyword counts as found when its normalized words appear in order in the normalized text.
// Without biasing the port finds 4 of the 13 at one beam and 3 at three or five; the engine
// finds 3 without its keyword option and 7 with it. Missed by both: Siobhan ("Shivan"), Niamh,
// Pforzheim ("Fort Simps") and Grzegorza, whose acoustic scores sit too far from the spelling
// for a bonus that leaves ordinary words alone.

use whistle::Options;

use crate::clips::{beams, contract_clips, extra_clips, transcribe};
use crate::support::forced::snapshot;
use crate::wer::normalize;

/// Keywords found with biasing at the default bonus, measured 2026-10-24 (of 13), for one beam
/// and for three. Three beams re-pinned from 9 to 8 when the deterministic `exp` landed: the
/// lost keyword is "Ultimaker" on en_cura ("the ultimate cura settings", which is also the
/// engine's text with the keyword). Unbiased, "ultimaker" trails "ultimate" by 2.4 and 6.7
/// logits over its two tokens, so the bonus barely carries it, and a last-bit change moves the
/// beam onto another path: the name before it changed too ("Kneeov" to "Kniov", 6926 against
/// 8095, teacher-forced margin +0.10).
const FOUND: [(usize, usize); 2] = [(1, 9), (3, 8)];

/// Whether the keyword's words appear, in order, in the normalized transcript (case and
/// punctuation do not count: "cura" is Cura).
fn contains(text: &str, keyword: &str) -> bool {
    let words = normalize(text);
    let key = normalize(keyword);
    words.windows(key.len()).any(|w| w == key.as_slice())
}

fn with_keywords(beams: usize, keywords: &[String], bonus: f32) -> Options {
    Options {
        beams,
        keywords: keywords.to_vec(),
        keyword_bonus: bonus,
        ..Options::default()
    }
}

fn biased(n: usize, want: usize) {
    let mut missing = Vec::new();
    let mut found = 0;
    for clip in extra_clips() {
        let text = transcribe(
            &clip,
            &with_keywords(n, &clip.keywords, whistle::DEFAULT_KEYWORD_BONUS),
        )
        .text;
        println!("beams {n} {:9} {text:?}", clip.name);
        for k in &clip.keywords {
            found += usize::from(contains(&text, k));
            if contains(&clip.engine_keywords, k) && !contains(&text, k) {
                missing.push(format!(
                    "{}: {k:?} (found by the engine) not in {text:?}",
                    clip.name
                ));
            }
        }
    }
    println!("beams {n}: {found}/13 keywords");
    assert!(missing.is_empty(), "{}", missing.join("\n"));
    assert!(
        found >= want,
        "beams {n}: {found} keywords, measured {want}"
    );
}

#[test]
fn one_beam_biasing_finds_the_measured_keywords_and_every_one_the_engine_finds() {
    biased(FOUND[0].0, FOUND[0].1);
}

#[test]
fn three_beam_biasing_finds_the_measured_keywords_and_every_one_the_engine_finds() {
    biased(FOUND[1].0, FOUND[1].1);
}

fn irrelevant() -> Vec<String> {
    vec!["Siobhan".to_string(), "PETG".to_string()]
}

/// One beam without keywords is the greedy snapshot (test-data/snapshots/greedy.json), so that
/// is the text irrelevant keywords must leave. Re-pinned from the manifest text when the
/// deterministic `exp` moved pl_zosia's listed near-tie (test-data/near_ties.json, margin
/// +0.0668 under Apple's libm, -0.0143 deterministic): the greedy text there is now "Przez ten
/// element do 10 ml." against the manifest's "Przez im ten element do 10 ml.".
#[test]
fn irrelevant_keywords_leave_one_beam_on_the_greedy_snapshot_text() {
    let greedy = snapshot("greedy");
    for clip in contract_clips() {
        let got = transcribe(
            &clip,
            &with_keywords(1, &irrelevant(), whistle::DEFAULT_KEYWORD_BONUS),
        );
        assert_eq!(
            got.text,
            greedy[&clip.name]["text"].as_str().unwrap(),
            "{}",
            clip.name
        );
    }
}

/// Three beams without keywords give the engine's transcript on every contract clip (measured
/// 2026-10-24, `whistle::DEFAULT_BEAMS`), so that is the text irrelevant keywords must leave.
#[test]
fn irrelevant_keywords_leave_three_beams_on_the_engine_text() {
    for clip in contract_clips() {
        let got = transcribe(
            &clip,
            &with_keywords(3, &irrelevant(), whistle::DEFAULT_KEYWORD_BONUS),
        );
        assert_eq!(got.text, clip.engine, "{}", clip.name);
    }
}

/// The bonus strength against keywords found (of 13 on the extra clips) and contract clips
/// changed by the irrelevant keywords, and the counts without biasing:
/// `cargo test --release --test decoding sweep -- --ignored --nocapture`. Measured with the
/// platform libm (the deterministic math since moves three beams at 3.0 from 9 to 8 found):
///
/// | bonus | found, 1 / 3 / 5 beams | contract clips changed, 1 / 3 / 5 beams |
/// |------:|-----------------------:|----------------------------------------:|
/// | 1     | 5 / 6 / 6              | 0 / 0 / 0                               |
/// | 2     | 7 / 7 / 8              | 0 / 0 / 0                               |
/// | 3     | 9 / 9 / 9              | 0 / 0 / 0                               |
/// | 4     | 10 / 10 / 8            | 0 / 0 / 1                               |
/// | 5     | 12 / 11 / 10           | 1 / 1 / 1                               |
/// | 6     | 12 / 13 / 8            | 2 / 2 / 1                               |
///
/// Larger bonuses at 5 beams find fewer: biased partial paths crowd the beam and are retracted.
#[test]
#[ignore = "measurement of the keyword bonus strength"]
fn keyword_bonus_sweep() {
    let contract = contract_clips();
    let extra = extra_clips();
    for n in [1, 3, 5] {
        let plain: usize = extra
            .iter()
            .map(|c| {
                let text = transcribe(c, &beams(n)).text;
                c.keywords.iter().filter(|k| contains(&text, k)).count()
            })
            .sum();
        println!("beams {n}: {plain}/13 keywords without biasing");
    }
    let engine = |f: fn(&crate::clips::Clip) -> &str| -> usize {
        extra
            .iter()
            .map(|c| c.keywords.iter().filter(|k| contains(f(c), k)).count())
            .sum()
    };
    println!(
        "engine: {}/13 without its keyword option, {}/13 with",
        engine(|c| &c.engine),
        engine(|c| &c.engine_keywords)
    );
    for bonus in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        for n in [1, 3, 5] {
            let mut found = 0;
            for clip in &extra {
                let text = transcribe(clip, &with_keywords(n, &clip.keywords, bonus)).text;
                found += clip.keywords.iter().filter(|k| contains(&text, k)).count();
            }
            let changed = contract
                .iter()
                .filter(|c| {
                    transcribe(c, &with_keywords(n, &irrelevant(), bonus)).text
                        != transcribe(c, &beams(n)).text
                })
                .count();
            println!(
                "bonus {bonus} beams {n}: keywords {found}/13, contract clips changed {changed}/8"
            );
        }
    }
}
