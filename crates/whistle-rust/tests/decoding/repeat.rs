// The repetition guard (`src/transcribe/repeat.rs`) on real garbled audio:
// de_anna with 128 ms blocks replayed as a late audio callback replays them
// (`support::garble::replayed_blocks`, seed 2) made the decoder read "B" and then "ib" 317
// times, to the 320-token limit, at one beam and at three. With the guard the decode is rolled
// back to the loop's first copy, the token that started the second is banned there, and it
// reads on to the end of the sentence; `Transcript::repeated` says the guard fired. On the
// contract clips it never fires.
//
// Measured: de_anna garbled this way reads "Biby Biby macht den Würfetz zweit Senti Mimi eta
// Höhe." at one beam (one roll-back, EOS after 31 tokens) and "Bibby Bibby macht den Würfetz
// zwei Zentimeter höher." at three (one roll-back, about 63 batched steps against 318
// without the guard). A one-beam scan of 288 garbled variants of the contract clips (blocks
// replayed, repeated, skipped back or shuffled; four seeds; 64, 128 and 250 ms blocks; the
// detector on and off: 576 decodes) had 5 decodes run to the length limit and 6 repeat a
// phrase until EOS without the guard; with it the guard fires on exactly those 11, none runs to
// the limit, and the other 565 decodes are unchanged, including repetitions below the
// thresholds ("He's the breakfast." five times, "drie" six).

use whistle::audio::read_wav;
use whistle::Options;

use crate::clips::{beams, contract_clips, transcribe, whistle};
use crate::support::garble::{longest_repetition, replayed_blocks};

fn garbled_de_anna() -> Vec<f32> {
    let clip = contract_clips()
        .into_iter()
        .find(|c| c.name == "de_anna")
        .expect("de_anna is a contract clip");
    replayed_blocks(&read_wav(&clip.wav).unwrap().samples, 2, 2048)
}

#[test]
fn garbled_audio_is_rolled_back_past_a_repetition_loop_and_read_on() {
    let samples = garbled_de_anna();
    let cases = [
        (1, "Biby Biby macht den Würfetz zweit Senti Mimi eta Höhe."),
        (3, "Bibby Bibby macht den Würfetz zwei Zentimeter höher."),
    ];
    for (beams, want) in cases {
        let options = Options {
            beams,
            ..Options::default()
        };
        let got = whistle().transcribe(&samples, &options).unwrap();
        let (copies, block, span) = longest_repetition(&got.tokens);
        assert!(
            copies < 5 || span < 30,
            "{beams} beams: {copies} copies of a {block}-token block in {:?}",
            got.text
        );
        assert_eq!(got.text, want, "{beams} beams");
        assert!(got.repeated, "{beams} beams");
    }
}

#[test]
fn the_guard_never_fires_on_the_contract_clips() {
    for clip in contract_clips() {
        for n in [1, 3] {
            assert!(
                !transcribe(&clip, &beams(n)).repeated,
                "{} at {n}",
                clip.name
            );
        }
    }
}
