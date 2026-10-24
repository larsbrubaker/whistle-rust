// The end-to-end contract on every speech clip (CLAUDE.md "Testing framework", net 2),
// teacher-forced: from the WAV file through our VAD window, features and encoder, the cached
// decoder fed the manifest's tokens picks the manifest's next token at every position (step 0
// over the language tags, then over the text ids), except the near-ties listed in
// test-data/near_ties.json (`support::forced`). Checked with the VAD off (`VadMode::Off`, the
// whole clip as the reference decodes it) and with the default trim (`VadMode::GateAndTrim`,
// only the speech span). Every position's margin is printed, so the next coin flip is visible.

use whistle::VadMode;

use crate::shared::{speech_clips, wav, whistle};
use crate::support::forced::check_contract;

fn clips() -> Vec<(String, Vec<u8>)> {
    speech_clips()
        .into_iter()
        .map(|c| (c.clone(), wav(&c)))
        .collect()
}

#[test]
fn every_speech_clip_teacher_forces_to_the_reference_tokens_with_vad_off() {
    check_contract(whistle(), &clips(), VadMode::Off);
}

#[test]
fn every_speech_clip_teacher_forces_to_the_reference_tokens_with_the_default_trim() {
    assert_eq!(whistle::Options::default().vad, VadMode::GateAndTrim);
    check_contract(whistle(), &clips(), VadMode::GateAndTrim);
}
