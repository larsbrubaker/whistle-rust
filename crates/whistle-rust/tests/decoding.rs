// Test area: decoding (`Options::beams`, `Options::keywords`). `clips` loads the model, the
// contract clips with their true texts and engine transcripts, the extra keyword clips, and
// re-scores a transcript; `wer` scores word error rates. The gate runs a few pinned checks at
// one beam and at three: `greedy` (one beam is the greedy contract; the option checks),
// `beams` (determinism, the measured score against greedy, the pinned WER) and `keywords`
// (biasing on the extra clips, irrelevant keywords harmless). The slow measurements (the bonus
// sweep, the score check at every beam count, timing) are `#[ignore]` tests that print the
// tables in docs/design/decoding.md; the WER table per ranking is `src/transcribe/tests.rs`.

#[path = "support/mod.rs"]
mod support;

#[path = "decoding/beams.rs"]
mod beams;
#[path = "decoding/clips.rs"]
mod clips;
#[path = "decoding/greedy.rs"]
mod greedy;
#[path = "decoding/keywords.rs"]
mod keywords;
#[path = "decoding/timing.rs"]
mod timing;
#[path = "decoding/wer.rs"]
mod wer;
