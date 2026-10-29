// Test area: windowing past 30 s (`whistle::plan_windows`; the cuts in `src/transcribe/window.rs`,
// the stitch in `long.rs`). `shared`
// binds the model and builds long inputs from the clips; `pauses` pins that a long input with
// pauses is cut inside them and decodes as its windows do alone, with word times offset into the
// input; `hard_cut` the fallback for speech with no pause (overlapping windows, words kept by
// midpoint); `limits` that 30 s is still one pass and over 10 minutes is an error; `language`
// that the first window's language is forced on the rest; `measure` (ignored) prints the cost.

#[path = "support/mod.rs"]
mod support;

#[path = "windowing/hard_cut.rs"]
mod hard_cut;
#[path = "windowing/language.rs"]
mod language;
#[path = "windowing/limits.rs"]
mod limits;
#[path = "windowing/measure.rs"]
mod measure;
#[path = "windowing/pauses.rs"]
mod pauses;
#[path = "windowing/shared.rs"]
mod shared;
#[allow(dead_code)] // only the normalizer is used here
#[path = "decoding/wer.rs"]
mod wer;
