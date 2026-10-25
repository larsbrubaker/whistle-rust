// Test area: streaming (`whistle::Stream`, docs/design/streaming.md). `run` feeds a stream in
// fixed chunks and keeps every update; `clips` pins the contract clips against the offline
// decode and chunk-size independence; `behaviour` that committed words never change, times are
// monotonic, silence commits nothing, a long stream stays under 30 s and cancel stops it;
// `engine` compares with the engine's own streaming output (test-data/engine-stream/); `garbled`
// that a pass the repetition guard rolled back commits its words;
// `offsets` streams clips after varying leading silence and hiss (re-reads committed twice, and
// an ignored sweep); `measure` (ignored) prints the per-update cost.

#[path = "support/mod.rs"]
mod support;

#[path = "streaming/behaviour.rs"]
mod behaviour;
#[path = "streaming/clips.rs"]
mod clips;
#[path = "streaming/engine.rs"]
mod engine;
#[path = "streaming/garbled.rs"]
mod garbled;
#[path = "streaming/measure.rs"]
mod measure;
#[path = "streaming/offsets.rs"]
mod offsets;
#[path = "streaming/run.rs"]
mod run;
#[allow(dead_code)] // the clip loader, concatenations and recall helpers are shared
#[path = "windowing/shared.rs"]
mod shared;
#[allow(dead_code)]
#[path = "decoding/wer.rs"]
mod wer;
