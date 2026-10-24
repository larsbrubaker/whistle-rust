// Test area: word timestamps (`Options::word_timestamps`, `Transcript::words`). `capture` pins
// the capture pass to the plain decoder pass bit for bit; `shared` binds the model and reads the
// clips and the engine's words; `engine` holds the words and their times to the engine's on the
// contract clips and on the held-out extra clips; `options` pins that timings off change nothing,
// that silence has no words, and that the detector's trim is added back exactly; `measure`
// (ignored) prints the cost and the probability comparison in docs/design/timestamps.md.

#[path = "support/mod.rs"]
mod support;

#[path = "timestamps/capture.rs"]
mod capture;
#[path = "timestamps/engine.rs"]
mod engine;
#[path = "timestamps/measure.rs"]
mod measure;
#[path = "timestamps/options.rs"]
mod options;
#[path = "timestamps/shared.rs"]
mod shared;
