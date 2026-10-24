// Test area: transcription (`whistle::Whistle`, and the cached decoder `model::DecoderState` it
// runs). `shared` binds the model once and loads the clips; `kv_cache` pins the cached step to the
// full recompute; `contract` the end-to-end token streams from WAV on every speech clip (with VAD
// off and with the default trim); `silence` the gate; `language` forced languages; `errors` the
// caller-facing errors and cancellation; `inputs` the inputs rejected or emptied before
// decoding; `threads` bit-identity across worker-thread counts; `timing` prints the cost of one
// clip and of 30 s.

#[path = "support/mod.rs"]
mod support;

#[path = "transcribe/contract.rs"]
mod contract;
#[path = "transcribe/errors.rs"]
mod errors;
#[path = "transcribe/inputs.rs"]
mod inputs;
#[path = "transcribe/kv_cache.rs"]
mod kv_cache;
#[path = "transcribe/language.rs"]
mod language;
#[path = "transcribe/shared.rs"]
mod shared;
#[path = "transcribe/silence.rs"]
mod silence;
#[path = "transcribe/threads.rs"]
mod threads;
#[path = "transcribe/timing.rs"]
mod timing;
