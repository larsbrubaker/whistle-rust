// Test area: voice-activity detection (`whistle::vad`). `signals` builds the synthetic
// non-speech inputs and loads the speech clips with the engine's word span; `non_speech` and
// `speech` pin the two sides of the decision, `properties` the output's shape, `config` the
// handling of extreme settings, `known_limits` the sounds still misread as speech, and
// `margins` how far each default threshold sits from flipping a case.

#[path = "support/mod.rs"]
mod support;

#[path = "vad/config.rs"]
mod config;
#[path = "vad/known_limits.rs"]
mod known_limits;
#[path = "vad/margins.rs"]
mod margins;
#[path = "vad/non_speech.rs"]
mod non_speech;
#[path = "vad/properties.rs"]
mod properties;
#[path = "vad/signals.rs"]
mod signals;
#[path = "vad/speech.rs"]
mod speech;
