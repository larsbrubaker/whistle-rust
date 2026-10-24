// Test area: the audio front end (`whistle::audio`): WAV decoding against the clips' `samples`
// fixtures and synthetic files, the resampler's band-limiting properties, and the mel features
// against the oracle's `features` fixtures and `mel.npy`.

#[path = "support/mod.rs"]
mod support;

#[path = "audio/features.rs"]
mod features;
#[path = "audio/resample.rs"]
mod resample;
#[path = "audio/wav.rs"]
mod wav;
