//! whistle: a pure-Rust port of Whistle, Cactus Compute's 16.9 MB on-device speech-to-text model
//! (16 kHz mono audio in; text, language and word timings out; English, German, French, Spanish,
//! Italian, Dutch and Polish).
//!
//! [`Whistle::from_cact`] loads the model, and [`Whistle::transcribe`] (or
//! [`Whistle::transcribe_wav`]) turns audio into a [`Transcript`]: text, language and token ids.
//!
//! The library takes the model archive (`whistle.cact`) as bytes and never fetches anything, so
//! the same code runs natively and in the browser. The port follows upstream's Apache-2.0 JAX
//! reference (`needle-reference/`, pinned in `REFERENCE`); `docs/design/` describes the model
//! and `PORTING_PLAN.md` the open work.
//!
//! [`cact::Archive::parse`] reads the archive's geometry and tensor directory,
//! [`cact::Tensor::to_f32`] dequantizes any tensor, [`tokenizer::Tokenizer`] encodes and decodes
//! text, [`audio`] turns a WAV file or 16 kHz mono samples into the model's input features, and
//! [`vad`] finds the parts of the audio that contain speech. [`model::Weights::from_archive`]
//! binds the archive's tensors by role, and [`model::Encoder::forward`] turns features into the
//! speech embedding (the stem, eight encoder blocks in their mHC lane stack, the final norm).
//! The decoder's inputs are ported too: [`model::Weights::embed`], the Engram keys and values
//! ([`model::Engram::keys_values`], [`model::engram_add`]) and [`model::Weights::decoder_memory`].

// The README follows as the rest of the crate docs, so its usage example compiles as a doctest.
#![doc = include_str!("../../../README.md")]

pub mod audio;
pub mod cact;
pub mod error;
pub mod model;
pub mod quant;
pub mod tokenizer;
mod transcribe;
pub mod vad;

pub use error::Error;
pub use model::{set_threads, threads, MAX_THREADS};
pub use transcribe::{
    plan_windows, Cancel, Options, Transcript, VadMode, Whistle, Window, Word, DEFAULT_BEAMS,
    DEFAULT_KEYWORD_BONUS, FRAME_SECONDS, LANGUAGES, MAX_BEAMS, MAX_INPUT_SAMPLES, MAX_SAMPLES,
    OVERLAP_SAMPLES,
};
