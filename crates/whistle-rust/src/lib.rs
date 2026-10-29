//! whistle: a pure-Rust port of Whistle, Cactus Compute's 16.9 MB on-device speech-to-text model
//! (16 kHz mono audio in; text, language and word timings out; English, German, French, Spanish,
//! Italian, Dutch and Polish).
//!
//! [`Whistle::from_cact`] loads the model, and [`Whistle::transcribe`] (or
//! [`Whistle::transcribe_wav`]) turns audio into a [`Transcript`]: text, language and token ids.
//! [`Whistle::stream`] transcribes live audio as it arrives ([`Stream`]).
//!
//! The library takes the model archive (`whistle.cact`) as bytes and never fetches anything, so
//! the same code runs natively and in the browser. `PORTING_PLAN.md` holds the open work.
//!
//! # How the pieces fit
//!
//! A call runs, in order: the input checks and resampling ([`audio`]), the voice-activity gate
//! ([`vad`]), the windows (one up to 30 s, more beyond), and per window the log-mel features
//! ([`audio::features`]), the encoder ([`model::Encoder`]: the convolutional stem, eight blocks
//! in an mHC lane stack, the final norm), the decoder memory, and the decode over the decoder's
//! KV cache ([`model::DecoderState`]): a language tag at step 0, then text tokens by beam search
//! with optional keyword biasing and a repetition guard, until EOS. Word times, when asked for,
//! come from one more decoder pass whose cross-attention is aligned to the audio frames. The
//! transcription module's header (`src/transcribe/mod.rs`) walks through the call; each model
//! module's header gives its stage's formula.
//!
//! The model itself (archive format, front end, every layer, the greedy decode) is a port of
//! upstream's Apache-2.0 JAX reference (`needle/model/whistle.py`, `architecture.py`,
//! `quantize.py`, `export.py` in `needle-reference/`, pinned in `REFERENCE`), run on the
//! Apache-2.0 `whistle.cact` weights, and is tested stage by stage against fixtures that
//! reference writes (`oracle/dump.py`). Everything the reference does not define is designed
//! here from the Needle engine's documented behaviour and measured against its transcripts as
//! black-box data: beam search, keyword biasing, word timestamps aligned from decoder
//! cross-attention, windowing past 30 s, streaming (committing the words two passes agree on),
//! the silence gate (silence and steady noise give an empty transcript) and the KV cache.
//!
//! [`cact::Archive::parse`] reads the archive's geometry and tensor directory,
//! [`cact::Tensor::to_f32`] dequantizes any tensor, [`tokenizer::Tokenizer`] encodes and decodes
//! text, [`audio`] turns a WAV file or 16 kHz mono samples into the model's input features, and
//! [`vad`] finds the parts of the audio that contain speech. [`model::Weights::from_archive`]
//! binds the archive's tensors by role, and [`model::Encoder::forward`] turns features into the
//! speech embedding (the stem, eight encoder blocks in their mHC lane stack, the final norm).
//! The decoder's inputs are ported too: [`model::Weights::embed`], the Engram keys and values
//! ([`model::Engram::keys_values`], [`model::engram_add`]) and [`model::Weights::decoder_memory`].

// The crate README (crates.io's page for the crate) follows as the rest of the crate docs, so its
// usage examples compile as doctests. It sits beside Cargo.toml so it is inside the package.
#![doc = include_str!("../README.md")]

pub mod audio;
pub mod cact;
pub mod error;
mod math;
pub mod model;
pub mod model_file;
pub mod quant;
pub mod tokenizer;
mod transcribe;
pub mod vad;

pub use error::Error;
pub use model::{set_threads, threads, MAX_THREADS};
pub use transcribe::{
    plan_windows, Cancel, Options, Stream, StreamUpdate, Transcript, VadMode, Whistle, Window,
    Word, DEFAULT_BEAMS, DEFAULT_KEYWORD_BONUS, DEFAULT_STREAM_STEP, FRAME_SECONDS, LANGUAGES,
    MAX_BEAMS, MAX_INPUT_SAMPLES, MAX_SAMPLES, OVERLAP_SAMPLES,
};
