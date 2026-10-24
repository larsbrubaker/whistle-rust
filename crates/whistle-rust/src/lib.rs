//! whistle: a pure-Rust port of Whistle, Cactus Compute's 16.9 MB on-device speech-to-text model
//! (16 kHz mono audio in; text, language and word timings out; English, German, French, Spanish,
//! Italian, Dutch and Polish).
//!
//! The library takes the model archive (`whistle.cact`) as bytes and never fetches anything, so
//! the same code runs natively and in the browser. The port follows upstream's Apache-2.0 JAX
//! reference (`needle-reference/`, pinned in `REFERENCE`); `docs/design/` describes the model
//! and `PORTING_PLAN.md` the open work.
//!
//! [`cact::Archive::parse`] reads the archive's geometry and tensor directory,
//! [`cact::Tensor::to_f32`] dequantizes any tensor, [`tokenizer::Tokenizer`] encodes and decodes
//! text, and [`audio`] turns a WAV file or 16 kHz mono samples into the model's input features.

pub mod audio;
pub mod cact;
pub mod error;
pub mod quant;
pub mod tokenizer;

pub use error::Error;
