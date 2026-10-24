//! The model file this crate version was tested against: where to download `whistle.cact`, its
//! SHA-256 and its size. The weights are not in the crate and the library never fetches
//! anything; these constants let a caller download, verify and cache the right file.
//!
//! A native app bundles the file or downloads it once into its data directory; a web host
//! downloads it (from [`MODEL_URL`] or its own copy) and caches it. Either way, check the
//! length against [`MODEL_BYTES`] and the digest against [`MODEL_SHA256`] before passing the
//! bytes to [`crate::Whistle::from_cact`]. The file is Copyright Cactus Compute, Inc., under the
//! Apache License 2.0: an app that ships it includes that license and `THIRD_PARTY_NOTICES.md`.
//!
//! The same pin is `REFERENCE` "whistle-weights" and `scripts/weights.sha256` in the repository
//! (checked by `tests/archive/model_file.rs`).

/// `whistle.cact` on Hugging Face (`Cactus-Compute/whistle`), at the revision this crate version
/// was tested against, so the URL keeps returning the same bytes when upstream publishes newer
/// weights.
pub const MODEL_URL: &str = "https://huggingface.co/Cactus-Compute/whistle/resolve/741118687e1f008f06d1351a402a731274789790/whistle.cact";

/// The SHA-256 of the file at [`MODEL_URL`], as 64 lowercase hex digits.
pub const MODEL_SHA256: &str = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb";

/// The size of the file at [`MODEL_URL`] in bytes (16.9 MB).
pub const MODEL_BYTES: u64 = 16_919_407;
