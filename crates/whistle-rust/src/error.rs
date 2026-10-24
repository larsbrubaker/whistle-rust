// The library's error type: every fallible call in `whistle` returns `Result<_, Error>`, and each
// variant carries a message a caller can show (CLAUDE.md "Numbers": a malformed archive is an
// error, never a panic). Used by `cact` (archive parsing), `quant` (dequantization) and
// `tokenizer`, `Input` by the model stages for inconsistent caller input, and `Audio` (an
// undecodable WAV file) and `Cancelled` by `transcribe`.

use std::fmt;

use crate::audio::AudioError;

/// What went wrong, with a human-readable explanation. Non-exhaustive: later phases may add
/// variants, so a caller's `match` needs a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The model archive is malformed, truncated, or not a Whistle `.cact` file; or a tensor
    /// was asked for something its record does not support.
    Archive(String),
    /// The tokenizer attachment is malformed.
    Tokenizer(String),
    /// An input the caller passed is inconsistent (for example features whose data does not
    /// match their frame count).
    Input(String),
    /// A WAV file passed to `Whistle::transcribe_wav` could not be decoded.
    Audio(AudioError),
    /// The caller cancelled the call (`Cancel::cancel`) before it finished.
    Cancelled,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Archive(message) => write!(f, "invalid model archive: {message}"),
            Error::Tokenizer(message) => write!(f, "invalid tokenizer: {message}"),
            Error::Input(message) => write!(f, "invalid input: {message}"),
            Error::Audio(error) => error.fmt(f),
            Error::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Audio(error) => Some(error),
            _ => None,
        }
    }
}

impl From<AudioError> for Error {
    fn from(error: AudioError) -> Error {
        Error::Audio(error)
    }
}
