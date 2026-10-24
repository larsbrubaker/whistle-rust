// The library's error type: every fallible call in `whistle` returns `Result<_, Error>`, and each
// variant carries a message a caller can show (CLAUDE.md "Numbers": a malformed archive is an
// error, never a panic). Used by `cact` (archive parsing), `quant` (dequantization) and
// `tokenizer`, and `Input` by the model stages for inconsistent caller input.

use std::fmt;

/// What went wrong, with a human-readable explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The model archive is malformed, truncated, or not a Whistle `.cact` file; or a tensor
    /// was asked for something its record does not support.
    Archive(String),
    /// The tokenizer attachment is malformed.
    Tokenizer(String),
    /// An input the caller passed is inconsistent (for example features whose data does not
    /// match their frame count).
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Archive(message) => write!(f, "invalid model archive: {message}"),
            Error::Tokenizer(message) => write!(f, "invalid tokenizer: {message}"),
            Error::Input(message) => write!(f, "invalid input: {message}"),
        }
    }
}

impl std::error::Error for Error {}
