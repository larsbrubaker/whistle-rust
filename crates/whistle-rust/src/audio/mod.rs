//! The audio front end: WAV decoding (`wav`), resampling to 16 kHz (`resample` for a whole
//! clip, `stream_resample` for audio that arrives in pieces) and Whistle's input features
//! (`mel`, built on the in-crate real FFT in `fft`). Everything the encoder's stem consumes
//! starts here; the model sees only the normalized log-mel `Features`.

pub(crate) mod fft;
mod mel;
mod resample;
mod stream_resample;
mod wav;

use std::fmt;

pub use mel::{features, mel_matrix, Features, FFT_BINS, HOP, MELS};
pub use resample::resample;
pub use stream_resample::StreamResampler;
pub use wav::{read_wav, Audio};

/// The sample rate Whistle's features are defined at (`WhistleConfig.rate`).
pub const SAMPLE_RATE: u32 = 16000;

/// The sample rates `read_wav` accepts and `resample` converts between, in Hz. Real recordings
/// sit well inside this range; outside it a header is more likely corrupt or hostile, and a
/// tiny rate would make resampling to 16 kHz allocate thousands of samples per input sample.
pub const SUPPORTED_RATES: std::ops::RangeInclusive<u32> = 1000..=768_000;

/// Why audio could not be decoded. Every message is meant to be shown to a caller as is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The bytes are not a RIFF/WAVE file with both a `fmt ` and a `data` chunk.
    NotWav,
    /// A WAV file whose header fields are inconsistent (the reason says which).
    Malformed(&'static str),
    /// A WAV encoding other than integer PCM 8/16/24/32-bit or IEEE float 32/64-bit.
    UnsupportedEncoding {
        /// The format tag (after resolving WAVE_FORMAT_EXTENSIBLE to its sub-format).
        format: u16,
        /// Bytes per sample (block align / channels).
        bytes_per_sample: u16,
    },
    /// A sample rate outside `SUPPORTED_RATES`.
    UnsupportedSampleRate(u32),
    /// The file decodes to zero samples.
    NoAudio,
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::NotWav => write!(f, "not a WAV file"),
            AudioError::Malformed(why) => write!(f, "malformed WAV file: {why}"),
            AudioError::UnsupportedEncoding {
                format,
                bytes_per_sample,
            } => write!(
                f,
                "unsupported WAV encoding (format tag {format}, {} bits per sample); expected \
                 integer PCM (8, 16, 24 or 32 bits) or IEEE float (32 or 64 bits)",
                u32::from(*bytes_per_sample) * 8
            ),
            AudioError::UnsupportedSampleRate(rate) => write!(
                f,
                "unsupported sample rate {rate} Hz (expected {} to {} Hz)",
                SUPPORTED_RATES.start(),
                SUPPORTED_RATES.end()
            ),
            AudioError::NoAudio => write!(f, "no audio data"),
        }
    }
}

impl std::error::Error for AudioError {}
