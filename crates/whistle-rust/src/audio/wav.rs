//! WAV decoding for the front end: a RIFF/WAVE chunk walk and integer/float sample decoding,
//! downmixed to mono and scaled as upstream's `agent.whistle._read_wav` does (signed integer /
//! 2^(bits-1), unsigned 8-bit offset by 128, float as is). Resampling to 16 kHz is the caller's
//! next step (`resample`), since the file may be at any rate.

use super::{AudioError, SUPPORTED_RATES};

/// Mono samples and the rate they were recorded at.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// The first `fmt ` and `data` chunks. Chunks are word aligned (an odd-sized chunk is followed by
/// a pad byte); a chunk that runs past the end of the file is cut at the end, as a truncated
/// download would be, rather than rejected.
fn chunks(bytes: &[u8]) -> Result<(&[u8], &[u8]), AudioError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::NotWav);
    }
    let (mut fmt, mut data) = (None, None);
    // Positions are u64 so the walk is the same on 32- and 64-bit targets: a 0xFFFFFFFF chunk
    // size (what streaming writers put in a header they cannot seek back to) cannot overflow it.
    let len = bytes.len() as u64;
    let mut at = 12u64;
    while at + 8 <= len {
        // `at + 8 <= len`, so these casts are in range of usize.
        let here = at as usize;
        let id = &bytes[here..here + 4];
        let size = u64::from(u32_at(bytes, here + 4).ok_or(AudioError::NotWav)?);
        let start = at + 8;
        // A chunk that runs past the end is cut to what is present.
        let end = (start + size).min(len);
        let body = &bytes[start as usize..end as usize];
        if id == b"fmt " && fmt.is_none() {
            fmt = Some(body);
        } else if id == b"data" && data.is_none() {
            data = Some(body);
        }
        at = start + size + (size & 1);
    }
    match (fmt, data) {
        (Some(fmt), Some(data)) => Ok((fmt, data)),
        _ => Err(AudioError::NotWav),
    }
}

/// Decodes a WAV file (integer PCM 8/16/24/32-bit or IEEE float 32/64-bit, including
/// WAVE_FORMAT_EXTENSIBLE, any channel count) to mono at the file's own rate.
pub fn read_wav(bytes: &[u8]) -> Result<Audio, AudioError> {
    let (fmt, data) = chunks(bytes)?;
    let short = AudioError::Malformed("the fmt chunk is too short");
    let mut format = u16_at(fmt, 0).ok_or(short.clone())?;
    let channels = u16_at(fmt, 2).ok_or(short.clone())?;
    let sample_rate = u32_at(fmt, 4).ok_or(short.clone())?;
    let align = u16_at(fmt, 12).ok_or(short.clone())?;
    if format == FORMAT_EXTENSIBLE {
        // The sub-format GUID starts at byte 24; its first two bytes are the format tag.
        format = u16_at(fmt, 24).ok_or(AudioError::Malformed(
            "the extensible fmt chunk is too short",
        ))?;
    }
    if channels == 0 {
        return Err(AudioError::Malformed("zero channels"));
    }
    if !SUPPORTED_RATES.contains(&sample_rate) {
        return Err(AudioError::UnsupportedSampleRate(sample_rate));
    }
    let width = align / channels;
    let decode: fn(&[u8]) -> f32 = match (format, width) {
        (FORMAT_PCM, 1) => |b| f32::from(b[0]),
        (FORMAT_PCM, 2) => |b| f32::from(i16::from_le_bytes([b[0], b[1]])),
        // Sign-extend the 24-bit value through the top of an i32.
        (FORMAT_PCM, 3) => |b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32,
        (FORMAT_PCM, 4) => |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        (FORMAT_FLOAT, 4) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (FORMAT_FLOAT, 8) => {
            |b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }
        _ => {
            return Err(AudioError::UnsupportedEncoding {
                format,
                bytes_per_sample: width,
            })
        }
    };
    // The same f32 arithmetic as `_read_wav`: each value widened to f32, the channels summed in
    // f32 and divided by the count, then the offset removed and the scale divided out.
    let (offset, scale) = match (format, width) {
        (FORMAT_FLOAT, _) => (0.0f32, 1.0f32),
        (_, 1) => (128.0, 128.0),
        (_, w) => (0.0, (1u64 << (8 * u32::from(w) - 1)) as f32),
    };
    let width = usize::from(width);
    let channels_n = usize::from(channels);
    let frame = width * channels_n;
    // A trailing partial frame (a truncated file) is dropped.
    let samples: Vec<f32> = data
        .chunks_exact(frame)
        .map(|f| {
            let sum = f.chunks_exact(width).fold(0.0f32, |acc, s| acc + decode(s));
            (sum / f32::from(channels) - offset) / scale
        })
        .collect();
    if samples.is_empty() {
        return Err(AudioError::NoAudio);
    }
    Ok(Audio {
        samples,
        sample_rate,
    })
}
