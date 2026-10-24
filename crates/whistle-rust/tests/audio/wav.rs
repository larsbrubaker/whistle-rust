// WAV decoding (`whistle::audio::read_wav`): the committed clips decode to their `samples`
// fixtures bit for bit, synthetic files in every supported encoding decode to the values
// upstream's `_read_wav` gives, and malformed files are errors, never panics.

use whistle::audio::{read_wav, AudioError};

use crate::support::{self, Fixture};

/// A WAV file: a fmt chunk (with the extensible layout when `extensible`), optional extra chunks
/// before the data, then the data chunk, each word aligned.
fn wav(
    format: u16,
    channels: u16,
    rate: u32,
    bits: u16,
    extra: &[(&[u8; 4], &[u8])],
    data: &[u8],
    extensible: bool,
) -> Vec<u8> {
    let align = channels * bits / 8;
    let mut fmt = Vec::new();
    fmt.extend((if extensible { 0xFFFEu16 } else { format }).to_le_bytes());
    fmt.extend(channels.to_le_bytes());
    fmt.extend(rate.to_le_bytes());
    fmt.extend(rate.wrapping_mul(u32::from(align)).to_le_bytes());
    fmt.extend(align.to_le_bytes());
    fmt.extend(bits.to_le_bytes());
    if extensible {
        fmt.extend(22u16.to_le_bytes());
        fmt.extend(bits.to_le_bytes());
        fmt.extend(0u32.to_le_bytes());
        fmt.extend(format.to_le_bytes());
        fmt.extend([0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
    }
    let mut body = b"WAVE".to_vec();
    let mut chunk = |id: &[u8; 4], bytes: &[u8]| {
        body.extend(id);
        body.extend((bytes.len() as u32).to_le_bytes());
        body.extend(bytes);
        if bytes.len() % 2 == 1 {
            body.push(0);
        }
    };
    chunk(b"fmt ", &fmt);
    for (id, bytes) in extra {
        chunk(id, bytes);
    }
    chunk(b"data", data);
    let mut file = b"RIFF".to_vec();
    file.extend((body.len() as u32).to_le_bytes());
    file.extend(body);
    file
}

#[test]
fn every_clip_decodes_to_its_samples_fixture_exactly() {
    for clip in support::clip_names() {
        let bytes = std::fs::read(support::repo_root().join(format!("test-data/clips/{clip}.wav")))
            .unwrap();
        let audio = read_wav(&bytes).unwrap_or_else(|e| panic!("{clip}: {e}"));
        let (_, expected) = Fixture::load(&clip).f32("samples");
        assert_eq!(audio.sample_rate, 16000, "{clip}");
        assert_eq!(audio.samples, expected, "{clip}");
    }
}

#[test]
fn integer_pcm_widths_scale_by_their_signed_range() {
    let pcm8 = wav(1, 1, 8000, 8, &[], &[0, 128, 255, 64], false);
    assert_eq!(
        read_wav(&pcm8).unwrap().samples,
        [-1.0, 0.0, 127.0 / 128.0, -0.5]
    );
    let pcm16: Vec<u8> = [i16::MIN, 0, 16384, i16::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        read_wav(&wav(1, 1, 16000, 16, &[], &pcm16, false))
            .unwrap()
            .samples,
        [-1.0, 0.0, 0.5, 32767.0 / 32768.0]
    );
    let pcm24 = [0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x40];
    assert_eq!(
        read_wav(&wav(1, 1, 16000, 24, &[], &pcm24, false))
            .unwrap()
            .samples,
        [-1.0, -1.0 / 8388608.0, 0.5]
    );
    let pcm32: Vec<u8> = [i32::MIN, -1, 1 << 30]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let decoded = read_wav(&wav(1, 1, 48000, 32, &[], &pcm32, false)).unwrap();
    assert_eq!(decoded.sample_rate, 48000);
    assert_eq!(decoded.samples, [-1.0, -1.0 / 2147483648.0, 0.5]);
}

#[test]
fn float_wavs_decode_as_is() {
    let f32s: Vec<u8> = [0.25f32, -1.5, 1e-3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        read_wav(&wav(3, 1, 16000, 32, &[], &f32s, false))
            .unwrap()
            .samples,
        [0.25, -1.5, 1e-3]
    );
    let f64s: Vec<u8> = [0.1f64, -0.75]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        read_wav(&wav(3, 1, 16000, 64, &[], &f64s, false))
            .unwrap()
            .samples,
        [0.1f64 as f32, -0.75]
    );
}

#[test]
fn channels_are_averaged_to_mono() {
    let stereo: Vec<u8> = [16384i16, 0, -32768, -16384]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        read_wav(&wav(1, 2, 16000, 16, &[], &stereo, false))
            .unwrap()
            .samples,
        [0.25, -0.75]
    );
    let three: Vec<u8> = [0.3f32, 0.6, 0.9]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let expected = (0.3f32 + 0.6 + 0.9) / 3.0;
    assert_eq!(
        read_wav(&wav(3, 3, 16000, 32, &[], &three, false))
            .unwrap()
            .samples,
        [expected]
    );
}

#[test]
fn extensible_format_uses_its_sub_format() {
    let pcm16: Vec<u8> = [8192i16, -8192]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        read_wav(&wav(1, 2, 44100, 16, &[], &pcm16, true))
            .unwrap()
            .samples,
        [0.0]
    );
    let f32s: Vec<u8> = [0.5f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let decoded = read_wav(&wav(3, 1, 22050, 32, &[], &f32s, true)).unwrap();
    assert_eq!((decoded.samples, decoded.sample_rate), (vec![0.5], 22050));
}

#[test]
fn odd_sized_chunks_before_data_are_skipped_word_aligned() {
    let pcm16: Vec<u8> = [16384i16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let extra: [(&[u8; 4], &[u8]); 2] = [(b"LIST", b"abc"), (b"junk", b"12345")];
    assert_eq!(
        read_wav(&wav(1, 1, 16000, 16, &extra, &pcm16, false))
            .unwrap()
            .samples,
        [0.5]
    );
}

#[test]
fn malformed_files_are_errors_not_panics() {
    let good = wav(1, 1, 16000, 16, &[], &[0, 0, 1, 0], false);
    assert_eq!(read_wav(b""), Err(AudioError::NotWav));
    assert_eq!(read_wav(b"RIFF\0\0\0\0WAVE"), Err(AudioError::NotWav));
    assert_eq!(read_wav(b"not a wav file at all"), Err(AudioError::NotWav));
    assert_eq!(read_wav(&good[..20]), Err(AudioError::NotWav));
    // Every truncation of a valid file is an error or a shorter decode, never a panic.
    for cut in 0..good.len() {
        let _ = read_wav(&good[..cut]);
    }
    assert_eq!(
        read_wav(&wav(1, 1, 16000, 16, &[], &[], false)),
        Err(AudioError::NoAudio)
    );
    assert!(matches!(
        read_wav(&wav(2, 1, 16000, 16, &[], &[0, 0], false)),
        Err(AudioError::UnsupportedEncoding {
            format: 2,
            bytes_per_sample: 2
        })
    ));
    assert!(matches!(
        read_wav(&wav(3, 1, 16000, 16, &[], &[0, 0], false)),
        Err(AudioError::UnsupportedEncoding { .. })
    ));
    assert!(matches!(
        read_wav(&wav(1, 0, 16000, 16, &[], &[0, 0], false)),
        Err(AudioError::Malformed(_))
    ));
    assert_eq!(
        read_wav(&wav(1, 1, 0, 16, &[], &[0, 0], false)),
        Err(AudioError::UnsupportedSampleRate(0))
    );
    // A chunk size that claims more than the file holds.
    let mut huge = good.clone();
    let at = huge.len() - 8;
    huge[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let _ = read_wav(&huge);
    assert_eq!(AudioError::NotWav.to_string(), "not a WAV file");
    assert_eq!(AudioError::NoAudio.to_string(), "no audio data");
}

/// The canonical 44-byte header with the RIFF and data sizes a streaming writer leaves when it
/// cannot seek back: 0xFFFFFFFF.
fn streaming(samples: &[i16]) -> Vec<u8> {
    let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut file = wav(1, 1, 16000, 16, &[], &data, false);
    file[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    file[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    file
}

#[test]
fn streaming_headers_with_unknown_sizes_decode_the_samples_present() {
    let file = streaming(&[16384, -16384, 8192]);
    assert_eq!(file.len(), 44 + 6);
    assert_eq!(read_wav(&file).unwrap().samples, [0.5, -0.5, 0.25]);
    // An odd byte count: the trailing partial sample is dropped.
    assert_eq!(read_wav(&file[..49]).unwrap().samples, [0.5, -0.5]);
}

#[test]
fn chunk_sizes_past_the_address_space_end_the_walk_without_panicking() {
    // A chunk before the data whose size, added to its offset, passes 2^32 (usize::MAX on
    // 32-bit targets): the walk stops there, so no data chunk is found.
    let pcm16: Vec<u8> = [16384i16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let extra: [(&[u8; 4], &[u8]); 1] = [(b"junk", b"abcd")];
    let mut file = wav(1, 1, 16000, 16, &extra, &pcm16, false);
    let junk = file.windows(4).position(|w| w == b"junk").unwrap();
    for size in [u32::MAX, u32::MAX - 1, u32::MAX - 8] {
        file[junk + 4..junk + 8].copy_from_slice(&size.to_le_bytes());
        assert_eq!(read_wav(&file), Err(AudioError::NotWav), "size {size:#x}");
    }
    // The same on the fmt chunk: it is cut to the bytes present and the data chunk is never
    // reached.
    let mut file = wav(1, 1, 16000, 16, &[], &pcm16, false);
    file[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(read_wav(&file), Err(AudioError::NotWav));
}

#[test]
fn sample_rates_outside_the_supported_range_are_rejected() {
    let pcm16 = [0u8, 0];
    for rate in [1u32, 999, 768_001, u32::MAX] {
        let error = read_wav(&wav(1, 1, rate, 16, &[], &pcm16, false)).unwrap_err();
        assert_eq!(error, AudioError::UnsupportedSampleRate(rate));
        assert!(error
            .to_string()
            .starts_with(&format!("unsupported sample rate {rate} Hz")));
    }
    for rate in [1000u32, 768_000] {
        assert_eq!(
            read_wav(&wav(1, 1, rate, 16, &[], &pcm16, false))
                .unwrap()
                .sample_rate,
            rate
        );
    }
}
