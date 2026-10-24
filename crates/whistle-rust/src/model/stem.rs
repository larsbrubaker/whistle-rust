//! The convolutional stem's forward pass (`whistle.Stem`) and the encoder's input step
//! (`Whistle.encode`: pad to an even frame count, broadcast to the lanes). Normalized log-mel
//! features (`audio::Features`, 10 ms frames) become one `d_model` vector per 80 ms; the
//! weights are the [`Stem`] bound in `weights`, the arithmetic is `ops`.

use crate::audio::Features;
use crate::error::Error;

use super::ops::{a8, silu};
use super::weights::{Matrix, Stem};

/// The stem's output: `frames × width` row-major, with `valid_frames` real frames (later ones
/// are padding computed from zeroed input).
#[derive(Debug, Clone, PartialEq)]
pub struct StemOutput {
    pub data: Vec<f32>,
    pub frames: usize,
    pub width: usize,
    pub valid_frames: usize,
}

/// A lane stack's input: `frames × lanes × width` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Lanes {
    pub data: Vec<f32>,
    pub frames: usize,
    pub lanes: usize,
    pub width: usize,
    pub valid_frames: usize,
}

/// A `[time, mel, channel]` activation.
struct Plane {
    data: Vec<f32>,
    time: usize,
    mel: usize,
}

/// The output size of a 3-wide, stride-2, padding-1 convolution: `ceil(n / 2)`.
fn halved(n: usize) -> usize {
    n.div_ceil(2)
}

/// A 3×3, stride 2, padding 1 convolution over `[time, mel]` writing `channels` outputs,
/// cross-correlation as `lax.conv_general_dilated` computes it. `input(t, m, c)` reads the
/// (in-bounds) input for output channel `c`; `w` is `[9, channels]`, tap `3·i + j` for time
/// offset `i` and mel offset `j`. Output frames at or past `valid` are zeroed.
fn conv3x3(
    time: usize,
    mel: usize,
    channels: usize,
    w: &Matrix,
    valid: usize,
    input: impl Fn(usize, usize, usize) -> f32,
) -> Plane {
    let (t_out, m_out) = (halved(time), halved(mel));
    let mut data = vec![0.0f32; t_out * m_out * channels];
    for t in 0..valid.min(t_out) {
        for m in 0..m_out {
            let cell = &mut data[(t * m_out + m) * channels..(t * m_out + m + 1) * channels];
            for (c, out) in cell.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..3 {
                    // Input row 2t + i - 1; row -1 and rows past the end are the zero padding.
                    let Some(ti) = (2 * t + i).checked_sub(1).filter(|&ti| ti < time) else {
                        continue;
                    };
                    for j in 0..3 {
                        let Some(mj) = (2 * m + j).checked_sub(1).filter(|&mj| mj < mel) else {
                            continue;
                        };
                        acc += input(ti, mj, c) * w.data[(3 * i + j) * channels + c];
                    }
                }
                *out = acc;
            }
        }
    }
    Plane {
        data,
        time: t_out,
        mel: m_out,
    }
}

impl Stem {
    /// The stem over `features` (`frames × mels` row-major). Features whose data is not
    /// `frames · mels` values, or that claim more valid frames than frames, are
    /// [`Error::Input`].
    pub fn forward(&self, features: &Features) -> Result<StemOutput, Error> {
        let mels = self.mels;
        let c = self.w.cols;
        let frames = features.frames;
        if frames.checked_mul(mels) != Some(features.data.len()) {
            return Err(Error::Input(format!(
                "features hold {} values, not {frames} frames of {mels} mel bins",
                features.data.len()
            )));
        }
        if features.valid_frames > frames {
            return Err(Error::Input(format!(
                "features claim {} valid frames of {frames}",
                features.valid_frames
            )));
        }
        let x = &features.data;
        // Each stage halves the valid length, rounding up; frames past it are zeroed.
        let mut valid = halved(features.valid_frames);
        let mut plane = conv3x3(frames, mels, c, &self.w, valid, |t, m, _| x[t * mels + m]);
        for stage in &self.stages {
            let act: Vec<f32> = plane.data.iter().map(|&v| silu(v)).collect();
            valid = halved(valid);
            let mel = plane.mel;
            let down = conv3x3(plane.time, mel, c, &stage.dw, valid, |t, m, ch| {
                act[(t * mel + m) * c + ch]
            });
            plane = Plane {
                data: stage.pw.project(&down.data),
                time: down.time,
                mel: down.mel,
            };
        }
        // SiLU, then each frame flattened channel-major (`swapaxes(2, 3)`: index c · mel + f).
        let width = c * plane.mel;
        let mut flat = vec![0.0f32; plane.time * width];
        for t in 0..plane.time {
            for f in 0..plane.mel {
                for ch in 0..c {
                    flat[t * width + ch * plane.mel + f] =
                        silu(plane.data[(t * plane.mel + f) * c + ch]);
                }
            }
        }
        a8(&mut flat, width);
        Ok(StemOutput {
            data: self.out.project(&flat),
            frames: plane.time,
            width: self.out.rows,
            valid_frames: valid,
        })
    }
}

impl StemOutput {
    /// The encoder's input (`Whistle.encode`): frames zero-padded to an even count, each frame
    /// broadcast to all `lanes`.
    pub fn to_lanes(&self, lanes: usize) -> Lanes {
        let frames = self.frames + self.frames % 2;
        let mut data = Vec::with_capacity(frames * lanes * self.width);
        for t in 0..frames {
            let row = self.data.get(t * self.width..(t + 1) * self.width);
            for _ in 0..lanes {
                match row {
                    Some(row) => data.extend_from_slice(row),
                    None => data.resize(data.len() + self.width, 0.0),
                }
            }
        }
        Lanes {
            data,
            frames,
            lanes,
            width: self.width,
            valid_frames: self.valid_frames,
        }
    }
}
