//! Transcription, the library's front door: [`Whistle`] owns the bound weights and the tokenizer
//! of one archive and turns 16 kHz mono audio (or a WAV file, through `audio`) into a
//! [`Transcript`]. It runs the voice-activity detector (`vad`) as [`Options::vad`] asks, the
//! front end and the encoder once, then the decode over the decoder's KV cache
//! (`model::DecoderState`): step 0 picks a language tag (or feeds the forced one), then
//! `search` picks text tokens, greedily as `whistle_finetune._score_quantised` defines with one
//! beam or by beam search with more, optionally biased towards [`Options::keywords`]
//! (`keywords`), until EOS or `max_seq_len` tokens. A [`Cancel`] handle is checked between the
//! stages, between encoder layers and between decoder steps. `docs/design/transcribe.md`
//! describes the call and `docs/design/decoding.md` the search.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::audio::{features, read_wav, resample, SAMPLE_RATE};
use crate::cact::Archive;
use crate::error::Error;
use crate::model::{DecoderState, Encoder, Weights};
use crate::tokenizer::Tokenizer;
use crate::vad;

mod keywords;
mod search;

use keywords::KeywordTrie;
pub use keywords::DEFAULT_KEYWORD_BONUS;
use search::{Ranking, Search};

/// The most beams [`Options::beams`] accepts: each beam is a decoder state stepped every
/// position, so the cost grows linearly (`docs/design/decoding.md`).
pub const MAX_BEAMS: usize = 16;

/// The default [`Options::beams`]: a 3-beam search (measured in `docs/design/decoding.md`).
pub const DEFAULT_BEAMS: usize = 3;

/// The language codes Whistle transcribes, in tag order.
pub const LANGUAGES: [&str; 7] = ["en", "de", "fr", "es", "it", "nl", "pl"];

/// The longest audio one call decodes, in 16 kHz samples (30 s, the encoder's training window),
/// after trimming.
/// Longer audio needs windowing, which is not ported yet.
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

/// The most audio [`VadMode::GateAndTrim`] accepts before trimming, in 16 kHz samples
/// (10 minutes): trimming can bring a long recording with a short utterance under
/// [`MAX_SAMPLES`], but the detector and the resampler cost time in proportion to the input,
/// so a sanity ceiling keeps a mistaken hour-long input from running for seconds before it
/// fails.
pub const MAX_INPUT_SAMPLES: usize = 600 * SAMPLE_RATE as usize;

/// Where a transcription is when it checks for cancellation (the tests' hook sees each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// After the input checks, before any work.
    Start,
    /// After resampling to 16 kHz.
    Resampled,
    /// After the voice-activity detector.
    Vad,
    /// After the front end.
    Features,
    /// Before each encoder layer.
    EncoderLayer,
    /// Before the decoder step at this position.
    DecoderStep(usize),
    /// After the decode.
    Done,
}

fn too_long(samples: u64, limit: usize) -> Error {
    Error::Input(format!(
        "{:.1} s of audio; one call transcribes at most {} s ({limit} samples at 16 kHz)",
        samples as f64 / f64::from(SAMPLE_RATE),
        limit / SAMPLE_RATE as usize
    ))
}

/// How the voice-activity detector gates and trims the audio before decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadMode {
    /// Decode the whole clip, whatever it holds (silence makes the decoder hallucinate).
    Off,
    /// No speech found gives an empty transcript; otherwise decode the whole clip.
    Gate,
    /// No speech found gives an empty transcript; otherwise decode only the samples from the
    /// first speech segment's start to the last one's end.
    #[default]
    GateAndTrim,
}

/// A cloneable cancellation handle: [`Cancel::cancel`] from any thread makes a transcription
/// using it (or any clone of it) return [`Error::Cancelled`] at its next check.
#[derive(Debug, Clone, Default)]
pub struct Cancel {
    flag: Arc<AtomicBool>,
}

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Asks every transcription holding this handle to stop.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// What one transcription does beyond the defaults.
#[derive(Debug, Clone)]
pub struct Options {
    /// Force the language (one of [`LANGUAGES`]); `None` lets the model detect it.
    pub language: Option<String>,
    /// How the voice-activity detector gates the audio; [`VadMode::GateAndTrim`] by default.
    pub vad: VadMode,
    /// Checked between the stages (resampling, detection, features), before each encoder layer
    /// and before every decoder step (each beam's step at each position).
    pub cancel: Option<Cancel>,
    /// Hypotheses the decode keeps, 1 to [`MAX_BEAMS`]. The default, [`DEFAULT_BEAMS`] (3), is
    /// the smallest count that reproduced every engine transcript on the test clips, with the
    /// best word error rate against the true texts, for about +195 ms on a 3.3 s clip
    /// (`docs/design/decoding.md`). `beams: 1` is the greedy decode the reference defines.
    /// 0 or more than [`MAX_BEAMS`] is [`Error::Input`].
    pub beams: usize,
    /// Words and phrases to favour during the search (names, places, product words, jargon);
    /// blank entries are ignored. Works with any number of beams.
    pub keywords: Vec<String>,
    /// The log-probability bonus per keyword token ([`DEFAULT_KEYWORD_BONUS`] by default);
    /// it must be finite and not negative.
    pub keyword_bonus: f32,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            language: None,
            vad: VadMode::default(),
            cancel: None,
            beams: DEFAULT_BEAMS,
            keywords: Vec::new(),
            keyword_bonus: DEFAULT_KEYWORD_BONUS,
        }
    }
}

/// The result of a transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// The decoded text; empty when no speech was found or nothing was said.
    pub text: String,
    /// The language code the text is in (detected or forced); `None` when the transcript is
    /// empty.
    pub language: Option<String>,
    /// The decoded text token ids, without BOS, the language tag and EOS.
    pub tokens: Vec<u32>,
}

impl Transcript {
    fn empty() -> Transcript {
        Transcript {
            text: String::new(),
            language: None,
            tokens: Vec::new(),
        }
    }
}

/// A loaded Whistle model: its weights and tokenizer, owned, so it can be shared across threads
/// and kept for the life of the program.
pub struct Whistle {
    weights: Weights,
    tokenizer: Tokenizer,
    /// Each supported language's tag id, in [`LANGUAGES`] order.
    tags: Vec<u32>,
    /// The first language tag id: ids below it are text, ids from it on are tags.
    first_tag: usize,
}

impl Whistle {
    /// Parses a `whistle.cact` archive and binds its weights and tokenizer. A malformed archive,
    /// or one whose tokenizer lacks a supported language's tag, is [`Error::Archive`] (or
    /// [`Error::Tokenizer`]).
    pub fn from_cact(bytes: &[u8]) -> Result<Whistle, Error> {
        let archive = Archive::parse(bytes)?;
        let weights = Weights::from_archive(&archive)?;
        let tokenizer = Tokenizer::from_archive(&archive)?;
        let tags = LANGUAGES
            .iter()
            .map(|code| {
                tokenizer.language_tag(code).ok_or_else(|| {
                    Error::Archive(format!("the tokenizer has no language tag for {code}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first_tag = tags.iter().copied().min().unwrap_or(0) as usize;
        let vocab = weights.embedding.rows();
        // The reference masks `arange(vocab) >= text` at step 0, so the tags must be exactly the
        // ids from the first tag to the end of the vocabulary.
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        let contiguous = sorted
            .iter()
            .enumerate()
            .all(|(i, &t)| t as usize == first_tag + i);
        if !contiguous || first_tag + tags.len() != vocab {
            return Err(Error::Archive(format!(
                "language tags {tags:?} are not the last {} ids of the {vocab}-token vocabulary",
                tags.len()
            )));
        }
        let ids = [tokenizer.bos_id(), tokenizer.eos_id()];
        if tags.iter().chain(&ids).any(|&id| id as usize >= vocab)
            || ids.iter().any(|&id| id as usize >= first_tag)
        {
            return Err(Error::Archive(format!(
                "language tags {tags:?}, BOS and EOS {ids:?} do not fit a {vocab}-token vocabulary \
                 with text before the tags"
            )));
        }
        Ok(Whistle {
            weights,
            tokenizer,
            tags,
            first_tag,
        })
    }

    /// The bound weights, for callers that run the model stages themselves.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Decodes a WAV file (any rate the reader supports, downmixed to mono), resamples it to
    /// 16 kHz if needed, then transcribes as [`Whistle::transcribe`] does. A file that cannot
    /// be decoded is [`Error::Audio`]. Audio too long to transcribe is rejected before it is
    /// resampled.
    pub fn transcribe_wav(&self, wav: &[u8], options: &Options) -> Result<Transcript, Error> {
        let audio = read_wav(wav)?;
        self.run(
            Cow::Owned(audio.samples),
            audio.sample_rate,
            options,
            &mut |_| {},
        )
    }

    /// Transcribes 16 kHz mono `samples`. A non-finite sample, an unsupported
    /// [`Options::language`], or audio longer than [`MAX_SAMPLES`] (after trimming; with
    /// [`VadMode::GateAndTrim`], any input over [`MAX_INPUT_SAMPLES`] before it) is
    /// [`Error::Input`]; a fired [`Options::cancel`] is [`Error::Cancelled`]. Empty audio,
    /// audio shorter than one 10 ms feature hop, or no speech with the gate on is an empty
    /// transcript.
    pub fn transcribe(&self, samples: &[f32], options: &Options) -> Result<Transcript, Error> {
        self.run(Cow::Borrowed(samples), SAMPLE_RATE, options, &mut |_| {})
    }

    /// The whole call on mono `samples` at `rate`. `hook` sees every [`Stage`] just before the
    /// cancellation check that follows it (the tests' handle for cancelling deterministically
    /// at any point).
    pub(crate) fn run(
        &self,
        samples: Cow<'_, [f32]>,
        rate: u32,
        options: &Options,
        hook: &mut dyn FnMut(Stage),
    ) -> Result<Transcript, Error> {
        self.run_ranked(samples, rate, options, hook, Ranking::Mean)
    }

    /// [`Whistle::run`] with the beam search's final ranking chosen (the tests measure the
    /// alternatives against the default).
    pub(crate) fn run_ranked(
        &self,
        samples: Cow<'_, [f32]>,
        rate: u32,
        options: &Options,
        hook: &mut dyn FnMut(Stage),
        ranking: Ranking,
    ) -> Result<Transcript, Error> {
        let forced = match options.language.as_deref() {
            None => None,
            Some(code) => Some(self.language_index(code)?),
        };
        if options.beams == 0 || options.beams > MAX_BEAMS {
            return Err(Error::Input(format!(
                "{} beams; the decode takes 1 to {MAX_BEAMS}",
                options.beams
            )));
        }
        if !options.keyword_bonus.is_finite() || options.keyword_bonus < 0.0 {
            return Err(Error::Input(format!(
                "keyword bonus {}; it must be finite and not negative",
                options.keyword_bonus
            )));
        }
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::Input(format!(
                "sample {i} is {}; audio must be finite",
                samples[i]
            )));
        }
        // The length at 16 kHz, before paying for the resampler or the detector. A rate of 0
        // never reaches here from `read_wav`; it counts as no audio.
        let at_16k = (samples.len() as u64 * u64::from(SAMPLE_RATE))
            .checked_div(u64::from(rate))
            .unwrap_or(0);
        let ceiling = match options.vad {
            VadMode::GateAndTrim => MAX_INPUT_SAMPLES,
            VadMode::Off | VadMode::Gate => MAX_SAMPLES,
        };
        if at_16k > ceiling as u64 {
            return Err(too_long(at_16k, ceiling));
        }
        let cancelled = || options.cancel.as_ref().is_some_and(Cancel::is_cancelled);
        let mut check = |stage: Stage| {
            hook(stage);
            if cancelled() {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        check(Stage::Start)?;
        let samples = if rate == SAMPLE_RATE {
            samples
        } else {
            // The original-rate buffer is dropped here, once resampled.
            let resampled = resample(&samples, rate, SAMPLE_RATE);
            drop(samples);
            check(Stage::Resampled)?;
            Cow::Owned(resampled)
        };
        let samples: &[f32] = match options.vad {
            VadMode::Off => &samples,
            mode => {
                let segments = vad::speech_segments(&samples);
                check(Stage::Vad)?;
                match (segments.first(), segments.last()) {
                    (Some(first), Some(last)) if mode == VadMode::GateAndTrim => {
                        &samples[first.start..last.end]
                    }
                    (Some(_), Some(_)) => &samples,
                    _ => return Ok(Transcript::empty()),
                }
            }
        };
        if samples.len() > MAX_SAMPLES {
            return Err(too_long(samples.len() as u64, MAX_SAMPLES));
        }
        let feats = features(samples);
        // Under one hop there is no valid frame: the decoder would attend to nothing.
        if feats.valid_frames == 0 {
            return Ok(Transcript::empty());
        }
        check(Stage::Features)?;

        let w = &self.weights;
        let enc =
            Encoder::new(w).forward_with(&feats, &mut || check(Stage::EncoderLayer).is_err())?;
        let memory = w.decoder_memory(&enc.data, enc.frames, enc.valid_frames)?;
        let max = w.geometry.max_seq_len as usize;
        let mut state = DecoderState::new(w, &memory, max)?;
        let (bos, eos) = (self.tokenizer.bos_id(), self.tokenizer.eos_id());
        // Step 0 picks the language every hypothesis shares.
        check(Stage::DecoderStep(0))?;
        let scores = state.step(bos)?;
        let index = match forced {
            Some(i) => i,
            None => self.detect_language(&scores),
        };
        let language = Some(index);
        let trie = KeywordTrie::new(
            &self.tokenizer,
            &options.keywords,
            self.first_tag,
            options.keyword_bonus,
        );
        let search = Search {
            beams: options.beams,
            eos,
            text_ids: self.first_tag,
            max_len: max,
            trie: trie.as_ref(),
            ranking,
        };
        let text_tokens = search::decode(&search, state, self.tags[index], &mut |pos| {
            check(Stage::DecoderStep(pos))
        })?;
        check(Stage::Done)?;
        if text_tokens.is_empty() {
            return Ok(Transcript::empty());
        }
        Ok(Transcript {
            text: self.tokenizer.decode(&text_tokens),
            language: language.map(|i| LANGUAGES[i].to_string()),
            tokens: text_tokens,
        })
    }

    /// The index in [`LANGUAGES`] of `code`, or [`Error::Input`] listing the supported codes.
    fn language_index(&self, code: &str) -> Result<usize, Error> {
        LANGUAGES.iter().position(|&c| c == code).ok_or_else(|| {
            Error::Input(format!(
                "unsupported language {code:?}; Whistle supports {}",
                LANGUAGES.join(", ")
            ))
        })
    }

    /// Step 0's pick: the best-scoring language tag (the reference masks every other id to
    /// −∞), as an index into [`LANGUAGES`].
    fn detect_language(&self, scores: &[f32]) -> usize {
        let tag_scores: Vec<f32> = self
            .tags
            .iter()
            .map(|&t| scores.get(t as usize).copied().unwrap_or(f32::NEG_INFINITY))
            .collect();
        argmax(&tag_scores)
    }
}

/// The index of the largest score, the first on a tie (`np.argmax`). A NaN never beats a
/// number; an all-NaN slice gives 0. (Finite audio through finite weights gives finite scores,
/// so this only matters for a corrupt archive, where any pick is as good as another.)
fn argmax(scores: &[f32]) -> usize {
    let mut best = 0;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] || scores[best].is_nan() {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests;
