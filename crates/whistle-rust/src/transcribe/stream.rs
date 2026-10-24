//! Streaming transcription (`docs/design/streaming.md`): [`Whistle::stream`] opens a [`Stream`]
//! that takes live 16 kHz mono audio in chunks of any size and commits the words two
//! consecutive passes agree on (local agreement). Every [`Options::stream_step`] of new audio it
//! runs the detector (`vad`) over its buffer and, if there is speech after the commit point,
//! decodes from the start of the utterance the commit point is in with `Whistle::decode_window`
//! (word times on), finds where the text already committed ends in the pass (a character
//! alignment, `realign`), and commits the longest prefix of the rest that the previous pass
//! agrees on. The buffer keeps 1 s before that utterance; a pause after speech commits the
//! whole utterance, and a buffer about to pass 30 s force-commits its older words. Committed
//! words are never retracted; the stitching (`long::join`, `long::append_in_order`) and language
//! (`long::MIN_LANGUAGE_WORDS`) rules are the windowed decode's. Stream positions are `u64`
//! samples, so a stream runs indefinitely on every target.

use super::long::{append_in_order, join, MIN_LANGUAGE_WORDS};
use super::search::Ranking;
use super::{
    align, check_finite, Cancel, Error, Options, Stage, Transcript, VadMode, Whistle, Word,
    LANGUAGES, MAX_SAMPLES, OVERLAP_SAMPLES,
};
use crate::audio::SAMPLE_RATE;
use crate::model::HeadFilter;
use crate::vad;

/// The default [`Options::stream_step`]: one decode per second of new audio.
pub const DEFAULT_STREAM_STEP: f32 = 1.0;

const RATE: u64 = SAMPLE_RATE as u64;

/// Audio the buffer keeps before the decoded utterance (1 s): the detector's noise floor looks
/// 1 s each side of a frame, so a buffer that starts mid-speech still holds a pause to measure
/// the floor in.
const CONTEXT: u64 = RATE;

/// Audio decoded before a commit point inside speech, where a pass cannot start at a pause (at
/// a forced cut, and after every commit with the detector off): 1 s, as windowing overlaps a
/// hard cut ([`OVERLAP_SAMPLES`]), so the word the cut is in is whole in the pass and is
/// recognised as committed by its content.
const LEAD_IN: u64 = OVERLAP_SAMPLES as u64;

/// Silence after the last speech that ends an utterance and commits everything pending (1 s
/// past the detector's segment end, which already includes its 0.2 s hangover).
const ENDPOINT: u64 = RATE;

/// Speech after the commit point shorter than this (0.1 s, the detector's `min_speech_s`) is not
/// decoded: re-running the detector on a grown buffer can move a segment's end by a few frames,
/// and decoding that sliver would invent a word.
const MIN_SPEECH: u64 = RATE / 10;

/// How close to the end of the audio a pass decoded a word may end and still be committed by
/// agreement (0.3 s): a word the buffer's end cuts can read as a shorter word ("den" cut to
/// "in") in two passes alike, so agreement only counts once the previous pass heard past it.
const EDGE: u64 = RATE * 3 / 10;

/// Where the commit point moves when the buffer holds no speech: this far (0.5 s) before its
/// end, so an onset the detector cannot confirm yet is still decoded by the next pass.
const ONSET: u64 = RATE / 2;

/// An alignment that changes more than half the committed characters is no alignment: the
/// pass read the utterance too differently to say where the committed text ends in it.
const RESTART_SHARE: usize = 2;

/// The lead-in a restarted pass decodes before the commit point (0.1 s, the detector's
/// `hangover_before_s`): a pass with none misreads its first word ("Please" as "These").
const RESTART_LEAD_IN: u64 = RATE / 10;

/// What one [`Stream::push`] or [`Stream::finish`] committed, and what is still unconfirmed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StreamUpdate {
    /// The words this update committed, with times in seconds from the start of the stream.
    pub committed: Vec<Word>,
    /// The committed words joined (as [`Transcript::text`] joins words).
    pub text: String,
    /// The latest pass's words after the committed ones: what may still change.
    pub pending: String,
    /// The language the stream is in: the forced one, the one fixed by the first pass that
    /// contributed at least 3 words, the first commit's, or the latest pass's detection; `None`
    /// while there are no words, committed or pending (as [`Stream::transcript`] is `None` while
    /// none are committed, and [`Whistle::transcribe`] for an empty transcript).
    pub language: Option<String>,
    /// Seconds of audio this update dropped without committing it: only at the 30 s cap, when
    /// the passes decoded no words for audio the detector calls speech (noise it lets through,
    /// mumbling), so nothing could be committed to make room.
    pub dropped: f32,
    /// `text` begins with the rest of the last word committed before this update (the pass
    /// read it longer: "2" committed, then "20"), so a caller appending it adds no space.
    pub continues: bool,
}

/// What aligning a pass against the committed text found.
enum Aligned {
    /// The pass's words after the committed text.
    New(Vec<Hyp>),
    /// More than half the committed characters had to change: restart the utterance.
    Restart,
    /// The pass decoded (almost) nothing: under half as many characters as are committed.
    Nothing,
}

/// One word of a pass with the token ids it was decoded from.
#[derive(Debug, Clone)]
struct Hyp {
    word: Word,
    tokens: Vec<u32>,
    /// The rest of a word whose start is already committed (the alignment cut inside it): its
    /// text continues the previous word's with no space.
    glued: bool,
}

/// A live transcription: push audio as it arrives, read the committed words from each
/// [`StreamUpdate`], and [`Stream::finish`] at the end. Opened by [`Whistle::stream`].
pub struct Stream<'a> {
    whistle: &'a Whistle,
    options: Options,
    /// The language every pass is decoded in: the forced one, or the one fixed by the first
    /// pass that contributed at least [`MIN_LANGUAGE_WORDS`] words and committed.
    language: Option<usize>,
    /// The language of the first commit, the label until one is fixed.
    label: Option<usize>,
    /// The latest pass's language, the label before anything is committed.
    detected: Option<usize>,
    /// Samples of new audio between passes.
    step: usize,
    /// The held audio; `buffer[0]` is sample `start` of the stream.
    buffer: Vec<f32>,
    start: u64,
    /// Everything before this sample is committed (or silence).
    commit: u64,
    /// Samples received since the last pass.
    since: usize,
    /// No pass decodes before this sample: a commit point inside speech minus [`LEAD_IN`].
    floor: u64,
    /// The floor a restart set: passes from it take their cut whatever it costs, so a restart
    /// is not taken again (and again) on the same lead-in.
    restart_floor: Option<u64>,
    /// Where the latest pass's decoded audio ended.
    decoded_end: u64,
    /// The latest pass's words after the committed ones.
    pending: Vec<Hyp>,
    /// Everything committed so far.
    words: Vec<Word>,
    tokens: Vec<u32>,
    text: String,
    /// Samples dropped uncommitted at the cap, in total.
    dropped: u64,
}

impl Whistle {
    /// Opens a live transcription with `options` (`docs/design/streaming.md`). The options are
    /// checked as [`Whistle::transcribe`] checks them, plus [`Options::stream_step`]; the words
    /// are always aligned, whatever [`Options::word_timestamps`] says, since agreement and
    /// trimming need their times.
    pub fn stream(&self, options: &Options) -> Result<Stream<'_>, Error> {
        let language = self.check_options(options)?;
        let step = options.stream_step;
        if !step.is_finite() || !(0.1..=10.0).contains(&step) {
            return Err(Error::Input(format!(
                "stream step {step} s; it must be 0.1 to 10 s"
            )));
        }
        Ok(Stream {
            whistle: self,
            options: options.clone(),
            language,
            label: None,
            detected: None,
            // At most 10 s, so the cast cannot overflow.
            step: (step * SAMPLE_RATE as f32).round() as usize,
            buffer: Vec::new(),
            start: 0,
            commit: 0,
            since: 0,
            floor: 0,
            restart_floor: None,
            decoded_end: 0,
            pending: Vec::new(),
            words: Vec::new(),
            tokens: Vec::new(),
            text: String::new(),
            dropped: 0,
        })
    }
}

impl Stream<'_> {
    /// Appends 16 kHz mono `samples` (any length) and runs a pass for every
    /// [`Options::stream_step`] of audio completed, all before it returns, so a push of many
    /// seconds blocks for many passes: live callers push small chunks (a step or less) as they
    /// arrive. A non-finite sample is [`Error::Input`] (nothing of the chunk is taken); a fired
    /// [`Options::cancel`] is [`Error::Cancelled`], checked on entry and at every pass. After an
    /// error [`Stream::transcript`] still holds every word committed before it (an update that
    /// failed is not returned, so its words reach the caller only through the transcript).
    pub fn push(&mut self, samples: &[f32]) -> Result<StreamUpdate, Error> {
        check_finite(samples)?;
        self.cancelled()?;
        let mut update = StreamUpdate::default();
        let mut rest = samples;
        while !rest.is_empty() {
            let left = self.step - self.since;
            // Room is made for the whole step at its first sample, so the cut does not depend
            // on how the caller chunks the audio.
            if self.buffer.len() + left > MAX_SAMPLES {
                self.force_commit(left, &mut update);
            }
            let take = left.min(rest.len());
            self.buffer.extend_from_slice(&rest[..take]);
            self.since += take;
            rest = &rest[take..];
            if self.since >= self.step {
                self.since = 0;
                self.pass(false, false, &mut update)?;
            }
        }
        self.finish_update(&mut update);
        Ok(update)
    }

    /// Ends the stream: decodes the audio since the last pass, if any, and commits every word
    /// left.
    pub fn finish(mut self) -> Result<StreamUpdate, Error> {
        self.cancelled()?;
        let mut update = StreamUpdate::default();
        if self.since == 0 {
            // The latest pass saw all the audio: its words are the final ones.
            let pending = std::mem::take(&mut self.pending);
            self.commit_words(pending, self.detected, &mut update);
        } else {
            self.pass(true, false, &mut update)?;
        }
        self.finish_update(&mut update);
        Ok(update)
    }

    /// Everything committed so far, as one transcript (times from the start of the stream).
    pub fn transcript(&self) -> Transcript {
        Transcript {
            text: self.text.clone(),
            language: self.label(!self.words.is_empty()),
            tokens: self.tokens.clone(),
            words: self.words.clone(),
        }
    }

    /// Samples of audio the stream holds: never more than [`MAX_SAMPLES`] (30 s).
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Seconds of audio dropped uncommitted at the 30 s cap so far ([`StreamUpdate::dropped`]).
    pub fn dropped(&self) -> f32 {
        seconds(self.dropped)
    }

    fn cancelled(&self) -> Result<(), Error> {
        if self
            .options
            .cancel
            .as_ref()
            .is_some_and(Cancel::is_cancelled)
        {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// The language to report when there are words to label (`any`), by one rule for updates
    /// and transcripts: fixed or forced, else the first commit's, else the latest detection.
    fn label(&self, any: bool) -> Option<String> {
        let index = self.language.or(self.label).or(self.detected);
        index.filter(|_| any).map(|i| LANGUAGES[i].to_string())
    }

    fn finish_update(&self, update: &mut StreamUpdate) {
        let mut pending = String::new();
        for h in &self.pending {
            glue_or_join(&mut pending, h.word.word.trim(), h.glued);
        }
        update.pending = pending;
        update.language = self.label(!self.words.is_empty() || !self.pending.is_empty());
    }

    fn end(&self) -> u64 {
        self.start + self.buffer.len() as u64
    }

    /// One pass over the buffer; `last` commits every word it finds (the end of the stream);
    /// `restarted` is the second run of a pass whose alignment failed, which takes its cut
    /// whatever it costs.
    fn pass(
        &mut self,
        last: bool,
        restarted: bool,
        update: &mut StreamUpdate,
    ) -> Result<(), Error> {
        self.cancelled()?;
        let end = self.end();
        let off = self.options.vad == VadMode::Off;
        let speech: Vec<(u64, u64)> = if off {
            vec![(self.start, end)]
        } else {
            vad::speech_segments(&self.buffer)
                .into_iter()
                .map(|s| (self.start + s.start as u64, self.start + s.end as u64))
                .filter(|&(_, e)| e > self.commit + MIN_SPEECH)
                .collect()
        };
        let (Some(&(first, _)), Some(&(_, speech_end))) = (speech.first(), speech.last()) else {
            // No speech since the commit point: what is pending was said before a pause that is
            // now long enough to end the utterance.
            let pending = std::mem::take(&mut self.pending);
            self.commit_words(pending, self.detected, update);
            self.commit = self.commit.max(end.saturating_sub(ONSET));
            self.trim_to(self.commit.saturating_sub(CONTEXT));
            return Ok(());
        };
        // The decode starts where the speech the commit point is in starts (or the next speech
        // after it), so the decoder hears the utterance from its start, never from inside a
        // word; it starts later only where a commit point inside speech set the floor.
        let decode_start = first.max(self.floor).max(self.start);
        if !off {
            self.trim_to(decode_start.saturating_sub(CONTEXT));
        }
        let decode_end = if self.options.vad == VadMode::GateAndTrim {
            speech_end
        } else {
            end
        };
        let endpoint = !off && speech_end + ENDPOINT <= end;
        let (hyps, language) = self.decode(decode_start, decode_end)?;
        let hyps = match self.after_committed(hyps, decode_start, restarted) {
            Aligned::New(hyps) => hyps,
            Aligned::Restart => {
                // Restart the utterance at the commit point and run this pass again from there,
                // with the detector's 0.1 s of lead-in; passes from that floor never restart.
                self.pending.clear();
                self.floor = self.floor.max(self.commit.saturating_sub(RESTART_LEAD_IN));
                self.restart_floor = Some(self.floor);
                return self.pass(last, true, update);
            }
            Aligned::Nothing => {
                // The pass heard (almost) nothing of what is committed and pending: no
                // information. What is pending stays, unless the stream or utterance ends.
                if last || endpoint {
                    let pending = std::mem::take(&mut self.pending);
                    self.commit_words(pending, self.detected, update);
                    self.commit = self.commit.max(decode_end);
                }
                return Ok(());
            }
        };
        let previous_end = seconds(self.decoded_end.saturating_sub(EDGE));
        self.decoded_end = decode_end;
        if language.is_some() {
            self.detected = language;
        }
        let committed = self.words.len();
        if last || endpoint {
            self.pending.clear();
            self.commit_words(hyps, language, update);
            self.commit = self.commit.max(decode_end);
        } else {
            let agreed = self
                .pending
                .iter()
                .zip(&hyps)
                .take_while(|(a, b)| {
                    a.word.end <= previous_end && normalize(&a.word.word) == normalize(&b.word.word)
                })
                .count();
            let mut hyps = hyps;
            self.pending = hyps.split_off(agreed);
            self.commit_words(hyps, language, update);
        }
        if self.words.len() > committed {
            // At least 3 words committed by this pass: enough speech to trust its language tag.
            let fixes = self.words.len() - committed >= MIN_LANGUAGE_WORDS;
            if fixes && self.language.is_none() {
                self.language = language;
            }
            if off {
                // No pauses to start a pass at: start the next one a lead-in before the commit
                // point, so an Off stream does not grow to 30 s and decode all of it every step.
                self.floor = self.floor.max(self.commit.saturating_sub(LEAD_IN));
                self.trim_to(self.floor);
            }
        }
        Ok(())
    }

    /// The words of `start..end` (stream samples) decoded in the stream's language, with times
    /// from the start of the stream, and the language decoded in.
    fn decode(&self, start: u64, end: u64) -> Result<(Vec<Hyp>, Option<usize>), Error> {
        let w = self.whistle;
        let cancel = self.options.cancel.clone();
        let mut check = |_: Stage| match &cancel {
            Some(c) if c.is_cancelled() => Err(Error::Cancelled),
            _ => Ok(()),
        };
        // Both lie in the buffer (at most 30 s), so the offsets fit any usize.
        let audio = &self.buffer[(start - self.start) as usize..(end - self.start) as usize];
        // Decoded at offset 0 and moved by the stream position here, which a usize sample
        // offset could not hold past 74 hours on wasm32.
        let Some(decoded) = w.decode_window(
            audio,
            0,
            self.language,
            &self.options,
            true,
            Ranking::Mean,
            HeadFilter::Only(align::ALIGNMENT_HEADS),
            &mut check,
        )?
        else {
            return Ok((Vec::new(), None));
        };
        let t = decoded.transcript;
        let base = seconds(start);
        let groups = align::word_groups(&w.tokenizer, &t.tokens);
        let hyps = groups
            .iter()
            .zip(t.words)
            .map(|(&(s, e), mut word)| {
                word.start += base;
                word.end += base;
                Hyp {
                    word,
                    tokens: t.tokens[s..e].to_vec(),
                    glued: false,
                }
            })
            .collect();
        Ok((hyps, Some(decoded.language)))
    }

    /// The words of a pass decoded from `decode_start` after the text committed in the
    /// utterance, found by aligning the committed text against the pass ([`realign::cut`]).
    /// The committed text is that of the words the pass can hear: at the start of an utterance
    /// (a pause) those whose midpoint lies after `decode_start` (a word's end can run on to the
    /// next word's start, across the pause); from a floor inside speech every word that ends
    /// after it, so the alignment absorbs the fragment of a word the lead-in hears.
    fn after_committed(&self, mut hyps: Vec<Hyp>, decode_start: u64, restarted: bool) -> Aligned {
        let region = seconds(decode_start);
        let inside = decode_start == self.floor && self.floor > 0;
        let heard: Vec<&Word> = self
            .words
            .iter()
            .filter(|w| {
                if inside {
                    w.end > region
                } else {
                    (w.start + w.end) / 2.0 >= region
                }
            })
            .collect();
        let committed: String = heard.iter().map(|w| normalize(&w.word)).collect();
        let marks = heard
            .iter()
            .rev()
            .take_while(|w| normalize(&w.word).is_empty())
            .count();
        let spelled: Vec<String> = hyps.iter().map(|h| normalize(&h.word.word)).collect();
        let size = committed.chars().count();
        let heard_chars: usize = spelled.iter().map(|w| w.chars().count()).sum();
        if realign::says_nothing(size, heard_chars) {
            return Aligned::Nothing;
        }
        let last = heard
            .iter()
            .rev()
            .map(|w| normalize(&w.word))
            .find(|w| !w.is_empty())
            .unwrap_or_default();
        let cut = realign::cut(&committed, &last, &spelled, marks);
        let settled = restarted || self.restart_floor == Some(decode_start);
        if !settled && cut.cost * RESTART_SHARE > size {
            return Aligned::Restart;
        }
        let mut new = hyps.split_off(cut.word.min(hyps.len()));
        if cut.offset > 0 {
            if let Some(first) = new.first_mut() {
                self.keep_rest(first, cut.offset, spelled[cut.word].chars().count());
            }
        }
        Aligned::New(new)
    }

    /// Turns `h` into the rest of its word after `offset` of its `len` normalized characters:
    /// its own text from there, the tokens from the one that holds that character on, and the
    /// share of its time the rest takes; glued to the committed word before it.
    fn keep_rest(&self, h: &mut Hyp, offset: usize, len: usize) {
        let tokenizer = &self.whistle.tokenizer;
        let split = (0..h.tokens.len())
            .find(|&t| {
                normalize(&tokenizer.decode(&h.tokens[..=t]))
                    .chars()
                    .count()
                    > offset
            })
            .unwrap_or(h.tokens.len());
        h.tokens.drain(..split);
        h.word.word = realign::rest_of(&h.word.word, offset).to_string();
        let share = offset as f32 / len.max(1) as f32;
        h.word.start += (h.word.end - h.word.start) * share;
        h.glued = true;
    }

    /// Commits `hyps` (decoded in `language`) and moves the commit point to the last one's end.
    fn commit_words(&mut self, hyps: Vec<Hyp>, language: Option<usize>, update: &mut StreamUpdate) {
        if hyps.is_empty() {
            return;
        }
        if let Some(language) = language {
            self.label.get_or_insert(language);
        }
        let mut words = Vec::with_capacity(hyps.len());
        let mut glued = Vec::with_capacity(hyps.len());
        for h in hyps {
            self.tokens.extend(h.tokens);
            words.push(h.word);
            glued.push(h.glued);
        }
        let first_new = self.words.len();
        append_in_order(&mut self.words, words);
        let new = &self.words[first_new..];
        if update.committed.is_empty() && glued.first() == Some(&true) {
            update.continues = true;
        }
        for (word, &glued) in new.iter().zip(&glued) {
            glue_or_join(&mut self.text, word.word.trim(), glued);
            glue_or_join(&mut update.text, word.word.trim(), glued);
        }
        update.committed.extend_from_slice(new);
        // The end after `append_in_order` moved it, so the commit point is the committed time.
        if let Some(last) = self.words.last() {
            let end = (f64::from(last.end) * RATE as f64).round().max(0.0) as u64;
            self.commit = self.commit.max(end);
        }
    }

    /// Drops the audio before stream sample `keep_from`.
    fn trim_to(&mut self, keep_from: u64) {
        // At most the buffer's length, so the cast cannot overflow.
        let drop = keep_from
            .saturating_sub(self.start)
            .min(self.buffer.len() as u64) as usize;
        self.buffer.drain(..drop);
        self.start += drop as u64;
    }

    /// Makes room for `incoming` samples in a buffer at the 30 s cap: commits the latest pass's
    /// words that end at least [`OVERLAP_SAMPLES`] before the audio it decoded ends (a word cut
    /// by the buffer's end is not committed), starts later passes a [`LEAD_IN`] before the new
    /// commit point, then drops whatever audio still does not fit, reporting it as dropped.
    fn force_commit(&mut self, incoming: usize, update: &mut StreamUpdate) {
        let limit = seconds(self.decoded_end.saturating_sub(OVERLAP_SAMPLES as u64));
        let keep = self
            .pending
            .iter()
            .position(|h| h.word.end > limit)
            .unwrap_or(self.pending.len());
        let rest = self.pending.split_off(keep);
        let forced = std::mem::replace(&mut self.pending, rest);
        self.commit_words(forced, self.detected, update);
        let fit = (self.end() + incoming as u64).saturating_sub(MAX_SAMPLES as u64);
        self.floor = self.commit.saturating_sub(LEAD_IN).max(fit);
        self.trim_to(self.floor.saturating_sub(CONTEXT).max(fit));
        if self.commit < fit {
            // The passes left nothing to commit for this audio: it goes, reported.
            self.dropped += fit - self.commit;
            update.dropped += seconds(fit - self.commit);
            self.commit = fit;
            let commit = seconds(self.commit);
            self.pending
                .retain(|h| (h.word.start + h.word.end) / 2.0 >= commit);
        }
    }
}

/// Seconds from the start of the stream. f32, as [`Word`] times are: exact to the sample for
/// the first 17 minutes, to about 30 ms at 74 hours.
fn seconds(sample: u64) -> f32 {
    (sample as f64 / RATE as f64) as f32
}

/// Appends `word` to `text`: straight on when it is the rest of the word before (`glued`),
/// otherwise as `long::join` joins words.
fn glue_or_join(text: &mut String, word: &str, glued: bool) {
    if glued {
        text.push_str(word);
    } else {
        join(text, word);
    }
}

/// A word as agreement compares it: lower case, letters and digits only. Two passes that read
/// the same word often differ only in its punctuation or capital ("print" mid-sentence, then
/// "print." once the sentence ends in the buffer), which is no reason to hold it back; the text
/// committed is the newer pass's, punctuation included.
fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

mod realign;

#[cfg(test)]
mod tests;
