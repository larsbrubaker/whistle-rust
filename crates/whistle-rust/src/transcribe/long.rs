//! The windowed decode of audio over 30 s: each window that `window::plan` chose (its header
//! says where the cuts go) is decoded alone by `Whistle::decode_window`, and the results are
//! stitched into one [`Transcript`].
//!
//! - **Language.** The first window whose kept text has at least [`MIN_LANGUAGE_WORDS`] words
//!   is detected (unless `Options::language` forces it), and its language is forced on every
//!   later window, so the transcript has one language, as a caller reading
//!   `Transcript::language` expects. Earlier windows (no text, or a 1–2 word fragment) are
//!   decoded with detection and do not fix it; when no window reaches the count, the first
//!   window with any text labels the transcript. Words rather than seconds, because a window's
//!   word count is known after its decode without word times and is what the tag was decided
//!   from. Forcing does not translate: a recording that switches language is decoded in the
//!   first one, the later part as the decoder's reading of it in that language. A caller that
//!   needs per-part languages decodes the windows `plan_windows` returns one by one.
//! - **Hard cuts.** Each side keeps only its words by midpoint, which needs the words' times,
//!   so a window with a hard cut is always aligned (one extra decoder pass) and its words are
//!   dropped afterwards when the caller did not ask for them; `text` and `tokens` are the same
//!   with timings on and off, as for a single pass.
//! - **Text** joins each window's trimmed text with single spaces, none before a window that
//!   starts with closing punctuation; empty windows add nothing. **Tokens** concatenate the
//!   windows' kept text tokens; they may not decode exactly to the text (a window's first token
//!   is decoded without the space the join adds), so `text` and `words` are authoritative.
//! - **Words** are timed in the input (the window's start is the alignment's offset), in order
//!   and never overlapping across windows; the pauses between `GateAndTrim` windows hold none.
//! - **Cancellation** is checked before each window as well as inside it. No window with text
//!   gives the empty transcript, with no language.

use super::search::Ranking;
use super::window::Window;
use super::{align, Error, Options, Stage, Transcript, Whistle, Word, LANGUAGES};
use crate::audio::SAMPLE_RATE;
use crate::model::HeadFilter;

/// The fewest words a window's kept text needs before its language is forced on the windows
/// after it: a fragment ("Yes.", a name, a hallucinated "Thank you." on a noise burst the gate
/// let through) is too little speech to trust its language tag for the whole recording, while
/// three words already carry the language in every clip measured.
pub(crate) const MIN_LANGUAGE_WORDS: usize = 3;

impl Whistle {
    /// Decodes `windows` of 16 kHz `samples` in order and stitches them. `check` is the call's
    /// cancellation check; it also runs before each window. The words are aligned with
    /// `align::ALIGNMENT_HEADS` (what the call's head filter defaults to) and no alignment pass
    /// is returned: the passes of several windows would not add up to one, and the head
    /// measurements that need a pass run on single windows.
    pub(crate) fn run_windows(
        &self,
        samples: &[f32],
        windows: &[Window],
        forced: Option<usize>,
        options: &Options,
        ranking: Ranking,
        check: &mut dyn FnMut(Stage) -> Result<(), Error>,
    ) -> Result<Transcript, Error> {
        // `language`: what later windows are decoded in; `label`: the transcript's language
        // when no window keeps enough words to fix one (the first window with any text).
        let mut language = forced;
        let mut label = None;
        let mut text = String::new();
        let mut tokens = Vec::new();
        let mut words: Vec<Word> = Vec::new();
        let mut repeated = false;
        for (i, w) in windows.iter().enumerate() {
            check(Stage::Window(i))?;
            let hard = w.keep_from.is_some() || w.keep_until.is_some();
            let Some(decoded) = self.decode_window(
                &samples[w.decode.clone()],
                w.decode.start,
                language,
                options,
                options.word_timestamps || hard,
                ranking,
                HeadFilter::Only(align::ALIGNMENT_HEADS),
                check,
            )?
            else {
                continue;
            };
            let t = decoded.transcript;
            repeated |= t.repeated;
            let (kept_tokens, kept_words) = if hard {
                self.keep_own_words(w, t.tokens, t.words)
            } else {
                (t.tokens, t.words)
            };
            if kept_tokens.is_empty() {
                continue;
            }
            let part = self.tokenizer.decode(&kept_tokens);
            let part = part.trim();
            if language.is_none() && part.split_whitespace().count() >= MIN_LANGUAGE_WORDS {
                language = Some(decoded.language);
            }
            label.get_or_insert(decoded.language);
            join(&mut text, part);
            tokens.extend(kept_tokens);
            if options.word_timestamps {
                append_in_order(&mut words, kept_words);
            }
        }
        if tokens.is_empty() {
            return Ok(Transcript::empty());
        }
        Ok(Transcript {
            text,
            language: language.or(label).map(|i| LANGUAGES[i].to_string()),
            tokens,
            words,
            repeated,
        })
    }

    /// The tokens and words of a hard-cut window that are its own: the words (and the token
    /// groups they come from) whose midpoint lies inside `keep_from..keep_until`.
    fn keep_own_words(
        &self,
        w: &Window,
        tokens: Vec<u32>,
        words: Vec<Word>,
    ) -> (Vec<u32>, Vec<Word>) {
        let rate = SAMPLE_RATE as f32;
        let from = w.keep_from.map_or(f32::NEG_INFINITY, |s| s as f32 / rate);
        let until = w.keep_until.map_or(f32::INFINITY, |s| s as f32 / rate);
        let groups = align::word_groups(&self.tokenizer, &tokens);
        let mut kept_tokens = Vec::new();
        let mut kept_words = Vec::new();
        for (&(s, e), word) in groups.iter().zip(words) {
            let mid = (word.start + word.end) / 2.0;
            if mid >= from && mid < until {
                kept_tokens.extend_from_slice(&tokens[s..e]);
                kept_words.push(word);
            }
        }
        (kept_tokens, kept_words)
    }
}

/// Appends one window's trimmed text to the transcript's, with a single space between them,
/// except before closing punctuation (a hard cut can leave a window starting with the "." or
/// "," that ends the previous window's sentence).
pub(crate) fn join(text: &mut String, part: &str) {
    if part.is_empty() {
        return;
    }
    let closing = part
        .chars()
        .next()
        .is_some_and(|c| ".,;:!?…)]}»%".contains(c));
    if !text.is_empty() && !closing {
        text.push(' ');
    }
    text.push_str(part);
}

/// Appends a window's `next` words after `words`, moving any start before the previous word's
/// end up to it (two overlapping windows time the words near a hard cut independently), so the
/// words stay in order and never overlap.
pub(crate) fn append_in_order(words: &mut Vec<Word>, next: Vec<Word>) {
    for mut word in next {
        if let Some(prev) = words.last() {
            word.start = word.start.max(prev.end);
            word.end = word.end.max(word.start);
        }
        words.push(word);
    }
}

#[cfg(test)]
mod tests {
    use super::join;

    #[test]
    fn windows_join_with_one_space_and_none_before_closing_punctuation() {
        let mut text = String::new();
        for part in ["Make the cube", "", "20 mm wide", ". Then", ", please"] {
            join(&mut text, part);
        }
        assert_eq!(text, "Make the cube 20 mm wide. Then, please");
    }
}
