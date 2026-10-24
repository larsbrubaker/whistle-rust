//! Word timestamps from decoder cross-attention (`docs/design/timestamps.md`): after the search,
//! `transcribe` runs one teacher-forced decoder pass over the final tokens
//! (`model::Decoder::forward_with_cross_attention`); this module turns the chosen alignment
//! heads' token-to-frame probabilities into a monotonic token-to-frame path (normalise each
//! head, median-filter over frames, average, dynamic time warping on the negated weights), groups
//! the text tokens into words at `▁` boundaries, and gives each word contiguous start and end
//! times and a probability from the pass's logits.

use crate::model::{CrossAttentionWeights, Logits};
use crate::tokenizer::{PieceType, Tokenizer};

/// One encoder frame, the alignment's resolution: 80 ms (the front end's 10 ms hop, ×8 by the
/// stem).
pub const FRAME_SECONDS: f32 = 0.08;

/// The (layer, head) pairs whose cross-attention follows the speech, chosen by measurement on
/// the contract clips (`docs/design/timestamps.md` has the per-head scores).
pub(crate) const ALIGNMENT_HEADS: &[(usize, usize)] = &[(7, 3), (4, 1), (5, 4)];

/// The pass row that predicts text token 0 (the tag's input row, after BOS's): text token `i`
/// is aligned and scored by row `TEXT_ROW + i`, whose output chooses it. Its attention aligned
/// better than the token's own input row (`docs/design/timestamps.md`). Row 0 (BOS's, which
/// predicts the tag) and the row that predicts EOS bracket the text as anchors.
pub(crate) const TEXT_ROW: usize = 1;

/// The median filter's width over frames (odd), smoothing single-frame spikes in a head.
pub(crate) const MEDIAN_WIDTH: usize = 3;

/// A word of the transcript with its time in the input audio and its probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    /// The word as it appears in [`super::Transcript::text`], punctuation attached.
    pub word: String,
    /// Seconds from the start of the input (before any trimming).
    pub start: f32,
    /// Seconds from the start of the input; the next word's start, or the last token's end.
    pub end: f32,
    /// `exp` of the mean log-probability of the word's tokens under the decoder.
    pub probability: f32,
}

/// The `[tokens, frames]` alignment matrix over pass rows `first_row..first_row + tokens`:
/// each chosen head's rows, normalised per frame across the
/// tokens (zero mean, unit variance), median-filtered over frames, averaged over the heads.
pub(crate) fn alignment_matrix(
    cross: &CrossAttentionWeights,
    heads: &[(usize, usize)],
    first_row: usize,
    tokens: usize,
) -> Vec<f32> {
    let f = cross.frames;
    let mut sum = vec![0.0f32; tokens * f];
    let mut used = 0usize;
    for &(layer, head) in heads {
        let rows: Option<Vec<&[f32]>> = (0..tokens)
            .map(|i| cross.row(layer, head, first_row + i))
            .collect();
        let Some(rows) = rows else { continue };
        let mut m: Vec<f32> = rows.concat();
        normalise_columns(&mut m, tokens, f);
        for row in m.chunks_exact_mut(f.max(1)) {
            let filtered = median_filter(row, MEDIAN_WIDTH);
            row.copy_from_slice(&filtered);
        }
        for (s, v) in sum.iter_mut().zip(&m) {
            *s += v;
        }
        used += 1;
    }
    if used > 0 {
        let n = used as f32;
        sum.iter_mut().for_each(|v| *v /= n);
    }
    sum
}

/// Each frame's column scaled to zero mean and unit variance across the tokens, so a frame
/// every token attends to (silence, an attention sink) does not pull the path.
fn normalise_columns(m: &mut [f32], rows: usize, cols: usize) {
    if rows == 0 {
        return;
    }
    for c in 0..cols {
        let mean = (0..rows).map(|r| m[r * cols + c]).sum::<f32>() / rows as f32;
        let var = (0..rows)
            .map(|r| (m[r * cols + c] - mean).powi(2))
            .sum::<f32>()
            / rows as f32;
        let sd = var.sqrt().max(1e-6);
        for r in 0..rows {
            m[r * cols + c] = (m[r * cols + c] - mean) / sd;
        }
    }
}

/// A median filter of odd `width` over `row`, the window shrinking at the edges.
pub(crate) fn median_filter(row: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    let mut window = Vec::with_capacity(width);
    (0..row.len())
        .map(|i| {
            window.clear();
            window.extend_from_slice(&row[i.saturating_sub(half)..(i + half + 1).min(row.len())]);
            window.sort_unstable_by(f32::total_cmp);
            window[window.len() / 2]
        })
        .collect()
}

/// Dynamic time warping through `cost` (`[rows, cols]`) from the top-left to the bottom-right
/// corner, steps down, right or diagonal: the cheapest monotonic path, as (row, col) pairs.
pub(crate) fn dtw(cost: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let w = cols + 1;
    let mut acc = vec![f32::INFINITY; (rows + 1) * w];
    // 0: diagonal, 1: from above (next token, same frame), 2: from the left (same token).
    let mut from = vec![0u8; (rows + 1) * w];
    acc[0] = 0.0;
    for i in 1..=rows {
        for j in 1..=cols {
            let options = [
                acc[(i - 1) * w + j - 1],
                acc[(i - 1) * w + j],
                acc[i * w + j - 1],
            ];
            let mut best = 0;
            for k in 1..3 {
                if options[k] < options[best] {
                    best = k;
                }
            }
            acc[i * w + j] = cost[(i - 1) * cols + j - 1] + options[best];
            from[i * w + j] = best as u8;
        }
    }
    let (mut i, mut j) = (rows, cols);
    let mut path = Vec::with_capacity(rows + cols);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        match from[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

/// Each token's first frame and one past its last on the DTW path through `−matrix`. The path
/// runs corner to corner moving at most one row per step, so it visits every row: each token
/// gets at least one frame. With more tokens than frames, tokens share frames, and words can
/// get `start == end`.
pub(crate) fn token_spans(matrix: &[f32], tokens: usize, frames: usize) -> Vec<(usize, usize)> {
    let cost: Vec<f32> = matrix.iter().map(|v| -v).collect();
    let mut spans = vec![(usize::MAX, 0usize); tokens];
    for (t, f) in dtw(&cost, tokens, frames) {
        let s = &mut spans[t];
        s.0 = s.0.min(f);
        s.1 = s.1.max(f + 1);
    }
    spans
}

/// Whether text token `id` starts a word: its piece begins with `▁` (a byte piece, with a
/// space byte). Punctuation and continuation pieces stay with the word before them.
pub(crate) fn starts_word(tokenizer: &Tokenizer, id: u32) -> bool {
    match tokenizer.piece_type(id) {
        Some(PieceType::Byte(b)) => b == b' ',
        Some(PieceType::Normal | PieceType::UserDefined) => {
            tokenizer.piece(id).is_some_and(|p| p.starts_with('▁'))
        }
        _ => false,
    }
}

/// The text tokens grouped into words: index ranges into `tokens`, each starting at a word
/// start (the first group also takes any leading continuation tokens). A group that decodes to
/// no visible text (a lone `▁`) joins the next word, or the previous one at the end. Text that
/// starts with a group holding no letter or digit (a lone opening mark, as in "« Bonjour")
/// joins it to the first word, so the first word's time is where the speech starts; that
/// word's text then holds the space ("« Bonjour").
pub(crate) fn word_groups(tokenizer: &Tokenizer, tokens: &[u32]) -> Vec<(usize, usize)> {
    let text = |g: (usize, usize)| tokenizer.decode(&tokens[g.0..g.1]);
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 1..=tokens.len() {
        if i == tokens.len() || starts_word(tokenizer, tokens[i]) {
            if text((start, i)).trim().is_empty() {
                if i < tokens.len() {
                    continue;
                }
                // Trailing: join the previous word, or (nothing visible at all) no words.
                if let Some(last) = groups.last_mut() {
                    last.1 = i;
                }
                break;
            }
            groups.push((start, i));
            start = i;
        }
    }
    if groups.len() > 1 && !text(groups[0]).chars().any(char::is_alphanumeric) {
        let first = groups.remove(0);
        groups[0].0 = first.0;
    }
    groups
}

/// The log-probability of each text token under `logits` (row [`TEXT_ROW`]` + i` predicts
/// text token `i`), a log-softmax over the ids below `text_ids` (the text pieces and EOS, the ids
/// the search chooses among).
pub(crate) fn token_log_probs(logits: &Logits, tokens: &[u32], text_ids: usize) -> Vec<f32> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let Some(row) = logits.row(TEXT_ROW + i) else {
                return f32::NEG_INFINITY;
            };
            let row = &row[..text_ids.min(row.len())];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row
                .iter()
                .map(|&v| crate::math::exp(f64::from(v - max)))
                .sum();
            let lse = max + crate::math::ln(sum) as f32;
            row.get(t as usize).map_or(f32::NEG_INFINITY, |&v| v - lse)
        })
        .collect()
}

/// What [`words`] needs about the pass and the audio.
pub(crate) struct Alignment<'a> {
    pub(crate) cross: &'a CrossAttentionWeights,
    pub(crate) logits: &'a Logits,
    /// The text tokens (no BOS, tag or EOS).
    pub(crate) tokens: &'a [u32],
    pub(crate) text_ids: usize,
    /// Seconds trimmed from the front of the input before decoding.
    pub(crate) offset: f32,
    /// Seconds of audio decoded (after trimming).
    pub(crate) duration: f32,
}

/// The words of a transcript with contiguous times (each word ends where the next starts) and
/// probabilities, from `heads`' cross-attention.
pub(crate) fn words(
    tokenizer: &Tokenizer,
    a: &Alignment<'_>,
    heads: &[(usize, usize)],
) -> Vec<Word> {
    let n = a.tokens.len();
    let frames = a.cross.frames;
    if n == 0 || frames == 0 {
        return Vec::new();
    }
    // Anchor rows around the text: row 0 (the tag's prediction) takes the lead-in frames and
    // EOS's prediction the trailing ones, so the path need not stretch the first and last
    // words over silence. A decode cut at `max_seq_len` has no EOS row.
    let eos = usize::from(TEXT_ROW + n < a.cross.positions);
    let rows = TEXT_ROW + n + eos;
    let matrix = alignment_matrix(a.cross, heads, 0, rows);
    let spans = token_spans(&matrix, rows, frames)[TEXT_ROW..].to_vec();
    let log_probs = token_log_probs(a.logits, a.tokens, a.text_ids);
    let groups = word_groups(tokenizer, a.tokens);
    let clamp = |frame: usize| (frame as f32 * FRAME_SECONDS).min(a.duration) + a.offset;
    let starts: Vec<f32> = groups.iter().map(|&(s, _)| clamp(spans[s].0)).collect();
    groups
        .iter()
        .enumerate()
        .map(|(g, &(s, e))| {
            let end = match starts.get(g + 1) {
                Some(&next) => next,
                // Where EOS starts (after the last token's first frame); without EOS, where
                // the last token ends.
                None if eos == 1 => clamp(spans[n].0.max(spans[e - 1].0 + 1)),
                None => clamp(spans[e - 1].1),
            };
            let mean = log_probs[s..e].iter().sum::<f32>() / (e - s) as f32;
            Word {
                word: tokenizer.decode(&a.tokens[s..e]).trim().to_string(),
                start: starts[g],
                end: end.max(starts[g]),
                probability: crate::math::expf(mean),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
