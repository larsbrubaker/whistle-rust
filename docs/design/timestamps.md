# Word timestamps (`Options::word_timestamps`)

The engine documents word timestamps as "start, end, and probability per word, aligned from
decoder attention". Nothing else about how it does it is known, so this is designed here; the
engine's output (`test-data/engine/*.json` `words`) is used only as data to measure against
and to pick the alignment heads.

## What a caller gets

`Transcript::words: Vec<Word { word, start, end, probability }>`, empty unless
`Options::word_timestamps` is set (default off) and empty for an empty transcript. `word` is a
whitespace-separated word of `text`, punctuation attached ("edges."). Times are seconds from the
start of the input: with `VadMode::GateAndTrim` the trimmed lead-in is added back. Words are
contiguous, as the engine's are: each word's `end` is the next word's `start`; the last word
ends where the decoder's EOS takes over (below), clamped to the decoded audio.

## How (`transcribe::align`)

1. **One teacher-forced pass.** After the search, `Decoder::forward_with_cross_attention` runs
   BOS, the tag, the text ids and EOS through the full decoder once. It is
   `Decoder::forward` with each layer's cross-attention swapped for
   `CrossAttention::forward_with_weights`: the same call, with a `ProbSink` handed to `attend`,
   which copies the wanted heads' probabilities over the valid frames right after its softmax.
   They are the weights the model mixed the values with, by construction
   (`model/cross_attention/tests.rs` rebuilds the attention output from them bit for bit), and
   with no sink `attend` does nothing more. The logits are bit-identical to `forward`'s
   (`tests/timestamps/capture.rs`), and nothing on the decode path changes, so `text` and
   `tokens` are the same with timings on and off. Only the heads asked for are kept
   (`HeadFilter::Only(ALIGNMENT_HEADS)`: about 1.4 MB at the longest decode over 30 s;
   `HeadFilter::All`, for the sweep, about 31 MB). A cancel is checked just before the pass
   (`Stage::Align`).
2. **Rows.** Text token `i` is aligned with row `1 + i`, the row whose output predicts it (its
   query is formed while the model decides on the token). Measured against row `2 + i` (the
   token's own input row), see the table: the predicting row is better on every count. Two
   anchor rows bracket the text: row 0 (BOS, predicting the tag) absorbs the lead-in frames and
   EOS's predicting row absorbs the trailing ones, so the path need not stretch the first and
   last words over silence. Without either anchor three last-word ends were 2 frames late;
   with only the tail anchor one still was; with both every time is within one frame.
3. **Combine.** For each chosen head: the `[rows, valid frames]` probabilities, each frame's
   column normalised to zero mean and unit variance across the rows (a frame every token
   attends to does not pull the path), a median filter of width 3 over frames, then the mean
   over the heads.
4. **DTW.** Dynamic time warping on the negated matrix from (0, 0) to (rows − 1, frames − 1),
   steps diagonal, down or right: a monotonic token-to-frame path over the valid frames only.
   A token's span is its first to one past its last frame on the path.
5. **Words.** A text id starts a word when its piece begins with `▁` (a byte piece: the space
   byte); continuation pieces, punctuation and other byte pieces stay with the word before. A
   group that decodes to nothing visible (a lone `▁`) joins the next word, or the previous one
   when it ends the text. Text that opens with a group holding no letter or digit (a spaced
   opening mark, "« Bonjour") joins it to the first word, whose text then holds the space
   ("« Bonjour"), so the first word starts where the speech does. A word starts at its first
   token's first frame × 80 ms (clamped to the decoded audio, plus the trim offset); it ends at
   the next word's start, the last at EOS's first frame. The DTW path visits every row, so each
   token has a frame; with more tokens than frames tokens share frames, and a word can get
   `start == end`.
6. **Probability.** `exp` of the mean log-probability of the word's tokens, each a log-softmax
   over the ids the search chooses among (text ids and EOS, not the tags) of the pass's logits.
   Against the engine's on the 73 words: mean 0.888 ours vs 0.895, mean |difference| 0.013, the
   largest gaps on low-confidence words (0.634 vs 0.737, 0.719 vs 0.795). Close enough that a
   better-calibrated alternative was not pursued.

## Head selection (measured 2026-10-24)

`align::ALIGNMENT_HEADS = [(7, 3), (4, 1), (5, 4)]` (layer, head), from
`src/transcribe/align/tests.rs` `alignment_head_sweep` (ignored; release, ~10 s). Each of the
64 heads alone through the whole pipeline, on the 8 contract clips at 3 beams; "start error" is
the mean |our start − engine start| in frames over the 73 words, "monotonic" the share of text
rows whose argmax frame does not go back, "peak" the mean peak probability. With the anchors:

| head (L, H) | start error (frames) | worst | monotonic | peak |
|---|---|---|---|---|
| 7, 3 | 0.25 | 2 | 0.96 | 0.67 |
| 6, 1 | 0.34 | 2 | 0.96 | 0.69 |
| 4, 1 | 0.36 | 1 | 0.97 | 0.81 |
| 5, 4 | 0.40 | 3 | 0.95 | 0.74 |
| 7, 4 | 0.42 | 2 | 0.96 | 0.69 |
| 7, 1 | 0.52 | 3 | 0.98 | 0.66 |
| 5, 5 | 0.67 | 2 | 0.98 | 0.79 |
| 6, 6 | 0.73 | 4 | 0.96 | 0.60 |
| 1, 1 | 0.75 | 5 | 0.99 | 0.80 |
| 7, 5 | 0.84 | 3 | 0.95 | 0.66 |

The median head is 6 frames off; the worst are 13–16 frames off (L0 H1, L4 H5, L1 H7), some of
them sharp (L1 H7 peaks at 0.92) but not following the speech. Sharpness and monotonicity alone
do not pick good heads; matching the engine does. Combinations of the best heads, in this
ranking: top 1 0.25 (worst 2), top 3 `[(7,3), (6,1), (4,1)]` 0.25 (worst 1), top 4 0.26
(worst 1), top 5 0.30 (worst 1). The fixed set was chosen in the same sweep before the anchors
were added (then it ranked first at width 3: 0.23, worst 1); with the anchors it gives the end
to end result below. Reading the input row instead (row `2 + i`) the best single head is L1 H2
at 0.52 (worst 4) and no set beats 0.49 (worst 4).

Median width, top-3 sets without anchors: width 1 0.27–0.29 (worst 1), **3 0.23 (worst 1)**,
7 0.82–1.04 (worst 4), 11 1.30–1.60 (worst 4–7). Words here last 2–6 frames, so a wide filter
smears neighbouring words together.

The heads are chosen on the same 8 clips the contract test checks, so that bound measures fit.
The held-out extra clips (`test-data/clips-extra/`, never used in the choice) measure
generalisation: over all 5 clips' 70 times (measured in review), 46 equal the engine's frame,
69 are within one frame and one is 2 frames off. The gate's held-out test checks the 3 clips whose 3-beam text is
the engine's (en_names, en_petg, en_cura; de_names and pl_names decode differently and are
skipped): within one frame, at most one time per clip within two.

## End to end (`tests/timestamps.rs`, measured 2026-10-24)

On the 8 contract clips, default options (3 beams): the words equal the engine's word lists, and
of the 146 starts and ends, 111 equal the engine's frame, 22 are one frame early and 13 one frame
late, none further (`contract_clips_words_and_times_match_the_engine_within_a_frame` pins
≤ 1 frame). The
systematic −1 is the first word's start: the engine starts most clips' first word at 0.08 s, we
at the start of the decoded audio.

**The trim and lead-in sensitivity.** With `GateAndTrim` the times are those of decoding the
trimmed samples, plus the samples trimmed off, exactly
(`the_trim_is_added_back_to_every_time_exactly`: 1 s of zeros before a clip, against decoding
the detector's `b0..b1` with the detector off). That is not the same as "1 s later than the
clean clip": the clean clips have speech from their first sample, so their trim is the whole
clip, while on the padded clip the detector's segment starts about 0.12 s into the zeros (its
0.1 s pad before speech plus about 20 ms of frame rounding), and the decode sees that lead-in.
The decode is chaotically sensitive to any lead-in: pl_zosia's tokens flip with 160, 320 or
1280 leading zeros, and 3 of the 8 contract clips decode to different (not worse) tokens with
the padding. en_daniel decodes the same, and its first word starts 0.1 s into the lead-in where
the clean clip has none to start in. So times are exact relative to the audio the decoder saw;
two different lead-ins can give slightly different transcripts and times.

Cost (release, Apple Silicon, best of 5): en_daniel (3.4 s) 466.5 ms off, 521.0 ms on,
+54.5 ms, measured before the capture moved inside `attend` (it then ran a second softmax per
layer, now gone): one full decoder pass over ~14 tokens; the DTW over ~16 × 42 cells is
negligible. Re-measuring after the change on a machine loaded by other builds gave −146 to
+124 ms between runs, too noisy to replace the figure.
