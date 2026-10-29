# Windowing past 30 s (`whistle::plan_windows`)

The engine documents "any length: one pass up to 30 seconds, windowed beyond that". Nothing else
about how it windows is known, so this is designed here. Whistle's encoder was trained on at most
30 s (480,000 samples at 16 kHz, `MAX_SAMPLES`), so a call splits longer audio into windows of
at most 30 s, decodes each alone, and stitches the results. The sanity cap stays: over
`MAX_INPUT_SAMPLES` (10 minutes) is `Error::Input`, checked before resampling, in every VAD mode.

## When a call windows

The span a single pass would decode is the trimmed span `first.start .. last.end` with
`GateAndTrim`, the whole clip with `Gate` or `Off`. If that span is at most 30 s it is one
window and the call is exactly the single pass it always was (same features, same decode, same
words; `transcribe/contract.rs` still pins every clip's tokens to the fixtures, and
`windowing/limits.rs` pins that every contract clip plans to one window over that span). Only
longer spans are windowed.

## Where the cuts go (`transcribe/window.rs`)

**With the detector on (`Gate`, `GateAndTrim`)**, the windows are packed from its speech
segments. From a window's start (its 30 s limit counted from there), every pause between two
segments that starts after the window's start and before its limit is a candidate cut. The cut
is the pause's middle, or the limit when the middle lies past it: a pause that crosses the
limit is still a pause, and cutting in it beats a hard cut inside speech. The window is cut in
the longest candidate, the later one on a tie. Candidates whose cut leaves the window at least
half full (≥ 15 s) are preferred over any earlier pause, so one long pause near the start does
not produce a run of short windows. Cutting in a pause means no word is split and no overlap is
needed. Segments are always more than 0.3 s apart (`docs/design/vad.md`), so every candidate
pause is a real one.

The next window starts where the pause ends, so its 30 s go to speech and the rest of a long
pause is never decoded. The one exception is `Gate` with a pause cut in its middle, below.

- `GateAndTrim` decodes each window's speech only: from its first segment's start to its last
  segment's end inside the window, as a single pass trims. The pause audio between windows is
  never decoded, so silence between windows produces nothing.
- `Gate` keeps its meaning, "decode everything the gate passes", as far as the windows can
  hold it: windows tile the clip, each half of a pause cut in its middle going to the window
  beside it; of a pause that crosses the limit, the window before it decodes up to the limit and
  the rest is skipped.

An earlier version took only pauses whose middle fit before the limit: a pause longer than
the window's remaining room forced a hard cut, the next window started inside the same pause
and skipped it again, so the second hard cut fell inside speech that would have fitted in one
window (and a window decoded 15 s of silence). `window.rs`'s
`a_pause_crossing_the_limit_is_cut_at_the_limit_and_skipped` and `windowing/pauses.rs` pin the
fix.

**Hard cuts.** When no pause fits (one segment runs more than 30 s, talking without a break), the
window is cut inside speech, at the middle of the lowest-energy 80 ms frame (one encoder frame)
in the 4 s before the latest possible cut: the quietest moment is the likeliest gap between
words or a stop closure. Each side of a hard cut at `c` decodes 1 s past it
(`OVERLAP_SAMPLES`): the earlier window ends at `c + 1 s`, the later one starts at `c − 1 s`,
so the latest cut is 29 s after the window's start and every window stays at most 30 s. A word
split by the cut is whole in at least one window (almost every word is under a second). Each
window then keeps only its own words: the earlier the words whose midpoint `(start + end) / 2`
is before `c`, the later those at or after `c`; their tokens go with them (`align::word_groups`
gives each word's tokens) and the window's text is decoded from the kept tokens. A word both
windows heard has one midpoint on one side of `c` in each window's timing (the two timings agree
to about a frame), so it is kept once; a word only one window heard whole is kept by that one
when its midpoint is on that window's side.

Keeping by midpoint needs the words' times, so a window with a hard cut is always aligned (one
extra decoder pass, `crates/whistle-rust/src/transcribe/align.rs`); with `word_timestamps` off its words are
dropped after the stitch. `text` and `tokens` are therefore the same with timings on and off,
as for a single pass.

**With the detector off (`Off`)**: no detector runs, so every cut is a hard cut at a fixed place,
29 s after the window's start (the latest), with the same overlap and midpoint rule: windows
decode `0..30 s`, `28..58 s`, `56..86 s`, and so on.

`plan_windows(samples, vad)` returns the windows a call will decode (`Window { decode,
keep_from, keep_until }`, sample ranges of the input; `keep_*` are the hard cuts): callers can
show or check the cuts, and the tests use them to build exact expectations.

## Language

The language of the first window whose kept text has at least 3 words (`MIN_LANGUAGE_WORDS`)
is detected (unless `Options::language` forces it) and then forced on every later window, so
the transcript is in one language with one `language` code, as a caller reading
`Transcript::language` expects. Windows before it (with no text, or a fragment of 1–2 words:
"Yes.", a name, a hallucinated "Thank you." on a noise burst) are decoded with detection and do
not fix it: too little speech to trust its tag for the whole recording. If no window reaches 3
words, the transcript is labelled with the first window with any text. Words rather than
seconds: a window's word count is known after its decode without word times, and is what the
tag was decided from. `windowing/language.rs` pins that a 2-word German fragment before a long
pause and English leaves the English window detected and the transcript English. A
recording that switches language part way is decoded in the first language: forcing a tag
does not translate, so a German part forced to English comes out as the decoder's English
reading of German speech (`windowing/language.rs` pins exactly that, and that the same window
alone detects German).
Per-window detection is not offered: `Transcript` has one `language`, and a caller that needs
per-part languages can decode the windows `plan_windows` returns one by one.

## Stitching (`transcribe/long.rs`)

- **Text**: each window's text, trimmed, joined with single spaces, except before a window that
  starts with closing punctuation (`.,;:!?…)]}»%`: a hard cut can leave the "." that ends the
  previous window's sentence); empty windows add nothing.
- **Tokens**: the windows' kept text tokens, concatenated. They may not decode exactly to the
  text: a window's first token is decoded without the space the stitch adds before it
  ("profile.Ask"), so `text` and `words` are authoritative (`Transcript::tokens`).
- **Words**: each window's words are timed in the input (the window's start is the alignment's
  offset, as the trim offset is for one pass). Within a window they are contiguous, as for a
  single pass; across windows they are in order and never overlap: a word whose start falls
  before the previous word's end (possible only at a hard cut, where two windows time the
  words near the cut independently) starts at that end instead. The pauses between
  `GateAndTrim` windows hold no words.
- **Cancel** is checked before each window (`Stage::Window`) as well as at every check inside it.
- **Empty**: no window with text gives the empty transcript (language `None`).
- **Alignment**: the windowed path always captures `ALIGNMENT_HEADS` and returns no
  alignment pass (the passes of several windows do not add up to one); the head measurements
  that need the pass run on single windows.

## Measured (2026-10-24, Apple Silicon, release, machine shared with a benchmark run)

Designing the tests showed one limit of the model: a 30 s window holding the same two
sentences repeated 4–5 times decodes only about two thirds of them (the decoder skips exact
repeats, and the word times then stretch over the skipped speech). With five different
sentences cycled, so a window holds each at most twice, 10–30 s windows with and without pauses
decode every sentence through the window's end. The tests therefore build long inputs from the
five English clips in turn; real recordings do not repeat themselves this way, but a long
recording of the same phrase over and over would lose repeats.

The ~68 s input of `windowing/measure.rs` (the five English clips in turn, 18 clips, with
0.5/0.75/1 s pauses) plans to 3 windows of 22.10, 22.49 and 21.34 s (each cut in a 1 s pause
after the 15 s mark). Per call, default `GateAndTrim`:

| beams | word times | ms per call (two runs) |
| --- | --- | --- |
| 1 | off | 3076, 2238 |
| 1 | on | 3876, 3634 |
| 3 | off | 4379, 4529 |
| 3 | on | 5450, 7143 |

(Two runs on a machine shared with other builds and benchmarks, so the spread is load; the
planner fix left these windows unchanged.)

About 33–45 ms per second of audio at one beam; most of it is the encoder on 22 s windows and the
~100-token decode per window (`docs/design/performance.md`). Rerun with
`cargo test --release -p whistle-rust --test windowing long_input_cost -- --ignored --nocapture`.

## Tests (`crates/whistle-rust/tests/windowing/`, about 6.4 s for the area under nextest)

- `pauses`: a 45 s input (the English clips in turn, 0.5/0.75/1 s pauses) is cut only in
  pauses, every window ≤ 30 s, every clip inside a window's decoded range, and the transcript
  is exactly each window's samples decoded alone (text joined, tokens concatenated); with word
  times on, each word equals its window's standalone word offset by the window's start (within
  one frame), starts within two frames of a clip (the window's frame grid does not start where
  the clip does, plus the alignment's one-frame precision), and the words are in order and
  never overlap. Pauses of 16 s (its middle fits) and 25.5 s (it crosses the limit; speech at
  0–17.5 and 42.7–60.3 s, the review's repro) between two runs of speech: two windows, no
  hard cut, every clip inside a window. In both, a content check that does not depend on the
  windows: each clip decoded on its own is the reference, one
  longest common subsequence against the transcript (`tests/decoding/wer.rs` normalization)
  counted per clip, and every clip keeps at least 0.5 of its words (measured minimum 0.57: a
  clip's names come out differently beside its neighbours than alone; a skipped clip keeps
  almost none; the bound guards against skipped clips, not single lost words). The per-clip
  reference is not an exact one: the decode depends on the lead-in and the context around the
  speech (`docs/design/vad.md`).
- `hard_cut`: two inputs, each with one hard cut. About 35 s of the English clips with no gaps
  and the detector off (the fixed cut at 29 s); and the clips' speech alone (each clip's
  detected span less its pads) joined into one 33 s run the detector sees as one segment, so
  `GateAndTrim` cuts at its quietest frame (26.0 s). Windows ≤ 30 s, text and tokens the same with
  timings on and off, words in order; and around the cut, against what each window heard
  there on its own: every word both windows heard (their longest common subsequence) is in the
  stitch, the stitch has between the smaller and the larger count, and it keeps at least 0.8 of
  the better window's own words (measured 0.86 and 0.83, where the two windows read the cut's
  last word differently; one more word lost would be 0.71 at most).
- `limits`: every contract clip is one window over the single-pass span; 30 s is one window and
  one more sample two; over 10 minutes is `Error::Input` in every mode; 45 s of silence with
  the gate is empty.
- `language`: English then German (cut at the 2 s pause between them) is labelled English, its
  German window decoded exactly as that window alone with English forced (alone it detects
  German); a 2-word German fragment, a 30 s pause and English: the language is not fixed by the
  fragment and the transcript is English.
- `src/transcribe/window.rs` unit tests: the cut choice, a pause crossing the limit, the hard
  cut and the fixed cuts; `src/transcribe/long.rs`: the join; `src/transcribe/tests.rs`: a
  cancel between windows stops before the next one.
