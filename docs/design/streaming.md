# Streaming (`whistle::Stream`)

Live audio, transcribed as it arrives. The engine documents its streaming as "live audio streams
indefinitely and commits words that two passes agree on", printing per update the committed
words (times from stream start), the unconfirmed tail, the language and the seconds received.
That behaviour is designed here (contract item 2); the engine's `--audio-stream` output on the
contract clips (`test-data/engine-stream/`, `oracle/engine_stream_transcripts.py`) is only a
black-box comparison.

## The API

`Whistle::stream(&Options)` checks the options as `transcribe` does, plus
`Options::stream_step` (seconds of new audio between passes, `DEFAULT_STREAM_STEP` = 1 s,
0.1–10 s), and returns a `Stream<'a>`, which borrows the `Whistle`: a long-lived app owns the
model alongside the stream (both in one struct, or the model in a `static`/`Arc` the stream
outlives). `Options::streaming()` is the live preset: the defaults with `beams: 1` (below).

`push(&[f32])` takes 16 kHz mono chunks of any size and runs every pass the chunk completes
before it returns, so a push of many seconds blocks for many passes; live callers push small
chunks (a step or less) as they arrive. `finish(self)` commits everything left. Each returns a
`StreamUpdate`: `committed` (the words new in this update), `text` (them joined), `pending` (the
latest pass's unconfirmed words), `language`, `dropped` (seconds dropped at the cap, below) and `continues` (its `text` begins
with the rest of the word before it: append it with no space).
`transcript()` is everything committed so far as a `Transcript`, also after an error (an update
that failed is not returned, so its words reach the caller only there); `buffered()` the samples
held (never over `MAX_SAMPLES`); `dropped()` the total dropped. `Options::cancel` is checked at
each push, before every pass and inside every decode (`Error::Cancelled`). Words are always
aligned: agreement and the cuts need times.

Stream positions are `u64` samples (a `usize` would overflow on wasm32 after 74.6 hours, and a
stream runs indefinitely); each pass decodes at offset 0 and moves its word times by the
stream position. Word times are f32 seconds like every `Word`: exact to the sample for the first
17 minutes, to about 30 ms at 74 hours.

**Language.** Both `StreamUpdate::language` and `transcript().language` follow one rule: the
forced language; else the one fixed by the first pass that committed at least 3 words
(`long::MIN_LANGUAGE_WORDS`, as windowing fixes it from the first window that keeps 3 words),
forced on every pass after it; else the first commit's;
else the latest pass's detection. Each is `None` while it has no words to label (the update
counts pending words, the transcript only committed ones), as `transcribe` gives `None` for an
empty transcript.

## One pass

A push splits its audio at `stream_step` boundaries counted from the start of the stream and
runs one pass at each, and makes room at the cap for a whole step at its first sample, so the
passes, and therefore the transcript, do not depend on the chunk size (pinned at 0.25, 1 and
3 s, and at 0.37 s, which divides no step, on 33 s that hits the force rule). A pass:

1. Runs the detector over the buffer and keeps the segments ending more than 0.1 s past the
   commit point (shorter slivers come and go as the detector re-measures a grown buffer).
   None: nothing is said after the commit point, so the pending words are committed (the pause
   ended their utterance), the commit point moves to 0.5 s before the buffer's end (an onset
   the detector cannot confirm yet stays), and no decode runs. Silence never decodes.
2. Decodes from the start of the first kept segment, the utterance the commit point is in, to
   the end of the last (`GateAndTrim`; to the buffer's end with `Gate`), with `decode_window`.
   The buffer keeps 1 s before the utterance: the detector's noise floor looks 1 s each side of
   a frame.
3. Finds where the committed text ends in the pass (**Aligning a pass**, below); the pass's
   words after that point are new.
4. If the last speech ended at least 1 s before the buffer's end (an endpoint) or the stream is
   finishing, commits every new word. Otherwise commits the longest prefix of the new words on
   which this pass and the previous one (its new words after its own alignment) agree, compared lower-cased with letters and digits only (passes differ
   in the punctuation and capitals of a word whose sentence end they have or have not heard
   yet), and only words the previous pass heard at least 0.3 s past (a word the buffer's end
   cuts can read as another word in two passes alike: de_anna's "den" read as "in" twice,
   measured). The newer pass's text is committed; the rest is `pending`.

## Aligning a pass

Consecutive passes split, join and time words differently, so neither time nor word-by-word
matching says where the committed text ends in a new pass. Measured failures of both: on the
45 s stream "matter" [11.50–11.66] was committed and the next pass read "Matterhak" over
11.50–12.16, so a midpoint rule committed "matter Matterhackers"; a prefix match on the last
words consumed "Matterhak" whole and lost "hackers"; a one-word match on a short word ("a",
"the") reached a later copy; the time fallback committed a re-read word twice ("Creeche
Krishtov"); a punctuation-only word ("Bonjour !") was committed twice.

So `realign::cut` aligns characters. C is the committed text of the utterance: the committed
words whose midpoint lies after the pass's start (a word's end can run on to the next word's
start across a pause, so the end is no test), normalized as agreement compares words and
joined without spaces. P is the pass's words, normalized the same way, each character knowing
its word and offset. An edit-distance table (match 0, substitution, insertion and deletion 1;
C wholly consumed from P's start, P's suffix free) gives the cost of every end position in P.
The cut is the cheapest; ties go to a word boundary, then to the latest position, so "make a"
against "make the cube" cuts before "the" (a mid-word cut there only borrows a letter), and
"on the" re-read as "on a" consumes the "a". Punctuation-only words have no characters, so a
boundary skips them with the word before: they are never new twice.

A cut inside a word is kept only when C ends with exactly that word's head ("matter" in
"Matterhackers", "2" in "20", "the" in "they"); otherwise the best boundary is taken (a poor
match spread over a re-read word, "Creeche" against "Krishtov", would leave a fragment). The
rest of the word becomes the first new word, its text recovered from the original spelling
(case and punctuation kept), its tokens from the one holding the first new character, and its
time the rest's share of the word's; it is **glued** to the committed word before it in every
text (the transcript, an update's `text`, `pending`), so the transcript reads "matterhackers"
or "20", never "matter hackers" or "2 0". An update's `text` can therefore begin with the rest
of a word: `StreamUpdate::continues` is then true, and a caller appending updates adds no space
before it. The head must be exactly the end of the committed text and hold the whole last
committed word: the "t" of "the" after a committed "about" is no continuation (it glued
"abouthe" on the 33 s no-pause stream).

**Tokens.** Committed tokens are never retracted, so when a continued word's cut lands inside a
token, that token is committed twice ("2" then "▁20" decodes to "2 20"; "the" then "y" to "the
they"). A stream's `Transcript::tokens` is the concatenation of each pass's committed tokens,
as a windowed transcript's is of each window's kept tokens, and neither need decode exactly to
`text`: `text` and `words` are authoritative.

**Punctuation-only words** have no characters to align. After a boundary the cut skips as many
of them as the committed text ends with (the committed marks) and no more, so "Bonjour !"
committed and then read again commits no second "!", while "Bonjour" committed without its "!"
still gets the "!" from the next pass.

**The lead-in.** From a pass's start at a pause, the committed text it aligns against is the
words whose midpoint lies after that start. From a floor inside speech (a forced cut, a restart,
every Off pass) it is every committed word that ends after the floor, so the alignment absorbs
the fragment of the word the lead-in hears instead of committing it again.

**A pass that says nothing.** A pass that spells fewer than half as many characters as are
committed (it decoded nothing, or a fragment) carries no information: nothing commits, pending
stays, the floor does not move and there is no restart (an empty pass would otherwise cost all
of the committed text and restart). If the stream or the utterance ends there, what is pending
commits.

**Consuming must match something.** A cut that costs no less than consuming nothing (every
committed character substituted or deleted) moves to the pass's start: no character of the
committed text matched, so the pass did not hear it, and nothing of the pass is consumed. A
settled restart's pass (which takes its cut whatever it costs) would otherwise lose a clean
word on a tie: committed "edges" against a pass reading "Please export the bracket" costs 5
for consuming "Please" and 5 for consuming nothing, and the latest-position rule dropped
"Please". Re-reads where some characters match ("on the" read as "on a") are still absorbed. A
possibly doubled article is better than a lost word.

**Accepted:** "make a" committed, then read "make the cube…", commits "make a the cube": no
letter repeats, but a word ("a" re-read as "the") does, since the committed "a" is final.

When the best alignment changes more than half of C's characters, the pass read the utterance
too differently to place the cut: the utterance restarts at the commit point, and the pass runs
again from 0.1 s (the detector's padding) before it; that run, and every later pass from the
same floor, takes its cut whatever it costs. Remembering the restart's floor matters: a short
committed word (under 0.2 s: "the", "a", "of") ending after it is in C, the pass hears only its
last 0.1 s and misreads it, and restarting again on every pass cleared pending each time, so
agreement never committed (on the 45 s Off stream nothing from 23 s to 45 s, the buffer at
29.5 s and 2.3 s updates, then the force rule committed 28 s at once). The
alignment is |C|·|P| cells, at most about 500 × 500 characters for a 30 s buffer: 0.5 ms native.

**The limitation.** A committed word is final. When a later pass reads it differently, the
committed text keeps the earlier reading and continues from the aligned point (the alignment
absorbs the difference in its cost), so the stream can read a word otherwise than one longer
pass over the same audio would. The invariant the
tests pin is that the committed text is within a small character edit distance of the full
offline decode's (normalized): 0 on every contract clip, 7 on the 18 s stream, 42 on the 45 s
one, 25 and 49 on 33 s with no pauses at 10 s and 1 s steps, 18 and 64 on 18 s and 45 s with
the detector off; with no word committed twice. With the detector off the buffer stays at most
4.7 s (18 s) and 9.2 s (45 s), with at most 2.0 s and 5.0 s of audio between commits.

Committed words are appended with the windowed decode's rules (start moved up to the previous
end, so times are monotonic), never retracted, and the commit point is the last committed word's
end after that adjustment.

**Why decode from the utterance start, not the commit point.** The first design trimmed the
buffer to the last committed word, as a plain local-agreement policy does. Decoding from inside
a sentence wrecked the rest of it: de_anna streamed as "Bitte macht in Wilfitzweit Centimeta
Huya" (word error rate 0.71 against the offline decode; en_daniel, en_samantha and nl_xander
also differed). Decoding from the pause before the utterance, the way windowing cuts in pauses,
gives the decoder the context the offline decode has: all 8 contract clips stream to exactly the
offline text. The cost is that a pass decodes the whole utterance so far, at most 30 s.

**The 30 s cap.** Before a step would take the buffer past 30 s, the latest pass's words that
end at least 1 s (`OVERLAP_SAMPLES`) before the audio it decoded are force-committed, and later
passes start 1 s before that commit point, as windowing overlaps a hard cut, so the word the cut
is in is whole in the pass and the alignment finds it committed (a 0.1 s lead-in started the
pass inside a committed word and committed "of" twice). Audio that still does not fit is
dropped uncommitted and reported in `StreamUpdate::dropped`: it happens only when the passes
decoded no words for audio the detector calls speech (noise it lets through, mumbling), so there
was nothing to commit to make room. 33 s of English with no pauses keeps at least 0.57 of every
clip's words and doubles none; 45 s with 0.5–1 s pauses keeps at least 0.71 (the offline windowed
decode 0.57), with no word committed twice.

**`VadMode::Off`.** Without the detector there are no pauses: no endpoint, no utterance start
to decode from, and silence is decoded (and may be hallucinated, as offline). After each commit
the next pass starts 1 s before the commit point and the buffer is trimmed there, so an Off
stream holds a few seconds (9.2 s at most on 45 s of speech) instead of climbing to 30 s and
re-decoding all of it every step (2.5–3.9 s per update before this). Long Off streams rely on
agreement and the force rule alone, and read worse: each pass starts inside a word, so 18 s is
18 characters from the offline decode.

`finish` commits the latest pass's pending words directly when no audio arrived since it (that
pass saw everything), and otherwise runs a final pass that commits all.

**Early commits.** Agreement commits before the end on 5 of the 8 contract clips; of the clips
of 2.5 s or more, es_monica (2.63 s) commits nothing early, as the engine's own stream does
not, and pl_zosia (2.51 s) has its first word read differently by its 1 s and 2 s passes.
Pinned, with the median commit latency (1.26 s; 1.31 s before the deterministic math; bound
1.46 s).

## Measured

Against the engine's stream (1 s chunks; ours at 1 beam, the engine at its default 3): the final committed text matches on 7 of 8
clips (pl_zosia, word error rate 0.375, against the engine's "10 milimetrów lewo.": our greedy
decode reads its end otherwise, offline as streamed; the engine runs 3 beams), mean word error
rate 0.047 (0.031 before the deterministic math moved pl_zosia's greedy decode). Commit latency
(seconds received when a word commits minus its end) has a median of 1.26 s, the engine's
1.15 s.

Per-update compute, 1 s steps, one thread, release (M5, another build running):

| | native, 1 beam | native, 3 beams | wasm (Node, simd128), 1 beam | wasm, 3 beams |
|---|---|---|---|---|
| 3 s clip, median / max | 107 / 158 ms | 204 / 265 ms | 172 / 289 ms | 404 / 605 ms |
| 45 s stream, median / max | 225 / 446 ms | 413 / 965 ms | 361 / 954 ms | 556 / 1456 ms |

(`cargo run --release --example stream -- models/whistle.cact <wav> 1 <beams>`, and the same
example built for wasm32-wasip1 under `scripts/wasm-bench.mjs`.) The max is a pass over a long
utterance near the cap. **Beams.** A stream decodes with `Options::beams` like any call; at the
default 3 a one-thread wasm update can exceed the 1 s step (1456 ms) and fall behind, at 1 beam
it stays under it (954 ms), and its stream already matches the engine's 3-beam stream on 7 of the
8 clips. So `Options::streaming()`, the live preset, is the defaults with `beams: 1`.
