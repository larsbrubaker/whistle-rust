# Decoding: beam search and keyword biasing

The engine documents three behaviours and nothing more: it benchmarks at 5 beams, `keywords`
"favors names, places, product words, and jargon during search", and its CLI reads keywords one
per line. Everything here is designed in this repo from those sentences and measured against the
engine's transcripts used as data (`test-data/engine/`, `test-data/engine-extra/`). The code is
`crates/whistle-rust/src/transcribe/search.rs` and `keywords.rs`.

## Options

- `beams` (default `DEFAULT_BEAMS` = 3, chosen from the measurements below): 1 is exactly the
  greedy contract decode, which a caller gets with `beams: 1`; 2 to `MAX_BEAMS` (16) is a beam
  search; 0 or more than 16 is `Error::Input`.
- `keywords` (default empty): words and phrases to favour; blank entries are ignored.
- `keyword_bonus` (default `DEFAULT_KEYWORD_BONUS` = 3.0 nats per keyword token): must be finite
  and not negative, else `Error::Input`.

## Beam search

- **Language.** Step 0 picks the tag as greedy does (or feeds the forced one), so every
  hypothesis shares one language.
- **Scoring.** Each step log-softmaxes the text ids only (below the first tag, in f64), so the
  tags never take probability mass after step 0. A hypothesis's score is its cumulative
  log-probability plus its keyword adjustments.
- **Expansion.** Each live hypothesis offers its `beams + 1` best text ids (so even with EOS among
  them, `beams` live extensions remain); all candidates are sorted
  by score, ties by token id, then by beam order (deterministic). A candidate ending in EOS is
  finished; the others fill the next `beams` live hypotheses.
- **Stopping.** After each step, the search stops when `beams` hypotheses have finished and
  the best finished one ranks at least as high as the best live one does now; or when none is
  live; or at the length limit (`[BOS, tag, text…]` never exceeds `max_seq_len`). Under the
  plain sum without keywords this is exact, since a live score only falls (a keyword bonus
  can raise it); under the mean it is a heuristic,
  since a live hypothesis's mean can still rise. At such a stop the live hypotheses never
  reached EOS and are dropped. Only at the real length limit are live hypotheses kept, cut
  there as greedy is, ranked by their mean without an EOS term, and only when no hypothesis
  reached EOS: any finished one beats every cut one, so a confident repetition loop running
  to the limit cannot outrank a real ending. (The first version stopped as
  soon as `beams` had finished and kept that step's live extensions as if cut: they always
  outranked the EOS that triggered the stop, so the search returned unfinished hypotheses.
  Stopping on the count alone, with the live ones dropped, made fr_thomas at 2 beams score
  below greedy, so the score test above failed; the score rule passes it on every clip.)
- **Final choice.** The finished hypothesis with the best **mean score per token** (EOS
  counted). Measured against the plain sum on the eight contract clips (normalized corpus
  WER over 72 words against the truth, 73 against the engine; `src/transcribe/tests.rs`
  `ranking_word_error_rates`):

  | beams | truth (sum) | engine (sum) | truth (mean) | engine (mean) |
  |------:|------------:|-------------:|-------------:|--------------:|
  | 1     | 0.1944      | 0.0274       | 0.1944       | 0.0274        |
  | 2     | 0.1667      | 0.0548       | 0.1667       | 0.0274        |
  | 3     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |
  | 5     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |
  | 8     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |

  Both gain two edits over greedy against the truth. The sum fixes fr_thomas ("5000 mètres"
  becomes "5 mm") and keeps pl_zosia's short "do 10 ml."; the mean keeps fr_thomas as greedy
  and the engine have it and recovers pl_zosia's ending ("do 10 milimetrów lewo"). From 3
  beams on, the mean reproduces every engine transcript word for word.
- **State.** Each hypothesis owns a `DecoderState`; it is `Clone`, with the RoPE table and
  the cross-attention memory (fixed for the decode) shared by `Arc`, so a fork copies only the
  per-position self-attention and Engram caches: 8 layers × 2 KV heads × (64 + 64) × 4 bytes ≈
  8 KiB of keys and values per position, 126 KiB at en_daniel's 18 positions, about 2.5 MiB per
  hypothesis at the 320-token limit. The children are chosen first; then each parent's last
  chosen child takes its state without a copy, and only its other children clone it. Prefix
  sharing was not needed: the copy is small next to a step's cost.
- **Batched steps.** At each position every live hypothesis steps together
  (`DecoderState::step_batch`, behind `Scorer::step_batch`): the embedding, the Engram
  projections, every attention, MLP and mHC projection and the head run on all the new rows at
  once, so each weight matrix is decoded once per step instead of once per hypothesis. The
  Engram gather and value conv, the self-attention taps, cache and attention stay per
  hypothesis. Every kernel sums each output row on its own, so the scores, and therefore the
  search, are bit-identical to stepping each hypothesis alone (`search/batch_tests.rs` checks
  every speech clip at 3 and 5 beams). One beam steps alone, as before.
- **Cancellation.** The handle is checked before every decoder step, each beam's at each
  position (all of a position's checks run before its batched step), so the latency stays one
  step.

### Per-beam WER per clip (mean ranking)

Normalized against the true text (raw in brackets) / against the engine:

| clip        | 1 beam                | 2 beams               | 3, 5, 8 beams         |
|-------------|-----------------------|-----------------------|-----------------------|
| en_samantha | 0 (0.182) / 0         | same                  | same                  |
| en_daniel   | 0 (0) / 0             | same                  | same                  |
| de_anna     | 0.143 (0.429) / 0     | same                  | same                  |
| fr_thomas   | 0.273 (0.273) / 0     | same                  | same                  |
| es_monica   | 0 (0.111) / 0         | same                  | same                  |
| it_alice    | 0.125 (0.125) / 0     | same                  | same                  |
| nl_xander   | 0.333 (0.556) / 0     | same                  | same                  |
| pl_zosia    | 0.857 (0.857) / 0.250 | 0.571 (0.714) / 0.250 | 0.571 (0.714) / 0     |

pl_zosia reads "do 10 ml." at one beam, "do 10 milimetru w lewo." at two and "do 10
milimetrów lewo." (the engine's) from three. Under the sum, fr_thomas is 0.091 / 0.182 from
2 beams ("5 mm") and pl_zosia stays at its one-beam text.

The normalizer lower-cases, turns every non-alphanumeric character into a space, and maps the
clips' spoken numbers and units to how the model writes them ("twenty" → 20, "millimeters" →
mm, "Zentimeter" → cm, …; the table is in `tests/decoding/wer.rs`). The raw WER keeps case
and punctuation and splits on spaces, so "twenty millimeters" against "20 mm" counts two
errors there. The gate pins the normalized corpus WER against the truth at one beam (14/72)
and three (12/72) as upper bounds, and checks on fr_thomas and pl_zosia that three beams score
no lower than greedy under the mean; `every_beam_count_scored_no_lower_than_greedy_on_every_clip`
(ignored) checks 2, 3, 5 and 8 beams on every clip. That is a measured result, not a
guarantee: pruning can drop greedy's path.

### Cost

en_daniel (3.3 s, 16 text tokens), release, Apple Silicon, one thread, alone, 2026-10-24, on
the SIMD kernels merged from main (`tests/decoding/timing.rs`):

| beams | ms per call | over greedy |
|------:|------------:|------------:|
| 1     | 207         | —           |
| 2     | 305         | +98         |
| 3     | 402         | +195        |
| 5     | 670         | +463        |
| 8     | 894         | +687        |
| 16    | 1347        | +1140       |
| 5 + 3 keywords | 442 | +235      |

The 5-beam call with keywords measured faster than without (one run against three); the
cause is not yet measured (a likely candidate is the stop rule ending sooner when the
keywords' retractions lower the live scores). Each extra beam costs one decoder step per
position.

**Default: 3 beams.** It is the smallest count that reproduces the engine's transcript on every
contract clip, it has the best WER against the true texts (12/72, against 14/72 greedy), and it
costs about +195 ms on the 3.3 s clip. 5 (the engine's benchmark count) gives the same text on
these clips for more than twice the extra cost. Greedy is `beams: 1`; the contract tests set it
explicitly, since the fixtures pin the greedy decode. `the_default_options_give_the_engine_transcript_on_every_clip`
pins the default's result.

## Keyword biasing

- **Trie.** A token-level prefix trie of every keyword in three spellings (as written, first
  letter upper-cased, all lower-case: the model writes jargon either way, "Matterhackers",
  "cura"), each tokenized twice: after a space (`" Cura"` → `▁C…`, a word inside the text) and
  without one (`"Cura"`; the tokenizer adds no dummy prefix, so this is how the transcript's
  first word comes out, and a word after a hyphen).
- **Bonus.** A token earns `keyword_bonus` when it extends a keyword prefix the hypothesis is
  inside, or starts a keyword at a word boundary: its piece starts with `▁` or a non-letter, or
  the text so far is empty or ends in a non-alphanumeric character (the unspaced variant).
  Byte-fallback pieces count by their byte: an ASCII space or punctuation byte is a boundary;
  a non-ASCII byte is part of a multi-byte character, which in these languages is a letter.
- **Retraction.** The bonus collected for a keyword in progress is subtracted when the
  hypothesis leaves it unfinished, or when a completed keyword is continued by more letters
  ("Cura" + "tor"). It is kept only when the keyword is complete and the next token is a word
  boundary or EOS. A hypothesis cut at the length limit has its pending bonus retracted.
- **One beam.** Greedy cannot take a token back, so retraction alone left one beam stuck after
  following a keyword's first piece ("▁S" of "Siobhan" turned es_monica's "Gira" into "Sira").
  With keywords and one beam, a step where the bonus overturns the model's own pick is
  checkpointed (a state clone). If that keyword is then abandoned unfinished (by an ordinary
  token, by starting another keyword, or by the length limit cutting it), the decode returns
  there and picks again without that token. The retry bound is per position: after 3 tries at
  one checkpoint it takes the unbiased pick there. A per-decode cap of 32 backtracks in all
  then turns biasing off for the rest of the decode, so a long keyword list cannot multiply
  the cost. Completing the checkpointed keyword releases the checkpoint; completing another
  keyword after abandoning it does not. The release comes before the next checkpoint is
  taken, so a token that completes one keyword and starts another by bonus checkpoints the
  new one. Without keywords, one beam is the plain greedy
  argmax.
- **Strength.** Swept on the extra clips (13 keywords) and on the contract clips with the
  irrelevant keywords `["Siobhan", "PETG"]`:

  | bonus | found, 1 / 3 / 5 beams | contract clips changed, 1 / 3 / 5 beams |
  |------:|-----------------------:|----------------------------------------:|
  | 1     | 5 / 6 / 6              | 0 / 0 / 0                               |
  | 2     | 7 / 7 / 8              | 0 / 0 / 0                               |
  | 3     | 9 / 9 / 9              | 0 / 0 / 0                               |
  | 4     | 10 / 10 / 8            | 0 / 0 / 1                               |
  | 5     | 12 / 11 / 10           | 1 / 1 / 1                               |
  | 6     | 12 / 13 / 8            | 2 / 2 / 1                               |

  3.0 is the default: the most found at every beam count while no contract clip changes; 4
  already changes one at 5 beams. Larger bonuses at 5 beams find fewer: biased partial paths
  crowd the beam and are retracted.

### Extra clips (`test-data/clips-extra/`, `oracle/make_extra_clips.py`)

Keywords counted as found when their normalized words appear in order in the normalized text
(case-insensitive). Without biasing ours finds 4 of 13 at one beam and 3 at three or five; the
engine finds 3 without its keyword option and 7 with it; ours with biasing finds 9 at one, three
and five beams, including all 7 the engine finds. Missed by both: Siobhan ("Shivan"), Niamh
("Kneeov"), Pforzheim ("Fort Simps"), Grzegorza: the model's acoustic score for these is too
far from the spelling for a bonus that leaves ordinary words alone. The gate pins 9 of 13 and
every engine-found keyword at one beam and at three
(`cargo test --release --test decoding sweep -- --ignored --nocapture` prints the sweep).
