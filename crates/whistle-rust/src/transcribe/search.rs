//! The text decode after the language tag: a beam search over cloned decoder states
//! (`model::DecoderState`, behind the [`Scorer`] trait so the unit tests can drive it with a
//! scripted model), with optional keyword biasing (`keywords`). One beam is exactly the greedy
//! decode `whistle_finetune._score_quantised` defines (each step's pick is the argmax of the
//! raw text scores, ties to the lowest id), plus the keyword adjustment when there is one.
//! Every hypothesis is checked after each step by the repetition guard (`repeat`), which stops
//! one that has run into a loop.
//!
//! Step 0 (the language tag) is the caller's, so every hypothesis shares one language. Each
//! step log-softmaxes the text ids only, in f64, so the tags take no probability mass after
//! step 0; a hypothesis ranks by its cumulative log-probability plus its keyword adjustments
//! during the search, and the finished ones by their mean score per token at the end
//! ([`Ranking`]). Forking is cheap enough that the search needs no prefix sharing: a hypothesis
//! owns a `DecoderState` whose per-position self-attention and Engram caches are about 8 KiB
//! (8 layers × 2 KV heads × (64 + 64) f32), about 2.5 MiB at the 320-token limit, while the RoPE
//! table and the cross-attention memory are shared. Each parent's last chosen child takes its
//! state without a copy; only its other children clone it. `ranking_word_error_rates`
//! (`src/transcribe/tests.rs`) and the decoding test area (`tests/decoding.rs`) hold the
//! measurements.

use crate::error::Error;
use crate::model::DecoderState;

use super::keywords::{KeywordTrie, Matcher};
use super::repeat::{Guard, Pieces, Verdict};

/// A decoder that can be forked: feed one token, get the next token's scores.
pub(crate) trait Scorer: Clone {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, Error>;

    /// [`Scorer::step`] on each of `states` with its token, scores in order. The default steps
    /// them one at a time; a model that can share work across them overrides it, and must give
    /// each the scores its own `step` would. A count mismatch is [`Error::Input`] before any
    /// step. The default cannot check the others' inputs first, so after an error part way the
    /// earlier states have stepped; the search abandons the decode on any error, so it never
    /// reuses them.
    fn step_batch(states: &mut [&mut Self], tokens: &[u32]) -> Result<Vec<Vec<f32>>, Error> {
        check_batch(states.len(), tokens.len())?;
        states
            .iter_mut()
            .zip(tokens)
            .map(|(s, &t)| s.step(t))
            .collect()
    }
}

/// One token per state, and later one row of scores per state: anything else is
/// [`Error::Input`], so a short batch can never be zipped silently against the hypotheses.
fn check_batch(states: usize, rows: usize) -> Result<(), Error> {
    if states != rows {
        return Err(Error::Input(format!(
            "a decoder batch of {states} states with {rows} tokens or score rows"
        )));
    }
    Ok(())
}

impl<'w> Scorer for DecoderState<'w> {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        DecoderState::step(self, token)
    }

    fn step_batch(
        states: &mut [&mut DecoderState<'w>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, Error> {
        DecoderState::step_batch(states, tokens)
    }
}

/// What the search needs from the call: the token ids and limits.
pub(crate) struct Search<'a> {
    pub beams: usize,
    pub eos: u32,
    /// Ids below this are text.
    pub text_ids: usize,
    /// The longest sequence, BOS and tag included.
    pub max_len: usize,
    pub trie: Option<&'a KeywordTrie>,
    pub ranking: Ranking,
    /// Which text ids start a word, for the repetition guard.
    pub pieces: &'a Pieces,
}

/// How the finished hypotheses are ranked for the final choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ranking {
    /// The mean score per token (the default). Against the plain sum on the contract clips
    /// both gain two edits over greedy against the true texts, but the mean keeps fr_thomas as
    /// greedy and the engine read it and recovers pl_zosia's ending, and from 3 beams on it
    /// reproduces every engine transcript word for word, where the sum does not.
    Mean,
    /// The plain sum, kept for the measurement that compares it.
    #[cfg(test)]
    Sum,
}

/// One hypothesis: its state (fed up to but not including `feed`), its text tokens, its score
/// (and the score after each token, for a roll-back), its keyword matcher and its repetition
/// guard.
#[derive(Clone)]
struct Hypothesis<S> {
    state: S,
    feed: u32,
    tokens: Vec<u32>,
    /// Model log-probability plus the keyword adjustments so far: what the search ranks by.
    score: f64,
    history: Vec<f64>,
    matcher: Matcher,
    guard: Guard,
}

impl<S> Hypothesis<S> {
    fn start(state: S, tag: u32) -> Hypothesis<S> {
        Hypothesis {
            state,
            feed: tag,
            tokens: Vec::new(),
            score: 0.0,
            history: Vec::new(),
            matcher: Matcher::start(),
            guard: Guard::default(),
        }
    }

    /// Appends `token` with the score after it.
    fn push(&mut self, token: u32, score: f64) {
        self.tokens.push(token);
        self.history.push(score);
        self.feed = token;
        self.score = score;
    }
}

/// A finished hypothesis: ended by EOS, or cut at the length limit or by the repetition guard
/// (`repeat`).
pub(crate) struct Finished {
    pub tokens: Vec<u32>,
    pub score: f64,
    pub ended: bool,
    /// The guard rolled this hypothesis back or cut it at least once.
    pub guarded: bool,
}

impl Finished {
    /// The final ranking: the mean score per token, EOS counted when there is one. The plain
    /// sum favours short hypotheses (`ranking_word_error_rates` measures both).
    pub(crate) fn ranked(&self, ranking: Ranking) -> f64 {
        match ranking {
            Ranking::Mean => {
                let n = self.tokens.len() + usize::from(self.ended);
                self.score / n.max(1) as f64
            }
            #[cfg(test)]
            Ranking::Sum => self.score,
        }
    }
}

/// A NaN score ranks below every number, so a corrupt archive cannot pick one over a real
/// score (the greedy argmax's rule).
fn finite(v: f64) -> f64 {
    if v.is_nan() {
        f64::NEG_INFINITY
    } else {
        v
    }
}

/// `log Σ exp` over `scores`, in f64.
fn log_sum_exp(scores: &[f32]) -> f64 {
    let max = scores
        .iter()
        .map(|&s| finite(f64::from(s)))
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = scores
        .iter()
        .map(|&s| crate::math::exp(finite(f64::from(s)) - max))
        .sum();
    max + crate::math::ln(sum)
}

/// The `k` best text ids of one row by raw score plus keyword adjustment, best first, ties to
/// the lower id (NaN ranks lowest). Ranking by the raw score (not the log-softmax) keeps one
/// beam bit-identical to the greedy argmax: the softmax's normalizer is the same for every id.
pub(crate) fn top_k(scores: &[f32], delta: Option<&[f64]>, k: usize) -> Vec<(u32, f64)> {
    let mut best: Vec<(u32, f64)> = Vec::with_capacity(k + 1);
    for (id, &s) in scores.iter().enumerate() {
        let v = finite(f64::from(s) + delta.map_or(0.0, |d| d[id]));
        if best.len() == k && best.last().is_some_and(|&(_, w)| v.total_cmp(&w).is_le()) {
            continue;
        }
        // Insert after every entry at least as good, so equal values keep id order.
        let at = best.partition_point(|&(_, w)| w.total_cmp(&v).is_ge());
        best.insert(at, (id as u32, v));
        best.truncate(k);
    }
    best
}

/// A candidate extension: which hypothesis, which token, and the new score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    pub beam: usize,
    pub token: u32,
    pub score: f64,
}

/// Best first (NaN as −∞, a total order); ties by token id, then by beam order.
pub(crate) fn sort_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|a, b| {
        finite(b.score)
            .total_cmp(&finite(a.score))
            .then(a.token.cmp(&b.token))
            .then(a.beam.cmp(&b.beam))
    });
}

/// What a decode chose: its text tokens, and whether the repetition guard rolled it back past a
/// loop or cut it at one at least once (`repeat`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Decode {
    pub tokens: Vec<u32>,
    pub looped: bool,
}

/// Decodes the text after the tag. `state` has fed BOS; the first step feeds `tag`. `check`
/// runs before every decoder step (each beam's, at its position; the cancellation point).
pub(crate) fn decode<S: Scorer>(
    search: &Search<'_>,
    state: S,
    tag: u32,
    check: &mut dyn FnMut(usize) -> Result<(), Error>,
) -> Result<Decode, Error> {
    let beams = search.beams.max(1);
    if let (1, Some(trie)) = (beams, search.trie) {
        return greedy_with_keywords(search, trie, state, tag, check);
    }
    let finished = beam_search(search, beams, state, tag, check)?;
    // A hypothesis that reached EOS beats any cut at the length limit, so a confident
    // repetition loop cannot outrank a real ending; cut ones count only if none finished.
    let any_ended = finished.iter().any(|f| f.ended);
    let mut best: Option<&Finished> = None;
    for f in finished.iter().filter(|f| f.ended == any_ended) {
        if best.is_none_or(|b| {
            finite(f.ranked(search.ranking))
                .total_cmp(&finite(b.ranked(search.ranking)))
                .is_gt()
        }) {
            best = Some(f);
        }
    }
    Ok(best.map_or_else(Decode::default, |f| Decode {
        tokens: f.tokens.clone(),
        looped: f.guarded,
    }))
}

/// The beam search proper; returns every finished hypothesis in the order it finished.
pub(crate) fn beam_search<S: Scorer>(
    search: &Search<'_>,
    beams: usize,
    state: S,
    tag: u32,
    check: &mut dyn FnMut(usize) -> Result<(), Error>,
) -> Result<Vec<Finished>, Error> {
    // The state with only BOS fed, from which a rolled-back hypothesis is fed again.
    let root = state.clone();
    let mut live = vec![Hypothesis::start(state, tag)];
    let mut finished: Vec<Finished> = Vec::new();
    // The sequence `[BOS, tag, text..]` never exceeds `max_len`: a hypothesis with
    // `max_len - 2` text tokens is cut there, as the greedy decode is, its pending keyword
    // bonus retracted. Hypotheses can differ in length once one has been rolled back.
    let limit = search.max_len.saturating_sub(2);
    loop {
        let (full, open): (Vec<_>, Vec<_>) =
            live.into_iter().partition(|h| h.tokens.len() >= limit);
        finished.extend(full.into_iter().map(|h| cut(search, h)));
        live = open;
        if live.is_empty() {
            return Ok(finished);
        }
        let mut candidates = Vec::new();
        // One check per hypothesis at its position (1 feeds the tag), as when each stepped
        // alone; then every hypothesis steps together, so the decoder reads its weights once
        // for all of them.
        for h in &live {
            check(h.tokens.len() + 1)?;
        }
        let feeds: Vec<u32> = live.iter().map(|h| h.feed).collect();
        let mut states: Vec<&mut S> = live.iter_mut().map(|h| &mut h.state).collect();
        let rows = S::step_batch(&mut states, &feeds)?;
        check_batch(live.len(), rows.len())?;
        for (b, (hyp, scores)) in live.iter().zip(rows).enumerate() {
            let text = &scores[..search.text_ids.min(scores.len())];
            let delta = search
                .trie
                .map(|t| t.deltas(&hyp.matcher, text.len(), search.eos));
            let lse = if beams > 1 { log_sum_exp(text) } else { 0.0 };
            let banned = hyp.guard.banned(hyp.tokens.len());
            // `beams + 1` per hypothesis: even if one of them is EOS, `beams` live
            // extensions remain, enough for the global best `beams`. Tokens the guard banned
            // here are skipped.
            let best = top_k(text, delta.as_deref(), beams + 1 + banned.len());
            let allowed = best.into_iter().filter(|(t, _)| !banned.contains(t));
            for (token, ranked) in allowed.take(beams + 1) {
                let lp = finite(f64::from(text[token as usize])) - lse;
                let adjust = delta.as_ref().map_or(0.0, |d| d[token as usize]);
                candidates.push(Candidate {
                    beam: b,
                    token,
                    score: if beams > 1 {
                        hyp.score + lp + adjust
                    } else {
                        ranked
                    },
                });
            }
        }
        sort_candidates(&mut candidates);
        // Choose first: EOS candidates finish, the best `beams` others continue.
        let mut chosen: Vec<Candidate> = Vec::with_capacity(beams);
        for c in candidates {
            if chosen.len() == beams {
                break;
            }
            if c.token == search.eos {
                // The candidate's score already holds EOS's keyword adjustment.
                finished.push(Finished {
                    tokens: live[c.beam].tokens.clone(),
                    score: c.score,
                    ended: true,
                    guarded: live[c.beam].guard.fired(),
                });
            } else {
                chosen.push(c);
            }
        }
        // Stop once `beams` hypotheses have reached EOS and the best of them ranks at least as
        // high as the best live one does now. Under the sum without keywords that is exact (a
        // live score only falls; a keyword bonus can raise it); under the mean it is a
        // heuristic, since a live mean can still rise. The live
        // ones never reached EOS, so they are dropped, not counted as cut. Loops the guard
        // cut do not count: they never reached EOS either.
        let ranked = |f: &Finished| finite(f.ranked(search.ranking));
        let ended = finished.iter().filter(|f| f.ended);
        let best_done = ended.clone().map(ranked).fold(f64::NEG_INFINITY, f64::max);
        let best_live = chosen
            .iter()
            .map(|c| {
                ranked(&Finished {
                    tokens: vec![0; live[c.beam].tokens.len() + 1],
                    score: c.score,
                    ended: false,
                    guarded: false,
                })
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if chosen.is_empty() || (ended.count() >= beams && best_done >= best_live) {
            return Ok(finished);
        }
        // Then build: each parent's last chosen child takes its state, earlier ones clone it.
        let mut remaining = vec![0usize; live.len()];
        for c in &chosen {
            remaining[c.beam] += 1;
        }
        let mut parents: Vec<Option<Hypothesis<S>>> = live.into_iter().map(Some).collect();
        let mut next = Vec::with_capacity(chosen.len());
        for c in chosen {
            remaining[c.beam] -= 1;
            let parent = if remaining[c.beam] == 0 {
                parents[c.beam].take()
            } else {
                parents[c.beam].clone()
            };
            let Some(mut hyp) = parent else { continue };
            if let Some(t) = search.trie {
                hyp.matcher = t.advance(&hyp.matcher, c.token, search.eos).0;
            }
            hyp.push(c.token, c.score);
            match hyp.guard.check(&hyp.tokens, search.pieces) {
                None => next.push(hyp),
                // The token made a loop: back to the end of its first copy, with the token
                // that started the second banned there, to read on past it.
                Some(Verdict::Rewind(keep)) => {
                    roll_back(search, &mut hyp, keep);
                    hyp.feed = hyp.tokens.last().copied().unwrap_or(tag);
                    hyp.state = refeed(&root, tag, &hyp.tokens, check)?;
                    next.push(hyp);
                }
                // Out of roll-backs: cut at the first copy, as at the length limit.
                Some(Verdict::Cut(keep)) => {
                    roll_back(search, &mut hyp, keep);
                    finished.push(cut(search, hyp));
                }
            }
        }
        if next.is_empty() {
            return Ok(finished);
        }
        live = next;
    }
}

/// Rolls `hyp`'s tokens, score and keyword matcher back to its first `keep` tokens (its state
/// is the caller's to rebuild).
fn roll_back<S>(search: &Search<'_>, hyp: &mut Hypothesis<S>, keep: usize) {
    hyp.tokens.truncate(keep);
    hyp.history.truncate(keep);
    hyp.score = hyp.history.last().copied().unwrap_or(0.0);
    if let Some(t) = search.trie {
        hyp.matcher = hyp.tokens.iter().fold(Matcher::start(), |m, &token| {
            t.advance(&m, token, search.eos).0
        });
    }
}

/// A state with `[BOS, tag, tokens..]` fed but the last: `root` (BOS fed) fed `tag` and every
/// token of `tokens` but the last, which is the hypothesis's next feed (as is `tag` when
/// `tokens` is empty). Rebuilding from the root costs one decoder step per kept token and
/// needs no snapshot of every position; `check` runs before each step, at its position.
fn refeed<S: Scorer>(
    root: &S,
    tag: u32,
    tokens: &[u32],
    check: &mut dyn FnMut(usize) -> Result<(), Error>,
) -> Result<S, Error> {
    let mut state = root.clone();
    let feeds = std::iter::once(tag).chain(tokens.iter().copied());
    for (i, token) in feeds.take(tokens.len()).enumerate() {
        check(i + 1)?;
        state.step(token)?;
    }
    Ok(state)
}

/// `hyp` finished without EOS (at the length limit, or cut by the guard): its pending keyword
/// bonus is retracted.
fn cut<S>(search: &Search<'_>, hyp: Hypothesis<S>) -> Finished {
    let pending = search
        .trie
        .map_or(0.0, |t| t.advance(&hyp.matcher, search.eos, search.eos).1);
    Finished {
        guarded: hyp.guard.fired(),
        tokens: hyp.tokens,
        score: hyp.score + pending,
        ended: false,
    }
}

/// How many alternatives one beam tries at a single position where a keyword it started was
/// then abandoned, before it takes the unbiased pick there.
pub(crate) const MAX_RETRIES: usize = 3;

/// How many times one decode may go back to a checkpoint in all; after that it takes only
/// unbiased picks, so a long keyword list cannot multiply the cost.
pub(crate) const MAX_BACKTRACKS: usize = 32;

/// The argmax of `scores` plus `delta`, skipping `banned`, ties to the lower id.
fn pick(scores: &[f32], delta: Option<&[f64]>, banned: &[u32]) -> u32 {
    let mut best: Option<(u32, f64)> = None;
    for (id, &s) in scores.iter().enumerate() {
        if banned.contains(&(id as u32)) {
            continue;
        }
        let v = finite(f64::from(s) + delta.map_or(0.0, |d| d[id]));
        if best.is_none_or(|(_, b)| v.total_cmp(&b).is_gt()) {
            best = Some((id as u32, v));
        }
    }
    best.map_or(0, |(id, _)| id)
}

/// Where a single-beam decode let a keyword bonus overturn the model's own pick: the
/// hypothesis after that position's step, its scores, and the biased picks already tried.
struct Checkpoint<S> {
    hyp: Hypothesis<S>,
    scores: Vec<f32>,
    pos: usize,
    banned: Vec<u32>,
}

/// One beam with keywords. A greedy decode cannot take a token back, so retraction alone
/// would leave a hypothesis that followed a keyword's first piece ("▁S" of "Siobhan") into an
/// ordinary word stuck with it. Instead, where the bonus changes the pick, the state is
/// checkpointed; if that keyword is then abandoned unfinished (by an ordinary token, by
/// starting another keyword, or by the length limit), the decode returns there and picks
/// again without that token (after [`MAX_RETRIES`] at one position, or [`MAX_BACKTRACKS`] in
/// the decode, the unbiased pick). Completing the checkpointed keyword releases it. A loop
/// (`repeat`) ends the decode where the guard finds it, as in the beam search.
pub(crate) fn greedy_with_keywords<S: Scorer>(
    search: &Search<'_>,
    trie: &KeywordTrie,
    state: S,
    tag: u32,
    check: &mut dyn FnMut(usize) -> Result<(), Error>,
) -> Result<Decode, Error> {
    let root = state.clone();
    let mut hyp = Hypothesis::start(state, tag);
    let mut checkpoint: Option<Checkpoint<S>> = None;
    let mut resume: Option<(Vec<f32>, Vec<u32>)> = None;
    let mut backtracks = 0;
    let limit = search.max_len.saturating_sub(1);
    let mut pos = 1;
    loop {
        if pos >= limit && resume.is_none() {
            // Cut at the length limit inside the checkpointed keyword: that abandons it too.
            let pending = trie.advance(&hyp.matcher, search.eos, search.eos).2;
            match checkpoint.take() {
                Some(c) if pending && backtracks < MAX_BACKTRACKS => {
                    backtracks += 1;
                    let tried = hyp.tokens.get(c.hyp.tokens.len()).copied().unwrap_or(0);
                    let mut banned = c.banned;
                    banned.push(tried);
                    hyp = c.hyp;
                    pos = c.pos;
                    resume = Some((c.scores, banned));
                    continue;
                }
                _ => break,
            }
        }
        let (scores, banned) = match resume.take() {
            Some(r) => r,
            None => {
                check(pos)?;
                let scores = hyp.state.step(hyp.feed)?;
                (
                    scores[..search.text_ids.min(scores.len())].to_vec(),
                    Vec::new(),
                )
            }
        };
        let biased = backtracks < MAX_BACKTRACKS && banned.len() < MAX_RETRIES;
        let delta = biased.then(|| trie.deltas(&hyp.matcher, scores.len(), search.eos));
        // The repetition guard's bans hold for the model's own pick too.
        let guarded = hyp.guard.banned(hyp.tokens.len());
        let unbiased = pick(&scores, None, &guarded);
        let token = if biased {
            pick(&scores, delta.as_deref(), &[&banned[..], &guarded].concat())
        } else {
            unbiased
        };
        let (matcher, _, abandoned) = trie.advance(&hyp.matcher, token, search.eos);
        if abandoned && backtracks < MAX_BACKTRACKS {
            if let Some(c) = checkpoint.take() {
                // Abandoned: go back to where the bonus took over and try without it.
                backtracks += 1;
                let tried = hyp.tokens.get(c.hyp.tokens.len()).copied().unwrap_or(token);
                let mut banned = c.banned;
                banned.push(tried);
                hyp = c.hyp;
                pos = c.pos;
                resume = Some((c.scores, banned));
                continue;
            }
        }
        // Released when the checkpointed keyword completes, or when nothing is in progress.
        if matcher.committed() > hyp.matcher.committed() && !abandoned || !matcher.inside() {
            checkpoint = None;
        }
        // Then a pick the bonus decided is checkpointed, after the release: the token that
        // completes one keyword may itself start the next by bonus.
        if token != unbiased && checkpoint.is_none() {
            checkpoint = Some(Checkpoint {
                hyp: hyp.clone(),
                scores: scores.clone(),
                pos,
                banned: banned.clone(),
            });
        }
        hyp.matcher = matcher;
        if token == search.eos {
            break;
        }
        hyp.push(token, 0.0);
        match hyp.guard.check(&hyp.tokens, search.pieces) {
            None => pos += 1,
            // A loop: back to the end of its first copy, its second copy's first token banned
            // there, as the beam search does. A checkpoint inside the dropped text goes with it.
            Some(Verdict::Rewind(keep)) => {
                roll_back(search, &mut hyp, keep);
                hyp.feed = hyp.tokens.last().copied().unwrap_or(tag);
                hyp.state = refeed(&root, tag, &hyp.tokens, check)?;
                if checkpoint
                    .as_ref()
                    .is_some_and(|c| c.hyp.tokens.len() >= keep)
                {
                    checkpoint = None;
                }
                pos = keep + 1;
            }
            Some(Verdict::Cut(keep)) => {
                hyp.tokens.truncate(keep);
                break;
            }
        }
    }
    Ok(Decode {
        looped: hyp.guard.fired(),
        tokens: hyp.tokens,
    })
}

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod tests;
