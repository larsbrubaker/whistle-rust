// Unit tests for the search (`search.rs`) on a scripted model, so each rule can be reached on
// purpose: candidate ordering and `top_k` under NaN, the early stop dropping hypotheses that
// never reached EOS, `beams + 1` candidates per beam, a finished hypothesis beating a cut one at
// the length limit, and the single-beam keyword backtracking (at the length limit, when one
// keyword is abandoned for another, after one completes and the next starts, and its two
// bounds, `MAX_RETRIES` per position and `MAX_BACKTRACKS` per decode), and the repetition
// guard rolling one beam (with and without keywords) and every beam back past a loop, and
// cutting after `MAX_REWINDS`.

use std::cell::RefCell;
use std::rc::Rc;

use super::super::keywords::KeywordTrie;
use super::super::repeat::{Pieces, MAX_REWINDS, MIN_SPAN};
use super::{
    beam_search, decode, greedy_with_keywords, sort_candidates, top_k, Candidate, Ranking, Scorer,
    Search, MAX_BACKTRACKS, MAX_RETRIES,
};
use crate::error::Error;

const EOS: u32 = 1;
const TAG: u32 = 12;
const TEXT_IDS: usize = 12;

/// A model whose next-token scores depend only on the last token fed; every fed prefix is
/// logged (shared across clones).
#[derive(Clone)]
struct Script {
    rows: fn(u32) -> Vec<(u32, f32)>,
    fed: Vec<u32>,
    log: Rc<RefCell<Vec<Vec<u32>>>>,
}

impl Scorer for Script {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        self.fed.push(token);
        self.log.borrow_mut().push(self.fed.clone());
        let mut row = vec![-30.0; TEXT_IDS + 2];
        for (id, s) in (self.rows)(token) {
            row[id as usize] = s;
        }
        Ok(row)
    }
}

fn script(rows: fn(u32) -> Vec<(u32, f32)>) -> Script {
    Script {
        rows,
        fed: Vec::new(),
        log: Rc::new(RefCell::new(Vec::new())),
    }
}

fn search(beams: usize, max_len: usize, trie: Option<&KeywordTrie>) -> Search<'_> {
    Search {
        beams,
        eos: EOS,
        text_ids: TEXT_IDS,
        max_len,
        trie,
        ranking: Ranking::Mean,
        pieces: &NO_PIECES,
    }
}

/// No piece starts a word: the scripted ids carry no text, so every loop is one inside a word.
static NO_PIECES: Pieces = Pieces::NONE;

fn no_check(_: usize) -> Result<(), Error> {
    Ok(())
}

#[test]
fn candidates_sort_totally_with_nan_last_and_ties_by_token_then_beam() {
    let c = |beam, token, score| Candidate { beam, token, score };
    let mut v = vec![
        c(0, 3, f64::NAN),
        c(1, 2, -1.0),
        c(0, 2, -1.0),
        c(0, 5, f64::INFINITY),
        c(1, 1, -1.0),
    ];
    sort_candidates(&mut v);
    let order: Vec<(usize, u32)> = v.iter().map(|c| (c.beam, c.token)).collect();
    assert_eq!(order, [(0, 5), (1, 1), (0, 2), (1, 2), (0, 3)]);
}

#[test]
fn top_k_of_an_all_nan_row_is_the_lowest_ids_in_order() {
    let row = [f32::NAN; 6];
    let ids: Vec<u32> = top_k(&row, None, 3).iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, [0, 1, 2]);
    let mixed = [f32::NAN, 1.0, f32::NEG_INFINITY, 1.0];
    let ids: Vec<u32> = top_k(&mixed, None, 4).iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, [1, 3, 0, 2]);
}

/// After the tag: 2 then 3. After 2: 4 slightly beats EOS. After 3: only EOS. With two beams
/// the second step finishes `[3]` and `[2]` with `[2, 4]` ranked between them, so the search
/// stops with `[2, 4]` live. Counted as cut, it would win the mean on a free EOS term; the
/// finished `[3]` (mean −0.46 against `[2]`'s −0.61) must.
fn early_stop_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, -0.1), (3, -0.5)],
        2 => vec![(4, -0.001), (EOS, -0.01)],
        3 => vec![(EOS, -0.01)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn an_early_stop_returns_only_hypotheses_that_reached_eos() {
    let s = search(2, 20, None);
    let finished = beam_search(&s, 2, script(early_stop_rows), TAG, &mut no_check).unwrap();
    assert!(
        finished.iter().all(|f| f.ended),
        "{:?}",
        finished.iter().map(|f| &f.tokens).collect::<Vec<_>>()
    );
    assert_eq!(finished.len(), 2);
    assert_eq!(
        decode(&s, script(early_stop_rows), TAG, &mut no_check)
            .unwrap()
            .tokens,
        [3]
    );
}

/// After the tag EOS is best, then 2 and 3.
fn eos_first_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(EOS, 0.0), (2, -0.2), (3, -0.3)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn a_beam_whose_best_is_eos_still_offers_beams_live_extensions() {
    let s = search(2, 20, None);
    let model = script(eos_first_rows);
    let log = model.log.clone();
    // Two beams: EOS finishes one, and 2 and 3 both still continue.
    beam_search(&s, 2, model, TAG, &mut no_check).unwrap();
    let second: Vec<Vec<u32>> = log
        .borrow()
        .iter()
        .filter(|p| p.len() == 2)
        .cloned()
        .collect();
    assert_eq!(second, [vec![TAG, 2], vec![TAG, 3]]);
}

/// Tokens: 2 `▁x`, 3 `▁b`, 4 `c`, 5 `▁a`, 6 `d`, 7 `e`, 8 `▁f`, 9 `g`, 10 `▁h`, 11 `i`; EOS
/// and 0 carry no text.
fn edges() -> Vec<(Option<char>, Option<char>)> {
    let p = |f: char, l: char| (Some(f), Some(l));
    vec![
        (None, None),
        (None, None),
        p(' ', 'x'),
        p(' ', 'b'),
        p('c', 'c'),
        p(' ', 'a'),
        p('d', 'd'),
        p('e', 'e'),
        p(' ', 'f'),
        p('g', 'g'),
        p(' ', 'h'),
        p('i', 'i'),
    ]
}

/// Unbiased: `[2]` then EOS. Keyword A `▁a d e` starts one nat behind 2; after `▁a` the model
/// wants `▁b` (keyword B, `▁b c`), which completes after `c`.
fn two_keyword_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0), (5, -1.0)],
        5 => vec![(3, 0.0)],
        3 => vec![(4, 0.0)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn one_beam_backtracks_when_a_keyword_is_abandoned_for_another() {
    let trie = KeywordTrie::build(&[vec![5, 6, 7], vec![3, 4]], &edges(), 3.0);
    let s = search(1, 20, Some(&trie));
    let got = greedy_with_keywords(&s, &trie, script(two_keyword_rows), TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, [2]);
}

/// Keyword A `▁a d e` again, the model following it; the limit cuts it after `▁a d`.
fn long_keyword_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0), (5, -1.0)],
        5 => vec![(6, 0.0)],
        6 => vec![(7, 0.0)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn one_beam_backtracks_when_the_length_limit_cuts_a_keyword() {
    let trie = KeywordTrie::build(&[vec![5, 6, 7]], &edges(), 3.0);
    // `[BOS, tag, text..]` of at most 4: two text tokens.
    let cut = search(1, 4, Some(&trie));
    let got = greedy_with_keywords(&cut, &trie, script(long_keyword_rows), TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, [2]);
    // With room, the keyword completes and is kept.
    let room = search(1, 20, Some(&trie));
    let got = greedy_with_keywords(&room, &trie, script(long_keyword_rows), TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, [5, 6, 7]);
}

/// Keyword A `▁b c` starts one nat behind `▁x` and the model follows it; right after it, B
/// `▁a d e` starts one nat behind `▁x` again, and the model then abandons B.
fn chained_keyword_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0), (3, -1.0)],
        3 => vec![(4, 0.0)],
        4 => vec![(2, 0.0), (5, -1.0)],
        5 => vec![(2, 0.0), (6, -10.0)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn one_beam_backtracks_a_keyword_started_by_the_token_that_completes_the_last() {
    let trie = KeywordTrie::build(&[vec![3, 4], vec![5, 6, 7]], &edges(), 3.0);
    let s = search(1, 20, Some(&trie));
    let got = greedy_with_keywords(&s, &trie, script(chained_keyword_rows), TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, [3, 4, 2]);
}

/// Four keyword starts after the tag, each a little behind `▁x`, each abandoned by the model.
fn four_starts_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0), (3, -1.0), (5, -1.1), (8, -1.2), (10, -1.3)],
        3 | 5 | 8 | 10 => vec![(2, 0.0), (last + 1, -10.0)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn one_beam_tries_at_most_max_retries_keywords_at_one_position() {
    let trie = KeywordTrie::build(
        &[vec![3, 4], vec![5, 6], vec![8, 9], vec![10, 11]],
        &edges(),
        3.0,
    );
    let s = search(1, 20, Some(&trie));
    let model = script(four_starts_rows);
    let log = model.log.clone();
    let got = greedy_with_keywords(&s, &trie, model, TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, [2]);
    // The tag, one step after each tried start, then `▁x`'s step to EOS.
    assert_eq!(MAX_RETRIES, 3);
    assert_eq!(log.borrow().len(), 1 + MAX_RETRIES + 1);
}

/// At every position two keyword starts sit a little behind `▁x`, and the model abandons both.
fn endless_starts_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        5 => vec![(2, 0.0), (6, -10.0)],
        3 => vec![(2, 0.0), (4, -10.0)],
        _ => vec![(2, 0.0), (5, -1.0), (3, -1.5)],
    }
}

#[test]
fn one_beam_backtracks_at_most_max_backtracks_times_per_decode() {
    let trie = KeywordTrie::build(&[vec![5, 6], vec![3, 4]], &edges(), 3.0);
    // 28 text positions: `▁x` at every one stays under the repetition guard's 30-token span,
    // and two abandoned starts at each would be 56 backtracks without the cap.
    let positions = MIN_SPAN - 2;
    let s = search(1, MIN_SPAN, Some(&trie));
    let model = script(endless_starts_rows);
    let log = model.log.clone();
    let got = greedy_with_keywords(&s, &trie, model, TAG, &mut no_check).unwrap();
    // `▁x` at every text position; one extra step per backtrack.
    assert_eq!(got.tokens, vec![2; positions]);
    assert!(!got.looped);
    assert_eq!(MAX_BACKTRACKS, 32);
    assert!(2 * positions > MAX_BACKTRACKS);
    assert_eq!(log.borrow().len(), positions + MAX_BACKTRACKS);
}

/// After the tag, 2 then EOS ends weakly; 3 starts a loop of 3s the model is sure of.
fn loop_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, -0.2), (3, -0.25)],
        2 => vec![(EOS, -2.0), (4, -2.1)],
        3 => vec![(3, 0.0)],
        _ => vec![(EOS, 0.0)],
    }
}

#[test]
fn at_the_length_limit_a_finished_hypothesis_beats_a_cut_one() {
    let s = search(2, 12, None);
    let finished = beam_search(&s, 2, script(loop_rows), TAG, &mut no_check).unwrap();
    // The loop is cut at the limit with a better mean than any real ending.
    let cut = finished
        .iter()
        .find(|f| !f.ended)
        .expect("a cut hypothesis");
    let ended: Vec<_> = finished.iter().filter(|f| f.ended).collect();
    assert!(!ended.is_empty());
    let best_ended = ended
        .iter()
        .max_by(|a, b| a.ranked(Ranking::Mean).total_cmp(&b.ranked(Ranking::Mean)))
        .unwrap();
    assert!(cut.ranked(Ranking::Mean) > best_ended.ranked(Ranking::Mean));
    let got = decode(&s, script(loop_rows), TAG, &mut no_check)
        .unwrap()
        .tokens;
    assert_eq!(got, best_ended.tokens);
}

#[test]
fn a_batch_with_fewer_tokens_than_states_is_an_input_error_before_any_step() {
    let mut a = script(early_stop_rows);
    let mut b = a.clone();
    let err = Scorer::step_batch(&mut [&mut a, &mut b], &[TAG]).unwrap_err();
    assert!(matches!(err, Error::Input(_)), "{err}");
    assert!(a.log.borrow().is_empty());
}

/// After the tag 2, then 3 for ever unless 3 is banned, when EOS (five nats behind) ends it.
fn runaway_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0)],
        _ => vec![(3, 0.0), (EOS, -5.0)],
    }
}

/// After the tag 2; then 3 and 4 each repeat themselves, the other a nat behind, EOS far.
fn two_loops_rows(last: u32) -> Vec<(u32, f32)> {
    match last {
        TAG => vec![(2, 0.0)],
        4 => vec![(4, 0.0), (3, -1.0), (EOS, -9.0)],
        _ => vec![(3, 0.0), (4, -1.0), (EOS, -9.0)],
    }
}

/// The longest prefix `model` was fed (the tag included).
fn deepest(model: &Script) -> usize {
    model.log.borrow().iter().map(Vec::len).max().unwrap_or(0)
}

#[test]
fn one_beam_rolls_back_past_a_loop_and_decodes_on_with_its_next_token_banned() {
    let s = search(1, 100, None);
    let model = script(runaway_rows);
    let got = decode(&s, model.clone(), TAG, &mut no_check).unwrap();
    // 2, then 3 thirty times: the guard fires, rolls back to [2, 3] and bans 3 after it, so
    // the model's next choice, EOS, ends the decode.
    assert_eq!(got.tokens, [2, 3]);
    assert!(got.looped);
    assert_eq!(deepest(&model), 1 + MIN_SPAN);
    // The roll-back feeds the tag and 2 again from the BOS state, then steps 3 to EOS.
    assert_eq!(model.log.borrow().len(), MIN_SPAN + 1 + 2 + 1);
}

#[test]
fn one_beam_is_cut_after_max_rewinds_loops() {
    let s = search(1, 300, None);
    let got = decode(&s, script(two_loops_rows), TAG, &mut no_check).unwrap();
    // Each roll-back bans the loop's token, so the other loop starts; the fourth loop is cut.
    assert_eq!(MAX_REWINDS, 3);
    assert_eq!(got.tokens, [2, 3, 4, 3, 4]);
    assert!(got.looped);
}

#[test]
fn one_beam_with_keywords_rolls_back_too() {
    let trie = KeywordTrie::build(&[vec![8, 9]], &edges(), 3.0);
    let s = search(1, 100, Some(&trie));
    let model = script(runaway_rows);
    let got = greedy_with_keywords(&s, &trie, model.clone(), TAG, &mut no_check).unwrap();
    assert_eq!(got.tokens, [2, 3]);
    assert!(got.looped);
    assert_eq!(model.log.borrow().len(), MIN_SPAN + 1 + 2 + 1);
}

#[test]
fn the_beam_search_rolls_back_each_looping_hypothesis() {
    let s = search(3, 100, None);
    let model = script(runaway_rows);
    let finished = beam_search(&s, 3, model.clone(), TAG, &mut no_check).unwrap();
    // The hypothesis that ran on to thirty 3s was rolled back and ended after [2, 3].
    assert!(finished
        .iter()
        .any(|f| f.tokens == [2, 3] && f.ended && f.guarded));
    // None runs on far past the loop (the deepest is a weak sibling of the looping step):
    // the beam does not chase it to the length limit.
    assert!(deepest(&model) <= 2 + MIN_SPAN, "{}", deepest(&model));
    // A hypothesis that took EOS before the loop was long enough to fire is no loop to the
    // guard and ranks on its score: 2 and twenty-nine 3s, then EOS.
    let got = decode(&s, script(runaway_rows), TAG, &mut no_check).unwrap();
    assert_eq!((got.tokens.len(), got.looped), (MIN_SPAN, false));
    let two = decode(&s, script(two_loops_rows), TAG, &mut no_check).unwrap();
    assert!(two.tokens.len() < 2 * MIN_SPAN, "{:?}", two.tokens);
}
