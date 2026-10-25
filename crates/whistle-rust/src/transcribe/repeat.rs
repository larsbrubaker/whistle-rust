//! The repetition guard (`docs/design/decoding.md` "Repetition guard"): a decode on audio the
//! model cannot read (garbled, noise, a language it was not trained on) can lock into
//! repeating one token or phrase, confidently, until the length limit ("je trouve que" 59
//! times; "B" then "ib" 317 times). `search` checks every hypothesis after each step with its
//! [`Guard`]: when the tail has become a loop ([`looped`]) the hypothesis is rolled back to the
//! end of the loop's first copy and the token that started the second copy is banned there, so
//! the decode can pick up the audio after it; after [`MAX_REWINDS`] roll-backs it is cut there
//! instead. Token-level and exact, so it costs a few comparisons per step and is the same on
//! every target.

use crate::tokenizer::Tokenizer;

use super::keywords::piece_edges;

/// Consecutive copies of one block the tail must hold before it counts as a loop: 5, so a
/// phrase said four times over ("I'm sorry, I'm sorry, …") is still speech.
pub(crate) const MIN_COPIES: usize = 5;

/// Tokens the repetition must span before it counts as a loop: 30. A one-token block fires at
/// its 30th copy at the soonest, a two-token block at its 15th, three at the 10th.
pub(crate) const MIN_SPAN: usize = 30;

/// Words (pieces that start with a space) the repetition must hold before it counts as a loop,
/// when it holds any: 20. People repeat a short word many times ("Ja, ja, …" 16 times, "Sì" 12,
/// "ha ha" 16), and in Polish or Italian one word can be five or six tokens, so a token span
/// alone cut real speech ("Więcej" six times); twenty words of one block is no longer speech.
/// A repetition inside one word ("Bibibib…") holds no word start and needs only the token span.
pub(crate) const MIN_WORDS: usize = 20;

/// How many times one hypothesis is rolled back past a loop before the guard cuts it there.
pub(crate) const MAX_REWINDS: usize = 3;

/// What the guard needs to know of each text id's piece.
#[derive(Debug, Clone, Default)]
pub(crate) struct Pieces {
    /// The piece starts with a space (`▁`): it starts a word.
    word: Vec<bool>,
    /// The piece starts with a space or a character that is not alphanumeric: it cannot
    /// continue the word before it, so a cut before it leaves that word whole.
    boundary: Vec<bool>,
}

impl Pieces {
    /// No piece starts a word or a boundary (the scripted tests' models carry no text).
    #[cfg(test)]
    pub(crate) const NONE: Pieces = Pieces {
        word: Vec::new(),
        boundary: Vec::new(),
    };

    /// The table for `tokenizer`'s first `text_ids` ids.
    pub(crate) fn new(tokenizer: &Tokenizer, text_ids: usize) -> Pieces {
        let firsts: Vec<Option<char>> = (0..text_ids as u32)
            .map(|id| piece_edges(tokenizer, id).0)
            .collect();
        Pieces::from_firsts(&firsts)
    }

    /// The table from each id's first character (`▁` read as a space; `None` for a piece with
    /// no text).
    pub(crate) fn from_firsts(firsts: &[Option<char>]) -> Pieces {
        Pieces {
            word: firsts.iter().map(|f| *f == Some(' ')).collect(),
            boundary: firsts
                .iter()
                .map(|f| f.is_some_and(|c| !c.is_alphanumeric()))
                .collect(),
        }
    }

    fn word(&self, id: u32) -> bool {
        self.word.get(id as usize).copied().unwrap_or(false)
    }

    fn boundary(&self, id: u32) -> bool {
        self.boundary.get(id as usize).copied().unwrap_or(false)
    }
}

/// If `tokens` ends in a loop, how many of them to keep: everything before the loop and the
/// loop's first copy. The loop is the shortest block of `n` tokens whose repetition ends the
/// sequence and spans at least [`MIN_COPIES`] blocks and [`MIN_SPAN`] tokens, and, unless it
/// holds no word start at all, [`MIN_WORDS`] words. It is measured as the run of tokens equal
/// to the one `n` before, back from the end, so a run that starts part way into a block
/// ("trouve que je" after "je") is found from its earliest token. When the token after the copy
/// would continue a word (the first copy capitalised, the later ones not: "Więcej, w|ięcej"),
/// the cut moves back to the last word boundary inside the copy, so the kept text never ends in
/// the head of a word.
pub(crate) fn looped(tokens: &[u32], pieces: &Pieces) -> Option<usize> {
    let len = tokens.len();
    for n in 1..=len / MIN_COPIES {
        // `run`: how many of the last tokens each equal the token `n` before them.
        let run = (n..len)
            .rev()
            .take_while(|&i| tokens[i] == tokens[i - n])
            .count();
        let span = run + n;
        if span < (MIN_COPIES * n).max(MIN_SPAN) {
            continue;
        }
        let words = tokens[len - span..]
            .iter()
            .filter(|&&t| pieces.word(t))
            .count();
        if words > 0 && words < MIN_WORDS {
            continue;
        }
        let keep = len - run;
        if pieces.boundary(tokens[keep]) {
            return Some(keep);
        }
        let inside = (keep + 1 - n..keep)
            .rev()
            .find(|&j| pieces.boundary(tokens[j]));
        return Some(inside.unwrap_or(keep));
    }
    None
}

/// What the guard decided about a hypothesis that has just run into a loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Roll back to this many tokens; the token after them is now banned there.
    Rewind(usize),
    /// Out of roll-backs: cut the hypothesis to this many tokens.
    Cut(usize),
}

/// One hypothesis's guard: the tokens banned at each text position by earlier roll-backs, how
/// many roll-backs it has had, and whether it ever fired. Forked with the hypothesis.
#[derive(Debug, Clone, Default)]
pub(crate) struct Guard {
    bans: Vec<(usize, u32)>,
    rewinds: usize,
    fired: bool,
}

impl Guard {
    /// The tokens banned as the text token at position `at` (0 is the first).
    pub(crate) fn banned(&self, at: usize) -> Vec<u32> {
        self.bans
            .iter()
            .filter(|&&(p, _)| p == at)
            .map(|&(_, t)| t)
            .collect()
    }

    /// The guard rolled back or cut this hypothesis at least once.
    pub(crate) fn fired(&self) -> bool {
        self.fired
    }

    /// Checks `tokens`, just extended by one, for a loop. On a roll-back the token that would
    /// start the loop's second copy is banned at that position (bans after it belonged to the
    /// text being dropped and go).
    pub(crate) fn check(&mut self, tokens: &[u32], pieces: &Pieces) -> Option<Verdict> {
        let keep = looped(tokens, pieces)?;
        self.fired = true;
        if self.rewinds == MAX_REWINDS {
            return Some(Verdict::Cut(keep));
        }
        self.rewinds += 1;
        self.bans.retain(|&(p, _)| p <= keep);
        self.bans.push((keep, tokens[keep]));
        Some(Verdict::Rewind(keep))
    }
}

#[cfg(test)]
mod tests;
