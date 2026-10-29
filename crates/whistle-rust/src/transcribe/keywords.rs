//! Keyword biasing (`Options::keywords`): a token-level prefix trie of the caller's keywords,
//! and the per-hypothesis matcher that adds `Options::keyword_bonus` to every token that
//! extends a keyword prefix the hypothesis is inside, or that starts a keyword at a word
//! boundary, and retracts the bonus collected for a partial keyword when the hypothesis leaves
//! it unfinished, or a completed one continued by more letters ("Cura" + "tor"). Only a
//! completed keyword followed by a word boundary or EOS keeps its bonus; a hypothesis cut at
//! the length limit has its pending bonus retracted. One beam cannot take a token back, so
//! `search` checkpoints where a bonus overturned the model's pick and returns there when that
//! keyword is abandoned. The measurements are in the decoding test area
//! (`tests/decoding/keywords.rs`).

use std::collections::HashMap;

use crate::tokenizer::{PieceType, Tokenizer};

/// The default [`super::Options::keyword_bonus`], in nats per keyword token: the bonus that
/// found the most keywords on the extra clips at every beam count while leaving every contract
/// clip unchanged by irrelevant keywords; 4 already changed one at 5 beams
/// (`keyword_bonus_sweep` in `tests/decoding/keywords.rs`).
pub const DEFAULT_KEYWORD_BONUS: f32 = 3.0;

/// The first and last character of text id `id`'s piece, `▁` read as a space. A byte piece
/// is its byte when that is ASCII; any other byte is part of a multi-byte character, which in
/// these languages is a letter, so it reads as one. Control and unknown pieces carry no text.
pub(crate) fn piece_edges(tokenizer: &Tokenizer, id: u32) -> (Option<char>, Option<char>) {
    match tokenizer.piece_type(id) {
        Some(PieceType::Byte(b)) => {
            let c = if b.is_ascii() { char::from(b) } else { 'a' };
            (Some(c), Some(c))
        }
        Some(PieceType::Normal | PieceType::UserDefined) => {
            let p = tokenizer.piece(id).unwrap_or("");
            let space = |c: char| if c == '▁' { ' ' } else { c };
            (p.chars().next().map(space), p.chars().last().map(space))
        }
        _ => (None, None),
    }
}

/// One trie node: its children by token id, and whether a keyword ends here.
#[derive(Debug, Default)]
struct Node {
    children: HashMap<u32, usize>,
    terminal: bool,
}

/// The keywords' token sequences as a prefix trie, plus which text ids begin a new word.
#[derive(Debug)]
pub(crate) struct KeywordTrie {
    nodes: Vec<Node>,
    /// Per text id: the piece starts with `▁` or with a character that is not alphanumeric,
    /// so it cannot continue the word before it.
    boundary: Vec<bool>,
    /// Per text id: the piece ends with a character that is not alphanumeric (a hyphen, an
    /// apostrophe, punctuation), so a keyword may start right after it without a space.
    ends_open: Vec<bool>,
    bonus: f64,
}

/// The capitalizations a keyword is matched in: as written, with the first letter upper-cased,
/// and all lower-case (the model writes jargon either way: "matter hackers", "Cura", "cura").
fn spellings(keyword: &str) -> Vec<String> {
    let mut out = vec![keyword.to_string()];
    let mut chars = keyword.chars();
    if let Some(first) = chars.next() {
        out.push(first.to_uppercase().chain(chars).collect());
    }
    out.push(keyword.to_lowercase());
    out.sort();
    out.dedup();
    out
}

impl KeywordTrie {
    /// The trie of `keywords` for `tokenizer`, `None` when there are none (blank entries are
    /// ignored). Each spelling goes in twice, as the tokenizer encodes it: after a space (`▁`
    /// first, how a word inside the text comes out) and without one (how the transcript's
    /// first word comes out, the tokenizer adds no dummy prefix, and a word after a hyphen).
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        keywords: &[String],
        text_ids: usize,
        bonus: f32,
    ) -> Option<KeywordTrie> {
        let keywords: Vec<&str> = keywords
            .iter()
            .map(|k| k.trim())
            .filter(|k| !k.is_empty())
            .collect();
        if keywords.is_empty() {
            return None;
        }
        let edges: Vec<(Option<char>, Option<char>)> = (0..text_ids as u32)
            .map(|id| piece_edges(tokenizer, id))
            .collect();
        let mut sequences = Vec::new();
        for keyword in keywords {
            for spelling in spellings(keyword) {
                for variant in [format!(" {spelling}"), spelling] {
                    let tokens = tokenizer.encode(&variant);
                    if tokens.iter().all(|&t| (t as usize) < text_ids) {
                        sequences.push(tokens);
                    }
                }
            }
        }
        Some(KeywordTrie::build(&sequences, &edges, bonus))
    }

    /// The trie of token `sequences`, given each text id's first and last character (`▁`
    /// counts as a space; `None` for pieces that carry no text).
    pub(crate) fn build(
        sequences: &[Vec<u32>],
        edges: &[(Option<char>, Option<char>)],
        bonus: f32,
    ) -> KeywordTrie {
        let word = |c: &Option<char>| c.is_some_and(char::is_alphanumeric);
        let mut trie = KeywordTrie {
            nodes: vec![Node::default()],
            // A piece that starts with a space or a non-letter cannot continue a word.
            boundary: edges.iter().map(|(f, _)| f.is_some() && !word(f)).collect(),
            // A piece that ends with a non-letter lets a keyword start right after it.
            ends_open: edges.iter().map(|(_, l)| l.is_some() && !word(l)).collect(),
            bonus: f64::from(bonus),
        };
        for tokens in sequences {
            trie.insert(tokens);
        }
        trie
    }

    /// The node count (the root included), for the tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    fn insert(&mut self, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        let mut node = 0;
        for &t in tokens {
            node = match self.nodes[node].children.get(&t) {
                Some(&next) => next,
                None => {
                    self.nodes.push(Node::default());
                    let next = self.nodes.len() - 1;
                    self.nodes[node].children.insert(t, next);
                    next
                }
            };
        }
        self.nodes[node].terminal = true;
    }

    fn is_boundary(&self, token: u32) -> bool {
        self.boundary.get(token as usize).copied().unwrap_or(true)
    }

    /// The score adjustment of every text id for a hypothesis in `m` (what
    /// [`KeywordTrie::advance`] would return for each), computed in one pass: an id outside
    /// the trie's reach abandons the partial keyword (or keeps a completed one when it starts
    /// a new word); only the ids the trie can continue with need the full step.
    pub(crate) fn deltas(&self, m: &Matcher, text_ids: usize, eos: u32) -> Vec<f64> {
        let completed = m.active.iter().any(|&n| self.nodes[n].terminal);
        let mut delta: Vec<f64> = (0..text_ids as u32)
            .map(|t| {
                if completed && (t == eos || self.is_boundary(t)) {
                    0.0
                } else {
                    -m.partial
                }
            })
            .collect();
        let reachable = m.active.iter().chain(std::iter::once(&0));
        for &n in reachable {
            for &t in self.nodes[n].children.keys() {
                if let Some(d) = delta.get_mut(t as usize) {
                    *d = self.advance(m, t, eos).1;
                }
            }
        }
        delta
    }

    /// The matcher after `token`, the score adjustment that token earns (`+bonus` when it
    /// extends or starts a keyword, minus the partial bonus it abandons, nothing for a token
    /// that leaves a completed keyword at a word boundary, EOS counting as one), and whether it
    /// abandons a partial keyword (its bonus is retracted).
    pub(crate) fn advance(&self, m: &Matcher, token: u32, eos: u32) -> (Matcher, f64, bool) {
        let at_boundary = token == eos || self.is_boundary(token);
        let completed = m.active.iter().any(|&n| self.nodes[n].terminal) && at_boundary;
        // A completed keyword followed by a boundary keeps what it collected.
        let mut kept = m.committed;
        let mut partial = m.partial;
        if completed {
            kept += partial;
            partial = 0.0;
        }
        let mut next: Vec<usize> = Vec::new();
        if token != eos {
            for &n in &m.active {
                if let Some(&c) = self.nodes[n].children.get(&token) {
                    next.push(c);
                }
            }
            let may_start = self.is_boundary(token) || m.open;
            if may_start {
                if let Some(&c) = self.nodes[0].children.get(&token) {
                    next.push(c);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        let open = token != eos && self.ends_open.get(token as usize).copied().unwrap_or(false);
        if next.is_empty() {
            // Leaving a partial keyword: retract its bonus.
            let delta = -partial;
            let matcher = Matcher {
                active: next,
                partial: 0.0,
                committed: kept,
                open,
            };
            return (matcher, delta, partial > 0.0);
        }
        let continues = m
            .active
            .iter()
            .any(|&n| self.nodes[n].children.contains_key(&token));
        // A token that starts a fresh keyword without continuing the old one abandons it.
        let (delta, new_partial) = if continues {
            (self.bonus, partial + self.bonus)
        } else {
            (self.bonus - partial, self.bonus)
        };
        let abandoned = !continues && partial > 0.0;
        let matcher = Matcher {
            active: next,
            partial: new_partial,
            committed: kept,
            open,
        };
        (matcher, delta, abandoned)
    }
}

/// Where one hypothesis is in the keyword trie: the nodes it is inside, the bonus collected
/// for the keyword it is part way through (retracted if it is abandoned), the bonus kept for
/// completed keywords, and whether its last token leaves a keyword free to start unspaced.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Matcher {
    active: Vec<usize>,
    partial: f64,
    committed: f64,
    open: bool,
}

impl Matcher {
    /// The state at the start of the text: no keyword in progress, and a keyword may start
    /// without a leading space.
    pub(crate) fn start() -> Matcher {
        Matcher {
            active: Vec::new(),
            partial: 0.0,
            committed: 0.0,
            open: true,
        }
    }

    /// Whether the hypothesis is part way through (or at the end of) a keyword.
    pub(crate) fn inside(&self) -> bool {
        !self.active.is_empty()
    }

    /// The bonus kept for completed keywords.
    pub(crate) fn committed(&self) -> f64 {
        self.committed
    }

    /// The bonus collected for the keyword in progress.
    #[cfg(test)]
    pub(crate) fn partial(&self) -> f64 {
        self.partial
    }
}

#[cfg(test)]
mod tests;
