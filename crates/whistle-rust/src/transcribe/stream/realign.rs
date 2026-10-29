//! Where a stream's committed text ends in a new pass: a character-level semi-global alignment of everything the stream committed in the
//! utterance against a prefix of the pass's words, both normalized (`super::normalize`: lower
//! case, letters and digits, no spaces). `stream.rs` takes the words after the cut as the pass's
//! new words, less those that re-read, in other words, committed words the alignment could not
//! match (`re_read`, by time); the tests in `tests.rs` pin the cases that word-level matching
//! got wrong.
//!
//! Characters, because consecutive passes split, join and time words differently, so neither
//! time nor word-by-word matching says where the committed text ends. Measured failures of both:
//! "matter" committed and the next pass reading "Matterhak" over the same time made a midpoint
//! rule commit "matter Matterhackers"; a prefix match on the last words consumed "Matterhak"
//! whole and lost "hackers"; a one-word match on a short word ("a", "the") reached a later copy;
//! a time fallback committed a re-read word twice ("Creeche Krishtov"); a punctuation-only word
//! ("Bonjour !") was committed twice. When even the best alignment changes more than half the
//! committed characters, the stream restarts the utterance at the commit point instead.
//!
//! The table is |C|·|P| cells, at most about 500 × 500 characters for a 30 s buffer: about
//! 0.5 ms natively.

/// Where the committed text ends in a pass's words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cut {
    /// The first word (index into the pass's words) that is not wholly committed; the number of
    /// words when all are.
    pub word: usize,
    /// Normalized characters of `word` that the committed text already holds: 0 at a word
    /// boundary, otherwise the cut is inside `word` and only the rest of it is new.
    pub offset: usize,
    /// The alignment's edit distance (characters).
    pub cost: usize,
    /// Characters at the end of the committed text that the alignment deleted: no character of
    /// the pass before the cut stands for them (0 when the committed text's last character is
    /// matched or substituted, and 0 when nothing matched: then the pass did not hear the
    /// committed text, so none of its words can be a re-read of it).
    pub unmatched: usize,
}

/// Aligns all of `committed` (normalized) against a prefix of `words` (each normalized; empty
/// for a punctuation-only word) with unit-cost substitutions, insertions and deletions, and
/// returns where the best alignment ends. Ties go first to a word boundary (a cut inside a word
/// at the same cost is the committed text's spelling borrowing letters of the next word, as
/// "make a" against "make the" would), then to the latest position (a re-read of the last
/// committed word, "the" read as "a", is consumed, not committed again). A cut inside a word is
/// kept only when that word's head is exactly the end of the committed text and holds the whole
/// last committed word (`last`, normalized: "matter" in "Matterhackers", "2" in "20"; the "t"
/// of "the" after a committed "about" is no continuation); otherwise it is a poor match spread over a word read
/// differently ("Creeche" against "Krishtov"), and the best boundary is taken instead. A cut
/// that costs no less than consuming nothing (all of `committed` substituted or deleted) is
/// moved to the pass's start: no character matched, so no word of the pass is consumed.
/// Punctuation-only words have no characters to align: after a boundary, up to
/// `committed_marks` of them (the punctuation-only words committed after the committed text's
/// last letter) are skipped as committed, so they are never new twice; any more are new
/// ("Bonjour" committed, "Bonjour ! Ça va" read: the "!" is new).
pub(super) fn cut(committed: &str, last: &str, words: &[String], committed_marks: usize) -> Cut {
    let c: Vec<char> = committed.chars().collect();
    // Each character of the pass with the word it is in and its offset in that word.
    let mut p: Vec<(char, usize, usize)> = Vec::new();
    for (w, word) in words.iter().enumerate() {
        p.extend(word.chars().enumerate().map(|(o, ch)| (ch, w, o)));
    }
    let m = p.len();
    // The last row of the edit-distance table, two rows at a time: `row[j]` aligns all of `c`
    // seen so far against `p[..j]`, the pass's start fixed and its suffix free.
    // `tail[j]` counts the committed characters the path to `row[j]` deleted last, in a row:
    // of equal-cost moves the one deleting least is taken, so `unmatched` never overstates.
    let mut row: Vec<usize> = (0..=m).collect();
    let mut next = vec![0; m + 1];
    let mut tail = vec![0; m + 1];
    let mut next_tail = vec![0; m + 1];
    for (i, &ch) in c.iter().enumerate() {
        next[0] = i + 1;
        next_tail[0] = i + 1;
        for j in 1..=m {
            let substitute = (row[j - 1] + usize::from(p[j - 1].0 != ch), 0);
            let insert = (next[j - 1] + 1, next_tail[j - 1]);
            let delete = (row[j] + 1, tail[j] + 1);
            (next[j], next_tail[j]) = substitute.min(insert).min(delete);
        }
        std::mem::swap(&mut row, &mut next);
        std::mem::swap(&mut tail, &mut next_tail);
    }
    let boundary = |j: usize| j == 0 || j == m || p[j].2 == 0;
    let mut best = 0;
    let mut best_boundary = 0;
    for j in 1..=m {
        let (cost, best_cost) = (row[j], row[best]);
        if cost < best_cost || (cost == best_cost && (boundary(j) || !boundary(best))) {
            best = j;
        }
        if boundary(j) && row[j] <= row[best_boundary] {
            best_boundary = j;
        }
    }
    if !boundary(best) {
        let offset = p[best].2;
        let head: String = p[best - offset..best].iter().map(|x| x.0).collect();
        if last.is_empty() || !committed.ends_with(&head) || !head.ends_with(last) {
            best = best_boundary;
        }
    }
    // Consuming pass characters must buy something: an alignment that costs as much as
    // consuming nothing matched no character of the committed text (the committed words lie
    // outside what the pass heard, "edges" against "Please export"), so nothing of the pass is
    // committed. A doubled word is better than a lost one.
    if row[best] >= row[0] {
        best = 0;
    }
    let cost = row[best];
    if best == 0 {
        // Nothing of the pass is committed: every word is new, punctuation first included, and
        // none is taken for a re-read by its time (a short new word right after the floor of a
        // settled restart has its midpoint before the committed end).
        return Cut {
            word: 0,
            offset: 0,
            cost,
            unmatched: 0,
        };
    }
    let unmatched = tail[best];
    if boundary(best) {
        // After the word holding the last aligned character, past the committed marks.
        let mut word = p[best - 1].1 + 1;
        let mut skipped = 0;
        while word < words.len() && words[word].is_empty() && skipped < committed_marks {
            word += 1;
            skipped += 1;
        }
        return Cut {
            word,
            offset: 0,
            cost,
            unmatched,
        };
    }
    let (_, word, offset) = p[best];
    Cut {
        word,
        offset,
        cost,
        unmatched,
    }
}

/// How many of the pass's words after a boundary cut re-read committed words the alignment
/// left unmatched: `committed` is the committed words the pass was aligned against, each as
/// its normalized length and its end (seconds), `unmatched` the cut's [`Cut::unmatched`], and
/// `pass` the pass's words from the cut on, each as its start and end. A committed word the
/// pass read with other characters ("mm" then "millimeters", "Q20" then "queue 20", "20" then
/// "twenty") costs less to delete than to align against the longer reading, so the cut lands
/// before the re-read and its words would be committed again. Time tells them apart: the
/// committed words whose characters were all deleted end at some time, and a word of the pass
/// whose midpoint lies before that time, and that ends no more than [`RE_READ_OVERRUN`] after
/// it, was said there, so it is that word read again, not a new one. Words are taken in order
/// and the first one that fails ends the run, so a new word is never taken for a re-read by a
/// later one's timing.
pub(super) fn re_read(committed: &[(usize, f32)], unmatched: usize, pass: &[(f32, f32)]) -> usize {
    // The committed words wholly deleted, from the end: the last lettered one's end bounds the
    // re-read. A word only partly deleted was partly heard before the cut, so it says nothing;
    // punctuation-only words have no characters to delete, so they never bound it (a "." after
    // a partly deleted "edges" would otherwise reach past "edge" into the next word).
    let mut left = unmatched;
    let mut end = None;
    for &(len, word_end) in committed.iter().rev() {
        if left == 0 || len > left {
            break;
        }
        left -= len;
        if len > 0 {
            end = Some(end.map_or(word_end, |e: f32| e.max(word_end)));
        }
    }
    let Some(end) = end else {
        return 0;
    };
    pass.iter()
        .take_while(|&&(start, word_end)| {
            (start + word_end) / 2.0 < end && word_end <= end + RE_READ_OVERRUN
        })
        .count()
}

/// How far (0.1 s) past the end of the committed words it re-reads a pass word may run: two
/// passes time the same word within a few hundredths of a second (the re-reads measured end at
/// most 0.06 s past), while a new word timed over a committed word the pass left out runs on
/// well past its end ("it" committed to 0.7 s, then a new "cubes" read over 0.45–0.9 s).
const RE_READ_OVERRUN: f32 = 0.1;

/// Whether a pass that spells `heard` characters says nothing about `committed` characters:
/// under half as many (an empty pass, or a fragment), it cannot hold the committed text, and
/// aligning against it would only measure its absence.
pub(super) fn says_nothing(committed: usize, heard: usize) -> bool {
    committed > 0 && heard * 2 < committed
}

/// `word` without its first `offset` normalized characters (letters and digits), so the rest
/// of a word the cut is inside keeps its own spelling, case and punctuation.
pub(super) fn rest_of(word: &str, offset: usize) -> &str {
    let mut seen = 0;
    for (i, ch) in word.char_indices() {
        if ch.is_alphanumeric() {
            if seen == offset {
                return &word[i..];
            }
            seen += ch.to_lowercase().count();
        }
    }
    ""
}
