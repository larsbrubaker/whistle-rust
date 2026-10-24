//! Whistle's SentencePiece BPE tokenizer, read from the archive's RAW attachment and matching
//! upstream's `export.parse_tokenizer_blob` and `RefTokenizer` exactly: user-defined markers
//! (longest first) split the text, each span is merged score-greedily (highest-score adjacent
//! pair, ties to the leftmost), unmatched symbols fall back to `<0xXX>` byte pieces, and spaces
//! are the `▁` meta symbol. Reads its blob through `cact`.

use std::collections::HashMap;

use crate::cact::{Archive, DType};
use crate::error::Error;

const META_SPACE: char = '\u{2581}';
const HEADER_BYTES: usize = 4 * 5 + 4;
const RECORD_BYTES: usize = 4 + 1 + 2;

/// A piece's role (`export.TK_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    /// A `<0xXX>` byte-fallback piece carrying its byte.
    Byte(u8),
}

/// The parsed tokenizer: pieces in id order with their scores and types.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    types: Vec<PieceType>,
    pad_id: u32,
    eos_id: u32,
    bos_id: u32,
    unk_id: u32,
    add_dummy_prefix: bool,
    byte_fallback: bool,
    /// Surface to id; a repeated surface maps to its last id, as the reference's dict does.
    piece_ids: HashMap<String, u32>,
    byte_ids: [u32; 256],
    /// User-defined pieces as chars, longest first (stable for equal lengths).
    markers: Vec<(Vec<char>, u32)>,
}

fn malformed(message: impl Into<String>) -> Error {
    Error::Tokenizer(message.into())
}

fn byte_of(piece: &str) -> Option<u8> {
    // `<0xXX>`: the reference reads the two hex digits at characters 3..5.
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
}

impl Tokenizer {
    /// Parses the tokenizer attachment (`export.parse_tokenizer_blob` layout).
    pub fn from_blob(blob: &[u8]) -> Result<Tokenizer, Error> {
        if blob.len() < HEADER_BYTES {
            return Err(malformed("shorter than its 24-byte header"));
        }
        let u32_at =
            |at: usize| u32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]]);
        let count = u32_at(0) as usize;
        let (pad_id, eos_id, bos_id, unk_id) = (u32_at(4), u32_at(8), u32_at(12), u32_at(16));
        let add_dummy_prefix = blob[20] != 0;
        let byte_fallback = blob[21] != 0;
        let mut pieces = Vec::with_capacity(count.min(blob.len() / RECORD_BYTES));
        let mut scores = Vec::with_capacity(pieces.capacity());
        let mut types = Vec::with_capacity(pieces.capacity());
        let mut at = HEADER_BYTES;
        for id in 0..count {
            let record = blob
                .get(at..at + RECORD_BYTES)
                .ok_or_else(|| malformed(format!("truncated at piece {id} of {count}")))?;
            let score = f32::from_le_bytes([record[0], record[1], record[2], record[3]]);
            let kind = record[4];
            let len = usize::from(u16::from_le_bytes([record[5], record[6]]));
            at += RECORD_BYTES;
            let surface = blob
                .get(at..at + len)
                .ok_or_else(|| malformed(format!("piece {id} runs past the end")))?;
            at += len;
            let surface = std::str::from_utf8(surface)
                .map_err(|_| malformed(format!("piece {id} is not UTF-8")))?;
            let kind = match kind {
                0 => PieceType::Normal,
                1 => PieceType::Unknown,
                2 => PieceType::Control,
                3 => PieceType::UserDefined,
                4 => PieceType::Byte(byte_of(surface).ok_or_else(|| {
                    malformed(format!("byte piece {id} is {surface:?}, not <0xXX>"))
                })?),
                other => return Err(malformed(format!("piece {id} has unknown type {other}"))),
            };
            pieces.push(surface.to_string());
            scores.push(score);
            types.push(kind);
        }
        let ids = u32::try_from(pieces.len()).map_err(|_| malformed("too many pieces"))?;
        let piece_ids = pieces.iter().cloned().zip(0..ids).collect();
        let mut byte_ids = [u32::MAX; 256];
        for (id, kind) in (0..ids).zip(&types) {
            if let PieceType::Byte(b) = kind {
                byte_ids[usize::from(*b)] = id;
            }
        }
        if byte_fallback && byte_ids.contains(&u32::MAX) {
            return Err(malformed(
                "byte fallback is on but some <0xXX> pieces are missing",
            ));
        }
        let mut markers: Vec<(Vec<char>, u32)> = (0..ids)
            .zip(&types)
            .filter(|(_, kind)| **kind == PieceType::UserDefined)
            .map(|(id, _)| (pieces[id as usize].chars().collect(), id))
            .collect();
        // Stable sort, longest first, as the reference's `sorted(..., reverse=True)`.
        markers.sort_by_key(|marker| std::cmp::Reverse(marker.0.len()));
        Ok(Tokenizer {
            pieces,
            scores,
            types,
            pad_id,
            eos_id,
            bos_id,
            unk_id,
            add_dummy_prefix,
            byte_fallback,
            piece_ids,
            byte_ids,
            markers,
        })
    }

    /// The tokenizer of an archive: its first RAW tensor.
    pub fn from_archive(archive: &Archive<'_>) -> Result<Tokenizer, Error> {
        let blob = archive
            .tensors()
            .iter()
            .find(|t| t.dtype() == DType::Raw)
            .ok_or_else(|| malformed("the archive carries no tokenizer"))?;
        Tokenizer::from_blob(blob.bytes())
    }

    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    pub fn piece_type(&self, id: u32) -> Option<PieceType> {
        self.types.get(id as usize).copied()
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// The id of the language tag `<|code|>` (e.g. `"de"`), if the tokenizer has it.
    pub fn language_tag(&self, code: &str) -> Option<u32> {
        let tag = format!("<|{code}|>");
        self.markers
            .iter()
            .find(|(chars, id)| {
                chars.len() == tag.chars().count() && self.pieces[*id as usize] == tag
            })
            .map(|(_, id)| *id)
    }

    /// Every language tag in id order: user-defined `<|xx|>` pieces whose code is lowercase
    /// ASCII letters (the chat markers such as `<|im_start|>` are not languages).
    pub fn language_tags(&self) -> Vec<(&str, u32)> {
        let mut tags: Vec<(&str, u32)> = self
            .markers
            .iter()
            .filter_map(|(_, id)| {
                let code = self.pieces[*id as usize]
                    .strip_prefix("<|")?
                    .strip_suffix("|>")?;
                (!code.is_empty() && code.bytes().all(|b| b.is_ascii_lowercase()))
                    .then_some((code, *id))
            })
            .collect();
        tags.sort_by_key(|&(_, id)| id);
        tags
    }

    /// Token ids for `text`, exactly as `RefTokenizer.encode`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut esc: Vec<char> = Vec::with_capacity(text.len() + 1);
        if self.add_dummy_prefix {
            esc.push(META_SPACE);
        }
        esc.extend(text.chars().map(|c| if c == ' ' { META_SPACE } else { c }));
        let (mut ids, mut span_start, mut i) = (Vec::new(), 0, 0);
        while i < esc.len() {
            let marker = self
                .markers
                .iter()
                .find(|(chars, _)| esc[i..].starts_with(chars));
            match marker {
                Some((chars, id)) if !chars.is_empty() => {
                    self.bpe(&esc[span_start..i], &mut ids);
                    ids.push(*id);
                    i += chars.len();
                    span_start = i;
                }
                _ => i += 1,
            }
        }
        self.bpe(&esc[span_start..], &mut ids);
        ids
    }

    /// Score-greedy BPE over one marker-free span (`RefTokenizer._bpe`).
    fn bpe(&self, span: &[char], ids: &mut Vec<u32>) {
        let mut syms: Vec<String> = span.iter().map(|c| c.to_string()).collect();
        let mut pair = String::new();
        while syms.len() > 1 {
            let mut best: Option<(f32, usize)> = None;
            for j in 0..syms.len() - 1 {
                pair.clear();
                pair.push_str(&syms[j]);
                pair.push_str(&syms[j + 1]);
                if let Some(&id) = self.piece_ids.get(pair.as_str()) {
                    let score = self.scores[id as usize];
                    // Strictly greater: ties keep the leftmost pair.
                    if best.is_none_or(|(s, _)| score > s) {
                        best = Some((score, j));
                    }
                }
            }
            let Some((_, j)) = best else { break };
            let right = syms.remove(j + 1);
            syms[j].push_str(&right);
        }
        for sym in &syms {
            if let Some(&id) = self.piece_ids.get(sym.as_str()) {
                ids.push(id);
            } else if self.byte_fallback {
                ids.extend(sym.bytes().map(|b| self.byte_ids[usize::from(b)]));
            } else {
                ids.push(self.unk_id);
            }
        }
    }

    /// Text for token ids, exactly as `RefTokenizer.decode`: byte pieces contribute their byte,
    /// control and unknown pieces nothing, invalid UTF-8 becomes U+FFFD, `▁` becomes a space.
    /// Ids outside the vocabulary contribute nothing, like unknown pieces.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.types.get(id as usize) {
                Some(PieceType::Byte(b)) => bytes.push(*b),
                Some(PieceType::Control | PieceType::Unknown) | None => {}
                Some(_) => bytes.extend_from_slice(self.pieces[id as usize].as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&bytes).replace(META_SPACE, " ");
        match text.strip_prefix(' ') {
            Some(rest) if self.add_dummy_prefix => rest.to_string(),
            _ => text,
        }
    }
}
