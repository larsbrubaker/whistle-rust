// Archive test support: the archive oracle fixture (test-data/fixtures/archive/, written by
// oracle/dump_archive.py), the model bytes, and a builder for small synthetic `.cact` archives
// that the width and malformed-input tests construct byte by byte from the documented layout,
// and a builder for synthetic tokenizer blobs (`export.parse_tokenizer_blob` layout).

use serde_json::Value;

use super::{npy, repo_root, Npy};

/// The model archive's bytes (models/whistle.cact).
pub fn model_bytes() -> Vec<u8> {
    let path = super::model_path();
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn folder() -> std::path::PathBuf {
    repo_root().join("test-data/fixtures/archive")
}

/// test-data/fixtures/archive/manifest.json.
pub fn manifest() -> Value {
    super::read_json(&folder().join("manifest.json"))
}

/// One full array the archive manifest names in a tensor's `file`.
pub fn array(file: &str) -> Npy {
    npy::read(&folder().join(file)).unwrap_or_else(|e| panic!("{file}: {e}"))
}

pub const TAG: u32 = 0x05E1_2A84;

/// One directory record and its blob, for [`build`].
pub struct Record {
    pub dtype: u8,
    pub shape: Vec<u32>,
    pub group: u32,
    pub bits: u32,
    pub blob: Vec<u8>,
}

/// A well-formed archive: the 49-field header (zero geometry apart from the tag, the tensor
/// count and the codebook length), the 28-value codebook, the directory, then 64-byte-aligned
/// blobs.
pub fn build(codebook: &[f32; 28], records: &[Record]) -> Vec<u8> {
    let mut header = [0u32; 49];
    header[0] = TAG;
    header[1] = records.len() as u32;
    header[2] = 28;
    let mut out: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.extend(codebook.iter().flat_map(|v| v.to_le_bytes()));
    let mut pos = out.len() + records.len() * 44;
    let mut offsets = Vec::new();
    for r in records {
        pos = pos.next_multiple_of(64);
        offsets.push(pos);
        pos += r.blob.len();
    }
    for (r, &offset) in records.iter().zip(&offsets) {
        out.push(r.dtype);
        out.push(r.shape.len() as u8);
        out.extend([0, 0]);
        for d in 0..4 {
            out.extend(r.shape.get(d).copied().unwrap_or(0).to_le_bytes());
        }
        out.extend((offset as u64).to_le_bytes());
        out.extend((r.blob.len() as u64).to_le_bytes());
        out.extend(r.group.to_le_bytes());
        out.extend(r.bits.to_le_bytes());
    }
    for (r, &offset) in records.iter().zip(&offsets) {
        out.resize(offset, 0);
        out.extend(&r.blob);
    }
    out
}

/// Packs indices LSB-first, `width` bits each, into one row's bytes (`export._pack_lsb`).
pub fn pack_lsb(indices: &[u8], width: usize) -> Vec<u8> {
    let mut row = vec![0u8; indices.len() * width / 8];
    for (k, &idx) in indices.iter().enumerate() {
        for b in 0..width {
            if idx >> b & 1 == 1 {
                let bit = k * width + b;
                row[bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    row
}

/// A tokenizer blob: header (pad 0, eos 1, bos 2, unk 3, no dummy prefix), then
/// `(score, type, surface)` records in id order.
pub fn tokenizer_blob(byte_fallback: bool, pieces: &[(f32, u8, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [pieces.len() as u32, 0, 1, 2, 3] {
        out.extend(v.to_le_bytes());
    }
    out.extend([0, u8::from(byte_fallback), 0, 0]);
    for &(score, kind, surface) in pieces {
        out.extend(score.to_le_bytes());
        out.push(kind);
        out.extend((surface.len() as u16).to_le_bytes());
        out.extend(surface);
    }
    out
}
