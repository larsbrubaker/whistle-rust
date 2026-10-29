//! The `.cact` model archive reader: header geometry, codebooks, tensor directory and blob
//! slices, every record bounds- and overflow-checked so a malformed or truncated archive is an
//! [`Error`], never a panic. Follows upstream's `export.read_export`. The archive borrows the
//! caller's bytes; `quant` turns a [`Tensor`] into f32 values and `tokenizer` reads the RAW
//! attachment.
//!
//! The layout, all little-endian: the header (48 u32 fields of geometry and `kv_bits`, then
//! `rope_theta` as f32), the CQ codebooks, a directory of nameless tensor records, then each
//! tensor's blob, 64-byte aligned, in the order upstream's exporter enumerates them (`bind`
//! gives each its role from that order), ending with the raw SentencePiece tokenizer (BPE pieces
//! with scores and types, the 7 language tags as user-defined pieces).

use crate::error::Error;

/// The archive tag every Whistle `.cact` file starts with.
pub const TAG: u32 = 0x05E1_2A84;
/// 48 u32 fields then `rope_theta` as f32.
const HEADER_BYTES: usize = 49 * 4;
/// `u8 dtype, u8 ndim, u16 pad, u32 shape[4], u64 offset, u64 nbytes, u32 group, u32 bits`.
const RECORD_BYTES: usize = 44;
/// The stored codebooks: `cb2[4] | cb3[8] | cb4[16]`.
const CODEBOOK_LEN: usize = 4 + 8 + 16;
/// The group size the stored codebooks were divided for (`cb / sqrt(group)` at export); a 2-4
/// bit CQ record must use it, since upstream recomputes the codebook for each record's group.
pub const CODEBOOK_GROUP: u32 = 128;
/// Record `bits` value that denotes ternary crumbs (three levels, 2-bit codes).
pub const TERNARY_RECORD_BITS: u32 = 5;

/// The model geometry from the archive header, field for field.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub num_tensors: u32,
    pub codebook_len: u32,
    pub kv_window: u32,
    pub kv_bits: u32,
    pub vocab: u32,
    pub out_vocab: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub max_seq_len: u32,
    pub hada_n: u32,
    pub mhc_lanes: u32,
    pub sliding_window: u32,
    /// Bit `i` set: layer `i` uses full attention.
    pub global_mask: u64,
    pub qkv_conv_taps: u32,
    pub engram_slots: u32,
    pub engram_sub_dim: u32,
    pub num_engram_tables: u32,
    pub engram_conv_taps: u32,
    pub engram_conv_dilation: u32,
    pub engram_seed_heads: u32,
    pub engram_orders: Vec<u32>,
    pub engram_sites: Vec<u32>,
    pub rope_theta: f32,
}

/// How a tensor's blob is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Fp16,
    Fp32,
    /// Cactus Quants: packed codebook indices then per-group fp16 norms.
    Cq,
    /// Uninterpreted bytes (the tokenizer).
    Raw,
}

/// One directory record with its blob, validated against the archive.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<'a> {
    dtype: DType,
    shape: Vec<usize>,
    group: u32,
    bits: u32,
    count: usize,
    bytes: &'a [u8],
}

impl<'a> Tensor<'a> {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The logical shape (`[out, in]` for CQ, unpadded).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The CQ group size (0 for other dtypes).
    pub fn group(&self) -> u32 {
        self.group
    }

    /// The CQ index width: 1-4, or [`TERNARY_RECORD_BITS`] (0 for other dtypes).
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The blob, borrowed from the archive bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of logical elements: the product of the shape (1 for a scalar), or the byte
    /// count for a RAW tensor, whose shape is not interpreted.
    pub fn element_count(&self) -> usize {
        self.count
    }
}

/// A parsed `.cact` archive borrowing the caller's bytes.
#[derive(Debug, Clone)]
pub struct Archive<'a> {
    geometry: Geometry,
    codebook: Vec<f32>,
    tensors: Vec<Tensor<'a>>,
}

fn malformed(message: impl Into<String>) -> Error {
    Error::Archive(message.into())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from(u32_at(bytes, at)) | (u64::from(u32_at(bytes, at + 4)) << 32)
}

fn to_usize(value: u64, what: &str) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| malformed(format!("{what} {value} does not fit in memory")))
}

impl<'a> Archive<'a> {
    /// Parses and validates the header, codebooks and directory. Blobs are borrowed, not copied.
    pub fn parse(bytes: &'a [u8]) -> Result<Archive<'a>, Error> {
        if bytes.len() < HEADER_BYTES {
            return Err(malformed(format!(
                "{} bytes is shorter than the {HEADER_BYTES}-byte header",
                bytes.len()
            )));
        }
        let field = |i: usize| u32_at(bytes, i * 4);
        let tag = field(0);
        if tag != TAG {
            return Err(malformed(format!(
                "tag {tag:#010x} is not a Whistle .cact archive (expected {TAG:#010x})"
            )));
        }
        let num_orders = field(26) as usize;
        let num_sites = field(31) as usize;
        if num_orders > 4 {
            return Err(malformed(format!("{num_orders} engram orders (at most 4)")));
        }
        if num_sites > 16 {
            return Err(malformed(format!("{num_sites} engram sites (at most 16)")));
        }
        let geometry = Geometry {
            num_tensors: field(1),
            codebook_len: field(2),
            kv_window: field(3),
            kv_bits: field(4),
            vocab: field(5),
            out_vocab: field(6),
            d_model: field(7),
            num_heads: field(8),
            num_kv_heads: field(9),
            num_layers: field(10),
            qk_head_dim: field(11),
            v_head_dim: field(12),
            max_seq_len: field(13),
            hada_n: field(14),
            mhc_lanes: field(15),
            sliding_window: field(16),
            global_mask: u64::from(field(17)) | (u64::from(field(18)) << 32),
            qkv_conv_taps: field(19),
            engram_slots: field(20),
            engram_sub_dim: field(21),
            num_engram_tables: field(22),
            engram_conv_taps: field(23),
            engram_conv_dilation: field(24),
            engram_seed_heads: field(25),
            engram_orders: (0..num_orders).map(|i| field(27 + i)).collect(),
            engram_sites: (0..num_sites).map(|i| field(32 + i)).collect(),
            rope_theta: f32::from_bits(field(48)),
        };
        if geometry.codebook_len as usize != CODEBOOK_LEN {
            return Err(malformed(format!(
                "codebook holds {} values; expected {CODEBOOK_LEN} (cb2 | cb3 | cb4)",
                geometry.codebook_len
            )));
        }
        let codebook_end = HEADER_BYTES + CODEBOOK_LEN * 4;
        if bytes.len() < codebook_end {
            return Err(malformed("truncated inside the codebook"));
        }
        let codebook = (0..CODEBOOK_LEN)
            .map(|i| f32::from_bits(u32_at(bytes, HEADER_BYTES + i * 4)))
            .collect();
        let num_tensors = geometry.num_tensors as usize;
        let directory_end = num_tensors
            .checked_mul(RECORD_BYTES)
            .and_then(|n| n.checked_add(codebook_end))
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| {
                malformed(format!(
                    "the directory of {num_tensors} tensors runs past the end of the {}-byte archive",
                    bytes.len()
                ))
            })?;
        let tensors = (0..num_tensors)
            .map(|i| parse_record(bytes, codebook_end + i * RECORD_BYTES, i))
            .collect::<Result<Vec<_>, _>>()?;
        debug_assert!(directory_end <= bytes.len());
        Ok(Archive {
            geometry,
            codebook,
            tensors,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The stored unit codebook for CQ width 2, 3 or 4 (`2^bits` sorted centroids, already
    /// divided by the square root of the export group size, [`CODEBOOK_GROUP`]; `parse` rejects
    /// 2-4 bit records with any other group). Widths 1 and ternary use analytic codebooks
    /// computed per group (`quant`), so they return `None`, as does any other width.
    pub fn codebook(&self, bits: u32) -> Option<&[f32]> {
        match bits {
            2 => Some(&self.codebook[0..4]),
            3 => Some(&self.codebook[4..12]),
            4 => Some(&self.codebook[12..28]),
            _ => None,
        }
    }

    /// Every tensor in file (canonical) order.
    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    pub fn tensor(&self, index: usize) -> Option<&Tensor<'a>> {
        self.tensors.get(index)
    }
}

/// Bytes per packed CQ row: `in_pad * bits / 8` (ternary crumbs are 2 bits each).
pub(crate) fn packed_row_bytes(in_pad: usize, bits: u32) -> usize {
    let width = if bits == TERNARY_RECORD_BITS {
        2
    } else {
        bits as usize
    };
    // in_pad is a multiple of 8 (checked by the caller), so this is exact.
    in_pad / 8 * width
}

fn parse_record<'a>(bytes: &'a [u8], at: usize, index: usize) -> Result<Tensor<'a>, Error> {
    let bad = |message: String| malformed(format!("tensor {index}: {message}"));
    let dtype = match bytes[at] {
        1 => DType::Fp16,
        2 => DType::Fp32,
        3 => DType::Cq,
        4 => DType::Raw,
        other => return Err(bad(format!("unknown dtype {other}"))),
    };
    let ndim = bytes[at + 1] as usize;
    if ndim > 4 {
        return Err(bad(format!("{ndim} dimensions (at most 4)")));
    }
    let shape = (0..ndim)
        .map(|d| to_usize(u64::from(u32_at(bytes, at + 4 + d * 4)), "dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let offset = u64_at(bytes, at + 20);
    let nbytes = u64_at(bytes, at + 28);
    let group = u32_at(bytes, at + 36);
    let bits = u32_at(bytes, at + 40);
    let end = offset
        .checked_add(nbytes)
        .filter(|&end| end <= bytes.len() as u64)
        .ok_or_else(|| {
            bad(format!(
                "blob at {offset} of {nbytes} bytes runs past the end of the {}-byte archive",
                bytes.len()
            ))
        })?;
    // Both fit: end <= bytes.len().
    let blob = &bytes[offset as usize..end as usize];
    // RAW blobs are uninterpreted bytes, so their shape is never multiplied out: the same
    // record is accepted on 32- and 64-bit targets.
    let count = if dtype == DType::Raw {
        blob.len()
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| bad(format!("shape {shape:?} overflows")))?
    };
    // fp16 and fp32 sizes equal the blob length, so every allocation `quant` makes for them is
    // bounded by bytes present; CQ bounds its padded row buffer by rejecting zero dimensions.
    let expected = match dtype {
        DType::Fp16 => count.checked_mul(2),
        DType::Fp32 => count.checked_mul(4),
        DType::Cq => Some(cq_bytes(&shape, group, bits).map_err(bad)?),
        DType::Raw => Some(blob.len()),
    };
    match expected {
        Some(n) if n == blob.len() => {}
        Some(n) => {
            return Err(bad(format!(
                "{dtype:?} shape {shape:?} needs {n} bytes, the record has {}",
                blob.len()
            )))
        }
        None => return Err(bad(format!("shape {shape:?} overflows"))),
    }
    let (group, bits) = if dtype == DType::Cq {
        (group, bits)
    } else {
        (0, 0)
    };
    Ok(Tensor {
        dtype,
        shape,
        group,
        bits,
        count,
        bytes: blob,
    })
}

/// The exact blob size of a CQ `[out, in]` record: packed indices then fp16 group norms.
fn cq_bytes(shape: &[usize], group: u32, bits: u32) -> Result<usize, String> {
    let [out, in_dim] = shape else {
        return Err(format!("CQ tensor has shape {shape:?}; expected [out, in]"));
    };
    if !matches!(bits, 1..=4 | TERNARY_RECORD_BITS) {
        return Err(format!("CQ width {bits} (expected 1-4 or 5 for ternary)"));
    }
    if *out == 0 || *in_dim == 0 {
        // With zero rows the blob size is zero whatever the width, so a huge width would reach
        // dequantization's padded row buffer with no bytes behind it.
        return Err(format!("CQ tensor has an empty shape {shape:?}"));
    }
    if group == 0 || !group.is_power_of_two() {
        return Err(format!("CQ group {group} is not a power of two"));
    }
    if matches!(bits, 2..=4) && group != CODEBOOK_GROUP {
        return Err(format!(
            "CQ width {bits} at group {group}: the stored codebook is for group {CODEBOOK_GROUP}"
        ));
    }
    let group = group as usize;
    let in_pad = in_dim
        .checked_next_multiple_of(group)
        .ok_or_else(|| format!("CQ input width {in_dim} overflows"))?;
    if in_pad % 8 != 0 {
        return Err(format!(
            "CQ padded width {in_pad} is not a multiple of 8 indices"
        ));
    }
    let row = packed_row_bytes(in_pad, bits)
        .checked_add(in_pad / group * 2)
        .ok_or_else(|| format!("CQ row of {in_pad} overflows"))?;
    out.checked_mul(row)
        .ok_or_else(|| format!("CQ shape {shape:?} overflows"))
}
