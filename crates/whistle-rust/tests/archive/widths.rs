// CQ dequantization at every record width (1, 2, 3, 4 bits and ternary crumbs), on synthetic
// archives built from the documented layout, against a dense evaluation of `_cq_unpack`'s
// formula in f64 (`(codebook[idx] * norm) @ H`, H the orthonormal Sylvester Walsh–Hadamard
// matrix); and fp16 conversion on every bit pattern. whistle.cact itself uses only 2 and 4 bits.

use whistle::cact::Archive;
use whistle::quant::fp16_to_f32;

use crate::support::archive::{build, pack_lsb, Record};
use crate::support::relative_l2;

const GROUP: usize = 128;
const OUT: usize = 3;
const IN: usize = 200; // padded to 256: two groups, the second partly padding

fn stored_codebook() -> [f32; 28] {
    let mut cb = [0.0f32; 28];
    for (i, v) in cb.iter_mut().enumerate() {
        *v = (i as f32 - 13.5) * 0.011;
    }
    cb
}

/// The unit codebook the reference uses at a record width.
fn reference_codebook(bits: u32) -> Vec<f64> {
    let root = (GROUP as f64).sqrt();
    let cb = stored_codebook();
    match bits {
        1 => {
            let c = ((2.0 / std::f64::consts::PI).sqrt() / root) as f32;
            vec![-f64::from(c), f64::from(c)]
        }
        5 => {
            let c = (1.224_006_4 / root) as f32;
            vec![-f64::from(c), 0.0, f64::from(c)]
        }
        2 => cb[0..4].iter().map(|&v| f64::from(v)).collect(),
        3 => cb[4..12].iter().map(|&v| f64::from(v)).collect(),
        4 => cb[12..28].iter().map(|&v| f64::from(v)).collect(),
        _ => unreachable!(),
    }
}

fn f16_bits(value: f32) -> u16 {
    // Exact for the few values used here (small dyadic rationals).
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    sign | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

fn synthetic(bits: u32) -> (Vec<u8>, Vec<f32>) {
    let in_pad = IN.next_multiple_of(GROUP);
    let levels = if bits == 5 { 3 } else { 1usize << bits };
    let mut state = 0x2545_f491_u32.wrapping_add(bits);
    let mut blob = Vec::new();
    let mut trits = Vec::new();
    for _ in 0..OUT {
        let row: Vec<u8> = (0..in_pad)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 16) as usize % levels) as u8
            })
            .collect();
        let codes: Vec<u8> = if bits == 5 {
            // Trits 0, 1, 2 are stored as crumb codes 3, 0, 1.
            row.iter().map(|&t| [3, 0, 1][t as usize]).collect()
        } else {
            row.clone()
        };
        blob.extend(pack_lsb(&codes, if bits == 5 { 2 } else { bits as usize }));
        trits.push(row);
    }
    let norms = [0.5f32, 1.25, 2.0, 0.75, 3.0, 1.5];
    for &n in &norms {
        blob.extend(f16_bits(n).to_le_bytes());
    }
    let cb = reference_codebook(bits);
    let mut expected = Vec::new();
    for (r, row) in trits.iter().enumerate() {
        let mut full = vec![0.0f64; in_pad];
        for g in 0..in_pad / GROUP {
            let norm = f64::from(norms[r * 2 + g]);
            for (j, out) in full[g * GROUP..(g + 1) * GROUP].iter_mut().enumerate() {
                *out = (0..GROUP)
                    .map(|i| {
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        cb[row[g * GROUP + i] as usize] * norm * sign
                    })
                    .sum::<f64>()
                    / (GROUP as f64).sqrt();
            }
        }
        expected.extend(full[..IN].iter().map(|&v| v as f32));
    }
    let archive = build(
        &stored_codebook(),
        &[Record {
            dtype: 3,
            shape: vec![OUT as u32, IN as u32],
            group: GROUP as u32,
            bits,
            blob,
        }],
    );
    (archive, expected)
}

#[test]
fn every_cq_width_dequantizes_per_the_reference_formula() {
    for bits in [1, 2, 3, 4, 5] {
        let (bytes, expected) = synthetic(bits);
        let archive = Archive::parse(&bytes).unwrap_or_else(|e| panic!("bits {bits}: {e}"));
        let tensor = &archive.tensors()[0];
        let values = tensor.to_f32(&archive).unwrap();
        assert_eq!(values.len(), OUT * IN);
        let err = relative_l2(&values, &expected);
        eprintln!("CQ bits {bits}: relative L2 {err:e}");
        assert!(err <= 1e-6, "bits {bits}: relative L2 {err:e}");
    }
}

#[test]
fn ternary_crumb_code_two_is_an_error() {
    let (mut bytes, _) = synthetic(5);
    let archive = Archive::parse(&bytes).unwrap();
    let start = archive.tensors()[0].bytes().as_ptr() as usize - bytes.as_ptr() as usize;
    bytes[start] = (bytes[start] & !0b11) | 0b10;
    let archive = Archive::parse(&bytes).unwrap();
    assert!(archive.tensors()[0].to_f32(&archive).is_err());
}

#[test]
fn fp16_converts_every_bit_pattern_exactly() {
    for half in 0..=u16::MAX {
        let got = fp16_to_f32(half);
        let sign = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((half >> 10) & 0x1f);
        let mantissa = f64::from(half & 0x3ff);
        if exponent == 0x1f {
            if mantissa == 0.0 {
                assert_eq!(got, sign as f32 * f32::INFINITY, "{half:#06x}");
            } else {
                assert!(got.is_nan(), "{half:#06x}");
            }
            continue;
        }
        let want = if exponent == 0 {
            sign * mantissa * 2f64.powi(-24)
        } else {
            sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15)
        };
        assert_eq!(f64::from(got), want, "{half:#06x}");
        assert_eq!(got.is_sign_negative(), sign < 0.0, "{half:#06x} sign");
    }
}
