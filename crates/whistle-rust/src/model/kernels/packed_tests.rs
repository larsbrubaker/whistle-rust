//! Unit tests of the packed-weight projection (`kernels::packed`): the SIMD path against the
//! scalar order bit for bit, and the scalar order against the dense projection of the
//! dequantized matrix (close, not equal: the summation differs). The SIMD side goes through
//! `best_path`, so it is NEON on aarch64, AVX2 on an x86_64 that has it, and simd128 on wasm
//! built with it: the same tests cover each path on its hardware.

use super::super::{best_path, Path};
use super::{Order, PackedMatrix};
use crate::cact::Archive;
use crate::quant::CqRows;

/// Every packed matrix of the bundled archive, if present (the tests skip without it).
fn archive_matrices() -> Option<Vec<PackedMatrix>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the workspace root")
        .join("models/whistle.cact");
    let bytes = std::fs::read(path).ok()?;
    let archive = Archive::parse(&bytes).ok()?;
    let mut out = Vec::new();
    for t in archive.tensors() {
        if t.dtype() != crate::cact::DType::Cq || t.shape().len() != 2 {
            continue;
        }
        let cq = CqRows::from_tensor_padded(t, &archive).ok()?;
        if let Some(p) = PackedMatrix::new(cq, Order::Rotated) {
            out.push(p);
        }
    }
    Some(out)
}

/// A deterministic pseudo-random activation row set.
fn inputs(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..rows * cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn the_simd_path_reproduces_the_scalar_order_on_every_archive_matrix() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    assert!(!matrices.is_empty());
    let mut widths = std::collections::BTreeSet::new();
    for (i, m) in matrices.iter().enumerate() {
        widths.insert(m.bits);
        // 1, 4 and 5 rows: the single-row path, one block, a block and a tail; 2, 3, 6 and 7
        // the zero-padded block of a 2- or 3-row tail.
        for rows in [1, 2, 3, 4, 5, 6, 7] {
            let x = inputs(rows, m.cols(), i as u32 + rows as u32);
            let count = m.rows().min(37);
            let first = m.rows() - count;
            let scalar = m.project(Path::Scalar, 1, &x, first, count);
            let simd = m.project(best_path(), 1, &x, first, count);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&scalar), bits(&simd), "matrix {i}, {rows} rows");
        }
    }
    // Whistle's archive has both widths: the 2-bit and 4-bit decodes are each exercised.
    assert_eq!(widths.into_iter().collect::<Vec<_>>(), vec![2, 4]);
}

#[test]
fn the_packed_projection_matches_the_dense_one_to_rounding() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    for (i, m) in matrices.iter().enumerate() {
        let x = inputs(3, m.cols(), 7 + i as u32);
        let dense = m.dense();
        let want = crate::model::ops::linear(&x, &dense, m.cols(), m.rows());
        let got = m.project(best_path(), 1, &x, 0, m.rows());
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in got.iter().zip(&want) {
            num += f64::from(a - b).powi(2);
            den += f64::from(*b).powi(2);
        }
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        assert!(rel < 1e-5, "matrix {i}: relative L2 {rel:e}");
    }
}

#[test]
#[cfg_attr(
    target_arch = "wasm32",
    ignore = "wasm32 cannot spawn worker threads; the library never asks for more than one there"
)]
fn threads_do_not_change_the_packed_projection() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    let m = matrices.iter().max_by_key(|m| m.rows() * m.cols()).unwrap();
    let x = inputs(64, m.cols(), 3);
    let one = m.project(best_path(), 1, &x, 0, m.rows());
    let four = m.project(best_path(), 4, &x, 0, m.rows());
    assert!(one
        .iter()
        .zip(&four)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn every_cq_matrix_the_model_projects_is_bound_packed_in_its_order() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the workspace root")
        .join("models/whistle.cact");
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    let archive = Archive::parse(&bytes).unwrap();
    let w = crate::model::Weights::from_archive(&archive).unwrap();
    // The intended table: the decoder projects by the rotated order (its full pass and cached
    // step share it), except the cross-attention keys and values, which only project the
    // encoder's frames; the encoder and the stem project by the dense order. A reordering in
    // `bind` that moved a matrix would change its numerics silently, so each is pinned.
    let mut rotated = vec![("embedding", &w.embedding)];
    let mut dense = Vec::new();
    for l in &w.decoder {
        let (a, c) = (&l.self_attn, &l.cross_attn);
        rotated.extend([
            ("self q", &a.q_proj),
            ("self k", &a.k_proj),
            ("self v", &a.v_proj),
            ("self gate", &a.gate_proj),
            ("self out", &a.out_proj),
            ("cross q", &c.q_proj),
            ("cross gate", &c.gate_proj),
            ("cross out", &c.out_proj),
        ]);
        dense.extend([("cross k", &c.k_proj), ("cross v", &c.v_proj)]);
    }
    let m = &w.decoder_mhc;
    rotated.extend([
        ("phi_pre", &m.phi_pre),
        ("phi_post", &m.phi_post),
        ("phi_res", &m.phi_res),
    ]);
    for e in &w.engrams {
        rotated.extend([("engram key", &e.key_proj), ("engram value", &e.value_proj)]);
    }
    for l in &w.encoder {
        let a = &l.self_attn;
        dense.extend([
            ("enc q", &a.q_proj),
            ("enc k", &a.k_proj),
            ("enc v", &a.v_proj),
            ("enc gate", &a.gate_proj),
            ("enc out", &a.out_proj),
            ("pw1", &l.pw1),
            ("pw2", &l.pw2),
        ]);
    }
    let m = &w.encoder_mhc;
    dense.extend([
        ("enc phi_pre", &m.phi_pre),
        ("enc phi_post", &m.phi_post),
        ("enc phi_res", &m.phi_res),
    ]);
    dense.extend(w.stem.stages.iter().map(|s| ("stem pw", &s.pw)));
    dense.push(("stem out", &w.stem.out));
    for (name, m) in &rotated {
        assert_eq!(m.packed_order(), Some(Order::Rotated), "{name}");
    }
    for (name, m) in &dense {
        assert_eq!(m.packed_order(), Some(Order::Dense), "{name}");
    }
    // Every CQ record but the two Engram tables (packed as `CqRows`) is one of these.
    assert_eq!(rotated.len() + dense.len() + w.engrams.len(), 152);
    // fp16 matrices stay dense.
    let (e, s) = (&w.encoder[0], &w.stem);
    assert!(!s.w.is_packed() && !e.dw.is_packed() && !w.mel.is_packed());
}

#[test]
fn the_dense_order_reproduces_the_dequantized_projection_bit_for_bit() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    for (i, m) in matrices.iter().enumerate() {
        let dense_order = PackedMatrix::new(m.cq.clone(), Order::Dense).unwrap();
        let x = inputs(5, m.cols(), 11 + i as u32);
        let want = super::super::linear_with(best_path(), 1, &x, &m.dense(), m.cols(), m.rows());
        let got = dense_order.project(best_path(), 1, &x, 0, m.rows());
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "matrix {i}"
        );
    }
}

#[test]
fn the_band_decode_reproduces_the_archive_dequantization_bit_for_bit_on_every_path() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    for (i, m) in matrices.iter().enumerate() {
        let want = bits(&m.cq.to_f32().unwrap());
        for path in [Path::Scalar, best_path()] {
            let mut got = vec![0.0f32; m.rows() * m.cols()];
            let mut padded = vec![0.0f32; m.in_pad];
            m.band_into(path, 0, &mut got, &mut padded);
            assert_eq!(bits(&got), want, "matrix {i}, {path:?}");
        }
    }
}

#[test]
#[cfg_attr(
    target_arch = "wasm32",
    ignore = "wasm32 cannot spawn worker threads; the library never asks for more than one there"
)]
fn threads_do_not_change_the_dense_order() {
    let Some(matrices) = archive_matrices() else {
        eprintln!("models/whistle.cact missing: skipped");
        return;
    };
    let m = matrices.iter().max_by_key(|m| m.rows() * m.cols()).unwrap();
    let m = PackedMatrix::new(m.cq.clone(), Order::Dense).unwrap();
    let x = inputs(64, m.cols(), 5);
    let (first, count) = (3, m.rows() - 3);
    let one = m.project(best_path(), 1, &x, first, count);
    let four = m.project(best_path(), 4, &x, first, count);
    assert!(one
        .iter()
        .zip(&four)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// The band decode's throughput over every archive matrix, best of 5, scalar and SIMD:
/// `cargo test --release --lib band_decode_timing -- --ignored --nocapture`. Measured
/// 2026-10-24 (Apple M5, load average 10–12): scalar 1.9–2.0 ns per weight, NEON 0.37–0.58.
#[test]
#[ignore = "release measurement of the band decode"]
fn band_decode_timing() {
    let m = archive_matrices().unwrap();
    let total: usize = m.iter().map(|m| m.rows() * m.cols()).sum();
    for path in [Path::Scalar, best_path()] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for m in &m {
                let mut out = vec![0.0f32; 64 * m.cols()];
                let mut padded = vec![0.0f32; m.in_pad];
                for b0 in (0..m.rows()).step_by(64) {
                    let w = 64.min(m.rows() - b0);
                    m.band_into(path, b0, &mut out[..w * m.cols()], &mut padded);
                }
                std::hint::black_box(&out);
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "{path:?}: {total} weights in {:.1} ms, {:.3} ns/weight",
            best * 1e3,
            best * 1e9 / total as f64
        );
    }
}

/// A synthetic archive of CQ records `[rows, cols]` at `bits` (group 128), pseudo-random
/// indices, norms and codebook, in the `.cact` layout `Archive::parse` checks.
fn synthetic(records: &[(usize, usize, u32)]) -> Vec<u8> {
    let mut seed = 0x1234_5678u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut header = [0u32; 49];
    header[0] = 0x05E1_2A84;
    header[1] = records.len() as u32;
    header[2] = 28;
    let mut out: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
    for _ in 0..28 {
        let v = (next() as f32 / u32::MAX as f32 - 0.5) * 0.3;
        out.extend(v.to_le_bytes());
    }
    let blobs: Vec<Vec<u8>> = records
        .iter()
        .map(|&(rows, cols, bits)| {
            let in_pad = cols.next_multiple_of(128);
            let mut blob: Vec<u8> = (0..rows * in_pad / 8 * bits as usize)
                .map(|_| next() as u8)
                .collect();
            for _ in 0..rows * in_pad / 128 {
                // A normal fp16 between about 0.25 and 4, either sign.
                let half = (0x3400 + (next() % 0x1000) as u16) | (((next() & 1) as u16) << 15);
                blob.extend(half.to_le_bytes());
            }
            blob
        })
        .collect();
    let mut pos = out.len() + records.len() * 44;
    let mut offsets = Vec::new();
    for b in &blobs {
        pos = pos.next_multiple_of(64);
        offsets.push(pos);
        pos += b.len();
    }
    for (&(rows, cols, bits), (b, &offset)) in records.iter().zip(blobs.iter().zip(&offsets)) {
        out.extend([3u8, 2, 0, 0]);
        for d in [rows as u32, cols as u32, 0, 0] {
            out.extend(d.to_le_bytes());
        }
        out.extend((offset as u64).to_le_bytes());
        out.extend((b.len() as u64).to_le_bytes());
        out.extend(128u32.to_le_bytes());
        out.extend(bits.to_le_bytes());
    }
    for (b, &offset) in blobs.iter().zip(&offsets) {
        out.resize(offset, 0);
        out.extend(b);
    }
    out
}

#[test]
fn padded_records_project_and_dequantize_as_the_archive_does() {
    // Rows of 200 (two groups, the second padded) and of 100 (one padded group), each width.
    let shapes = [(37, 200, 2), (37, 200, 4), (21, 100, 2), (21, 100, 4)];
    let bytes = synthetic(&shapes);
    let archive = Archive::parse(&bytes).unwrap();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    for (i, t) in archive.tensors().iter().enumerate() {
        let want = t.to_f32(&archive).unwrap();
        let cq = CqRows::from_tensor_padded(t, &archive).unwrap();
        let rotated = PackedMatrix::new(cq.clone(), Order::Rotated).unwrap();
        let dense_order = PackedMatrix::new(cq, Order::Dense).unwrap();
        let (rows, cols) = (rotated.rows(), rotated.cols());
        assert!(cols < rotated.in_pad, "record {i} is padded");
        // The whole matrix and every row, on every path, as `Tensor::to_f32`.
        assert_eq!(bits(&rotated.dense()), bits(&want), "record {i} dense");
        for path in [Path::Scalar, best_path()] {
            let mut got = vec![0.0f32; rows * cols];
            let mut padded = vec![0.0f32; rotated.in_pad];
            rotated.band_into(path, 0, &mut got, &mut padded);
            assert_eq!(bits(&got), bits(&want), "record {i} band {path:?}");
        }
        let mut row = vec![0.0f32; cols];
        for r in 0..rows {
            rotated.row_into(r, &mut row).unwrap();
            assert_eq!(
                bits(&row),
                bits(&want[r * cols..(r + 1) * cols]),
                "record {i} row {r}"
            );
        }
        assert!(rotated.row_into(rows, &mut row).is_err());
        // Through `Matrix`: values, row gather and equality without a dense copy.
        let m = crate::model::Matrix::packed(rotated.clone());
        assert_eq!(bits(&m.data()), bits(&want), "record {i} Matrix::data");
        m.row_into(rows - 1, &mut row).unwrap();
        assert_eq!(
            bits(&row),
            bits(&want[(rows - 1) * cols..]),
            "record {i} Matrix::row_into"
        );
        assert!(m == crate::model::Matrix::new(rows, cols, want.clone()).unwrap());
        assert!(m == crate::model::Matrix::packed(dense_order.clone()));
        // Projections: 1, 5 and 8 rows (the few-row and panel dense kernels).
        for n in [1, 5, 8] {
            let x = inputs(n, cols, 31 + i as u32);
            let reference = super::super::linear_with(best_path(), 1, &x, &want, cols, rows);
            let got = dense_order.project(best_path(), 1, &x, 0, rows);
            assert_eq!(
                bits(&got),
                bits(&reference),
                "record {i}, {n} rows, dense order"
            );
            let scalar = rotated.project(Path::Scalar, 1, &x, 0, rows);
            let simd = rotated.project(best_path(), 1, &x, 0, rows);
            assert_eq!(bits(&scalar), bits(&simd), "record {i}, {n} rows, rotated");
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (a, b) in simd.iter().zip(&reference) {
                num += f64::from(a - b).powi(2);
                den += f64::from(*b).powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(
                rel < 1e-5,
                "record {i}, {n} rows: rotated vs dense relative L2 {rel:e}"
            );
        }
    }
}
