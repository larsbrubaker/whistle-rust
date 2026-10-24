// Unit tests for `quant`'s private helpers: the bitstream field reader and the transform's
// agreement with the dense Sylvester Walsh–Hadamard matrix.

use super::*;

#[test]
fn bitstream_fields_are_lsb_first() {
    // Indices 1..=8 at 3 bits each, packed LSB-first into one 24-bit word.
    let word: u32 = (0..8).map(|i| ((i + 1) as u32 & 7) << (3 * i)).sum();
    let row = [word as u8, (word >> 8) as u8, (word >> 16) as u8];
    let got: Vec<usize> = (0..8).map(|k| field(&row, k, 3)).collect();
    assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7, 0]);
}

#[test]
fn walsh_hadamard_equals_the_dense_sylvester_matrix() {
    let n = 16;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut fast = x.clone();
    walsh_hadamard(&mut fast);
    for (j, &got) in fast.iter().enumerate() {
        // Sylvester H[i][j] = (-1)^popcount(i & j).
        let dense: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                f64::from(v)
                    * if (i & j).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
            })
            .sum::<f64>()
            / (n as f64).sqrt();
        assert!(
            (f64::from(got) - dense).abs() < 1e-6,
            "{j}: {got} vs {dense}"
        );
    }
}
