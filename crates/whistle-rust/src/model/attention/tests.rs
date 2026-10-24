// Unit tests for `attention`'s RoPE step (`rotate`): the start position is explicit, so a
// sequence rotated from position 0 matches the same rows rotated one at a time at their own
// positions, as a decoder step rotating one cached position needs.

use super::rotate;
use crate::model::ops::Rope;

#[test]
fn rotating_a_sequence_from_zero_equals_rotating_each_row_at_its_position() {
    let (heads, frames, dim) = (3usize, 5usize, 8usize);
    // A table longer than the sequence, as a decoder's would be.
    let rope = Rope::new(dim, frames + 4, 1e5);
    let x: Vec<f32> = (0..heads * frames * dim)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let mut whole = x.clone();
    rotate(&mut whole, &rope, dim, frames, 0);
    for h in 0..heads {
        for t in 0..frames {
            let at = (h * frames + t) * dim;
            let mut row = x[at..at + dim].to_vec();
            rotate(&mut row, &rope, dim, 1, t);
            assert_eq!(row, whole[at..at + dim], "head {h} frame {t}");
        }
    }
    // A shifted start moves every row by the same offset.
    let mut shifted = x[..frames * dim].to_vec();
    rotate(&mut shifted, &rope, dim, frames, 3);
    let mut row = x[dim..2 * dim].to_vec();
    rotate(&mut row, &rope, dim, 1, 4);
    assert_eq!(row, shifted[dim..2 * dim]);
}
