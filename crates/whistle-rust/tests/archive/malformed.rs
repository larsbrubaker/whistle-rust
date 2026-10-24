// Malformed or truncated archives are errors with a message, never panics: empty input, a bad
// tag, a truncated header or directory, a record pointing past the end, absurd or inconsistent
// shapes, and every prefix of a small valid archive.

use whistle::cact::Archive;
use whistle::Error;

use crate::support::archive::{build, Record, TAG};

fn small() -> Vec<u8> {
    build(
        &[0.0; 28],
        &[
            Record {
                dtype: 1,
                shape: vec![4],
                group: 0,
                bits: 0,
                blob: vec![0; 8],
            },
            Record {
                dtype: 3,
                shape: vec![2, 128],
                group: 128,
                bits: 2,
                blob: vec![0; 2 * 32 + 2 * 2],
            },
            Record {
                dtype: 4,
                shape: vec![],
                group: 0,
                bits: 0,
                blob: vec![1, 2, 3],
            },
        ],
    )
}

fn assert_archive_error(bytes: &[u8], what: &str) {
    match Archive::parse(bytes) {
        Err(Error::Archive(message)) => assert!(!message.is_empty(), "{what}"),
        Err(other) => panic!("{what}: unexpected error {other}"),
        Ok(_) => panic!("{what}: parsed"),
    }
}

/// Directory record `i`'s byte offset in a `build` archive.
fn record(i: usize) -> usize {
    49 * 4 + 28 * 4 + i * 44
}

#[test]
fn the_small_archive_is_well_formed() {
    let bytes = small();
    let archive = Archive::parse(&bytes).expect("well-formed");
    assert_eq!(archive.tensors().len(), 3);
    assert_eq!(archive.tensors()[2].bytes(), &[1, 2, 3]);
}

#[test]
fn empty_and_short_input_is_an_error() {
    assert_archive_error(&[], "empty");
    assert_archive_error(&TAG.to_le_bytes(), "tag only");
}

#[test]
fn a_bad_tag_is_an_error() {
    let mut bytes = small();
    bytes[0] ^= 1;
    assert_archive_error(&bytes, "bad tag");
    let message = Archive::parse(&bytes).unwrap_err().to_string();
    assert!(message.contains("not a Whistle .cact archive"), "{message}");
}

#[test]
fn every_truncation_is_an_error() {
    let bytes = small();
    for len in 0..bytes.len() {
        assert_archive_error(&bytes[..len], &format!("truncated to {len}"));
    }
}

#[test]
fn a_record_past_the_end_is_an_error() {
    let mut bytes = small();
    let len = bytes.len() as u64;
    bytes[record(2) + 20..record(2) + 28].copy_from_slice(&(len - 1).to_le_bytes());
    assert_archive_error(&bytes, "offset past the end");
    let mut bytes = small();
    bytes[record(2) + 20..record(2) + 28].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_archive_error(&bytes, "offset overflows");
    let mut bytes = small();
    bytes[record(2) + 28..record(2) + 36].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_archive_error(&bytes, "length overflows");
}

#[test]
fn a_huge_tensor_count_is_an_error() {
    let mut bytes = small();
    bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_archive_error(&bytes, "tensor count");
}

#[test]
fn absurd_and_inconsistent_shapes_are_errors() {
    let set_u32 = |bytes: &mut Vec<u8>, at: usize, v: u32| {
        bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    let cases: [(&str, usize, u32); 9] = [
        ("fp16 dims overflow", record(0) + 4, u32::MAX),
        ("fp16 size mismatch", record(0) + 4, 5),
        ("cq rows overflow", record(1) + 4, u32::MAX),
        ("cq columns overflow", record(1) + 8, u32::MAX),
        ("cq group not a power of two", record(1) + 36, 96),
        ("cq group zero", record(1) + 36, 0),
        ("cq width 6", record(1) + 40, 6),
        ("cq width 0", record(1) + 40, 0),
        ("cq width 3 with a 2-bit blob", record(1) + 40, 3),
    ];
    for (what, at, value) in cases {
        let mut bytes = small();
        set_u32(&mut bytes, at, value);
        assert_archive_error(&bytes, what);
    }
    let mut bytes = small();
    bytes[record(0) + 1] = 5;
    assert_archive_error(&bytes, "five dimensions");
    let mut bytes = small();
    bytes[record(1) + 1] = 3;
    assert_archive_error(&bytes, "cq with three dimensions");
    let mut bytes = small();
    bytes[record(0)] = 9;
    assert_archive_error(&bytes, "unknown dtype");
    let mut bytes = small();
    set_u32(&mut bytes, 26 * 4, 5);
    assert_archive_error(&bytes, "five engram orders");
    let mut bytes = small();
    set_u32(&mut bytes, 31 * 4, 17);
    assert_archive_error(&bytes, "seventeen engram sites");
    let mut bytes = small();
    set_u32(&mut bytes, 2 * 4, 1000);
    assert_archive_error(&bytes, "codebook length");
}

fn single(dtype: u8, shape: Vec<u32>, group: u32, bits: u32, blob: Vec<u8>) -> Vec<u8> {
    build(
        &[0.0; 28],
        &[Record {
            dtype,
            shape,
            group,
            bits,
            blob,
        }],
    )
}

#[test]
fn a_cq_record_with_a_zero_dimension_is_an_error() {
    // Zero rows make the expected blob size zero whatever the width, so without this check a
    // huge width would reach dequantization with no bytes behind it.
    for shape in [
        vec![0, 1 << 31],
        vec![0, u32::MAX],
        vec![0, 128],
        vec![4, 0],
    ] {
        let bytes = single(3, shape.clone(), 128, 2, Vec::new());
        assert_archive_error(&bytes, &format!("CQ shape {shape:?}"));
    }
}

#[test]
fn a_stored_width_cq_record_needs_the_codebook_group() {
    // [2, 128] at group 64, 2 bits: 2 rows of 32 packed bytes, then 2 x 2 fp16 norms.
    let bytes = single(3, vec![2, 128], 64, 2, vec![0; 2 * 32 + 2 * 2 * 2]);
    let message = Archive::parse(&bytes).unwrap_err().to_string();
    assert!(
        message.contains("group 64") && message.contains("128"),
        "{message}"
    );
    // The analytic widths are computed per group, so the same record at 1 bit is accepted.
    let bytes = single(3, vec![2, 128], 64, 1, vec![0; 2 * 16 + 2 * 2 * 2]);
    let archive = Archive::parse(&bytes).expect("1-bit record at group 64");
    assert_eq!(archive.tensors()[0].to_f32(&archive).unwrap().len(), 256);
}

#[test]
fn a_raw_record_ignores_its_shape() {
    // RAW blobs are uninterpreted bytes, so an overflowing shape is accepted identically on
    // 32- and 64-bit targets and the element count is the byte count.
    let bytes = single(4, vec![u32::MAX; 4], 0, 0, vec![7; 5]);
    let archive = Archive::parse(&bytes).expect("RAW with any shape");
    assert_eq!(archive.tensors()[0].element_count(), 5);
}
