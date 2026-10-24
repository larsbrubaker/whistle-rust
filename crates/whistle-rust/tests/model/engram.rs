// The decoder's Engram sites (`architecture.Engram`, called from `Whistle.decode`): the n-gram
// hash against upstream's `engram_indices` (test-data/fixtures/engram_indices.json, written by
// oracle/dump_engram.py), every clip's Engram keys and values against `decoder/engram_k` and
// `decoder/engram_v`, and the gated Engram add at the top of `DecoderBlock` against
// en_samantha's decoder layer 3 (an Engram site) internals.
//
// The hash is exact (ids identical). The keys and values have an A8 step inside (the gathered
// rows are fake-quantized before the projections), so they are quantized stages and their bounds
// are set from the error measured when they landed (CLAUDE.md "Tolerance policy").

use serde_json::Value;
use whistle::cact::Archive;
use whistle::model::{
    engram_add, engram_indices, ngram_in_sequence, tap_in_sequence, Engram, Weights,
};
use whistle::Error;

use crate::shared::weights;
use crate::support::archive::model_bytes;
use crate::support::{clip_names, read_json, relative_l2, repo_root, Fixture};

fn hash_fixture() -> Value {
    read_json(&repo_root().join("test-data/fixtures/engram_indices.json"))
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("unsigned")).expect("fits u32"))
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<u32>> {
    v.as_array().expect("rows").iter().map(u32s).collect()
}

#[test]
fn engram_indices_match_the_reference_hash_including_u32_wraparound() {
    let fx = hash_fixture();
    let orders = u32s(&fx["orders"]);
    let heads = fx["heads"].as_u64().unwrap() as usize;
    let slots = fx["slots"].as_u64().unwrap() as u32;
    let seed_heads = fx["seed_heads"].as_u64().unwrap() as usize;
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 5);
    for case in cases {
        let tokens = u32s(&case["tokens"]);
        let want: Vec<u32> = rows(&case["indices"]).concat();
        let got = engram_indices(&tokens, &orders, heads, slots, seed_heads).unwrap();
        assert_eq!(got, want, "tokens {tokens:?}");
    }
}

#[test]
fn engram_masks_match_the_reference_for_an_unpadded_sequence() {
    let fx = hash_fixture();
    let orders = u32s(&fx["orders"]);
    let heads = fx["heads"].as_u64().unwrap() as usize;
    let dilation = *orders.iter().max().unwrap() as usize;
    for case in fx["cases"].as_array().unwrap() {
        let ngram_ok = rows(&case["ngram_ok"]);
        for (t, row) in ngram_ok.iter().enumerate() {
            let tables = orders.iter().flat_map(|&o| std::iter::repeat_n(o, heads));
            for (table, order) in tables.enumerate() {
                let got = u32::from(ngram_in_sequence(t, order as usize));
                assert_eq!(got, row[table], "position {t} table {table}");
            }
        }
        for (j, row) in rows(&case["tap_ok"]).iter().enumerate() {
            for (t, &want) in row.iter().enumerate() {
                assert_eq!(
                    u32::from(tap_in_sequence(t, j, dilation)),
                    want,
                    "tap {j} at {t}"
                );
            }
        }
    }
}

// Measured (2026-10-24): 15 of 18 clip-sites at 3.0e-7 to 3.8e-7 for both keys and values (f32
// rounding). es_monica site 0 (keys 1.62e-4, values 1.82e-4), fr_thomas site 0 (2.04e-4,
// 2.31e-4) and fr_thomas site 1 (1.58e-4, 2.03e-4) each carry one position at about 1e-3, all
// of it one A8 step in the gathered row: rebuilt from upstream's ops on the archive's tables, the
// reference row is reproduced to 4e-7, and moving that one element by one step gives exactly
// the port's error (es_monica t = 12, element 41: 1.186e-3). The cause is upstream of this
// stage, in dequantization: upstream's `read_export` (which wrote the fixture's tables)
// multiplies each group by the Hadamard matrix densely (`rot @ H`), while this crate's `quant`
// module uses a fast Walsh-Hadamard transform (FWHT). The two sum in different orders, so 386
// of that row's 512 values differ by an ulp or so, and element 41 lands at x / scale =
// -0.49999145 in this crate where upstream has -0.50000095.

/// Relative L2 against `decoder/engram_k`, worst clip and site. Measured when this landed:
/// 2.04e-4 (fr_thomas site 0); the bound leaves room for about one more flipped step.
const KEYS_BOUND: f64 = 5e-4;
/// Relative L2 against `decoder/engram_v`, worst clip and site. Measured when this landed:
/// 2.31e-4 (fr_thomas site 0); the bound leaves room for about one more flipped step.
const VALUES_BOUND: f64 = 5e-4;

#[test]
fn engram_keys_and_values_reproduce_the_reference_on_every_clip() {
    let w = weights();
    let (mut worst_k, mut worst_v) = ((0.0f64, String::new()), (0.0f64, String::new()));
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let tokens = f.tokens();
        let (k_shape, want_k) = f.f32("decoder/engram_k");
        let (_, want_v) = f.f32("decoder/engram_v");
        assert_eq!(k_shape, vec![w.engrams.len(), tokens.len(), 512], "{clip}");
        let per_site = k_shape[1] * k_shape[2];
        for (s, engram) in w.engrams.iter().enumerate() {
            let (k, v) = engram.keys_values(&tokens).unwrap();
            assert_eq!((k.rows(), k.cols()), (tokens.len(), 512), "{clip}");
            let site = s * per_site..(s + 1) * per_site;
            let ek = relative_l2(k.data(), &want_k[site.clone()]);
            let ev = relative_l2(v.data(), &want_v[site]);
            println!("{clip} site {s}: keys {ek:.3e}, values {ev:.3e}");
            let name = format!("{clip} site {s}");
            if ek > worst_k.0 {
                worst_k = (ek, name.clone());
            }
            if ev > worst_v.0 {
                worst_v = (ev, name);
            }
        }
    }
    println!(
        "worst keys {:.3e} ({}), values {:.3e} ({})",
        worst_k.0, worst_k.1, worst_v.0, worst_v.1
    );
    assert!(
        worst_k.0 <= KEYS_BOUND,
        "keys {:.3e} on {}",
        worst_k.0,
        worst_k.1
    );
    assert!(
        worst_v.0 <= VALUES_BOUND,
        "values {:.3e} on {}",
        worst_v.0,
        worst_v.1
    );
}

/// Relative L2 of layer 3's `ZCRMSNorm_0` output, from the fixture's block input and Engram
/// keys and values. There is no A8 step in the add or the norm, so this is an exact stage.
/// Measured when this landed: 1.78e-7.
const ADD_BOUND: f64 = 1e-6;

#[test]
fn engram_add_then_norm_in_reproduces_decoder_layer3() {
    let w = weights();
    let f = Fixture::load("en_samantha");
    let (_, x) = f.f32("decoder/layer03/block_in");
    let (_, k) = f.f32("decoder/engram_k");
    let (_, v) = f.f32("decoder/engram_v");
    let (_, want) = f.f32("decoder/layer03/block/ZCRMSNorm_0/0");
    // Layer 3 is the first Engram site: flags [1, 0], as `LaneStack` builds them.
    let sites: Vec<f32> = w
        .geometry
        .engram_sites
        .iter()
        .map(|&l| if l == 3 { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(sites, vec![1.0, 0.0]);
    let added = engram_add(&x, &k, &v, &sites, 512).unwrap();
    let got = w.decoder[3].norm_in.forward(&added);
    let err = relative_l2(&got, &want);
    println!("decoder layer03 Engram add + norm_in: relative L2 {err:.3e}");
    assert!(err <= ADD_BOUND, "relative L2 {err:.3e}");
}

#[test]
fn engram_add_with_no_site_flagged_is_the_identity_and_bad_shapes_are_errors() {
    let x: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
    let k = vec![1.0f32; 16];
    let v = vec![2.0f32; 16];
    assert_eq!(engram_add(&x, &k, &v, &[0.0, 0.0], 4).unwrap(), x);
    for (k, v, flags, width) in [
        (&k[..15], &v[..], &[0.0f32, 0.0][..], 4),
        (&k[..], &v[..8], &[0.0, 0.0][..], 4),
        (&k[..], &v[..], &[0.0][..], 4),
        (&k[..], &v[..], &[0.0, 0.0][..], 3),
    ] {
        match engram_add(&x, k, v, flags, width) {
            Err(whistle::Error::Input(m)) => assert!(!m.is_empty()),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn engram_indices_with_an_overflowing_size_are_input_errors() {
    match engram_indices(&[1, 2, 3], &[2, 3], usize::MAX, 18432, 0) {
        Err(Error::Input(m)) => assert!(!m.is_empty()),
        other => panic!("{other:?}"),
    }
    assert!(!tap_in_sequence(usize::MAX, 2, usize::MAX));
}

/// The archive with u32 header fields (index into the 49-field header) set as given.
fn patched_header(fields: &[(usize, u32)]) -> Vec<u8> {
    let mut bytes = model_bytes();
    for &(field, value) in fields {
        bytes[field * 4..field * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn a_header_whose_engram_hashing_is_not_whistles_is_rejected_at_bind() {
    // Header field 24 is engram_conv_dilation (Whistle: max(orders) = 3), 25 engram_seed_heads
    // (Whistle: 0, which hashes the same as 2, the hash heads per order).
    assert_eq!(weights().geometry.engram_conv_dilation, 3);
    for (fields, what) in [
        (vec![(24, 2)], "engram_conv_dilation"),
        (vec![(24, 4)], "engram_conv_dilation"),
        (vec![(25, 1)], "engram_seed_heads"),
        (vec![(25, 3)], "engram_seed_heads"),
    ] {
        let bytes = patched_header(&fields);
        let archive = Archive::parse(&bytes).unwrap();
        let err = Weights::from_archive(&archive).unwrap_err();
        assert!(
            matches!(&err, Error::Archive(m) if m.contains(what)),
            "{fields:?}: {err}"
        );
    }
    // seed_heads = heads is the reference's own default (`seed_heads or heads`), so it binds and
    // hashes exactly as 0 does.
    let bytes = patched_header(&[(25, 2)]);
    let archive = Archive::parse(&bytes).unwrap();
    let w = Weights::from_archive(&archive).unwrap();
    assert_eq!(w.engrams[0].hash, weights().engrams[0].hash);
}

#[test]
fn engram_weights_whose_shapes_do_not_fit_are_archive_errors() {
    let w = weights();
    let tokens = [2u32, 8192, 300, 1];
    let base = &w.engrams[0];
    // Each a Matrix of the wrong shape from elsewhere in the model: the mel bank [257, 80],
    // a decoder query-tap kernel [taps, 480].
    let wrong_taps = w.decoder[0].self_attn.taps.as_ref().unwrap().q.clone();
    assert_ne!(wrong_taps.cols(), 512);
    let mut cases: Vec<Engram> = Vec::new();
    let mut e = base.clone();
    e.key_proj = w.mel.clone();
    cases.push(e);
    let mut e = base.clone();
    e.value_proj = w.mel.clone();
    cases.push(e);
    let mut e = base.clone();
    e.taps = wrong_taps;
    cases.push(e);
    let mut e = base.clone();
    e.hash.slots += 1;
    cases.push(e);
    let mut e = base.clone();
    e.hash.heads = 3;
    cases.push(e);
    for (i, e) in cases.iter().enumerate() {
        match e.keys_values(&tokens) {
            Err(Error::Archive(m)) => assert!(!m.is_empty()),
            other => panic!("case {i}: {other:?}"),
        }
    }
    assert!(base.keys_values(&tokens).is_ok());
}
