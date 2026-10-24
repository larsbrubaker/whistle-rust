// whistle.cact's header geometry, stored codebooks and tensor directory equal what upstream's
// `read_export` reads (the archive fixture's manifest).

use serde_json::Value;
use whistle::cact::{Archive, DType};

use crate::support::archive::{manifest, model_bytes};

fn u(value: &Value) -> u64 {
    value.as_u64().expect("unsigned integer")
}

fn list(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("list")
        .iter()
        .map(|v| u(v) as u32)
        .collect()
}

#[test]
fn geometry_equals_the_header_upstream_reads() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).expect("whistle.cact parses");
    let g = archive.geometry();
    let m = &manifest()["geometry"];
    assert_eq!(u(&m["tag"]), 0x05E1_2A84);
    let fields: [(&str, u64); 25] = [
        ("num_tensors", g.num_tensors.into()),
        ("codebook_len", g.codebook_len.into()),
        ("kv_window", g.kv_window.into()),
        ("kv_bits", g.kv_bits.into()),
        ("vocab", g.vocab.into()),
        ("out_vocab", g.out_vocab.into()),
        ("d_model", g.d_model.into()),
        ("num_heads", g.num_heads.into()),
        ("num_kv_heads", g.num_kv_heads.into()),
        ("num_layers", g.num_layers.into()),
        ("qk_head_dim", g.qk_head_dim.into()),
        ("v_head_dim", g.v_head_dim.into()),
        ("max_seq_len", g.max_seq_len.into()),
        ("hada_n", g.hada_n.into()),
        ("mhc_lanes", g.mhc_lanes.into()),
        ("sliding_window", g.sliding_window.into()),
        ("global_mask", g.global_mask),
        ("qkv_conv_taps", g.qkv_conv_taps.into()),
        ("engram_slots", g.engram_slots.into()),
        ("engram_sub_dim", g.engram_sub_dim.into()),
        ("num_engram_tables", g.num_engram_tables.into()),
        ("engram_conv_taps", g.engram_conv_taps.into()),
        ("engram_conv_dilation", g.engram_conv_dilation.into()),
        ("engram_seed_heads", g.engram_seed_heads.into()),
        ("num_tensors", archive.tensors().len() as u64),
    ];
    for (name, value) in fields {
        assert_eq!(value, u(&m[name]), "{name}");
    }
    assert_eq!(g.engram_orders, list(&m["engram_orders"]));
    assert_eq!(g.engram_sites, list(&m["engram_sites"]));
    assert_eq!(g.rope_theta, m["rope_theta"].as_f64().unwrap() as f32);
}

#[test]
fn stored_codebooks_equal_upstreams() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    // The manifest holds f32 values printed as f64; JSON parsing may be an f64 ulp off, so
    // compare after rounding back to f32 (exact for a value that was an f32).
    let expected: Vec<f32> = manifest()["codebook"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let stored: Vec<f32> = [2, 3, 4]
        .iter()
        .flat_map(|&bits| archive.codebook(bits).expect("stored width").to_vec())
        .collect();
    assert_eq!(stored, expected);
    assert_eq!(archive.codebook(2).unwrap().len(), 4);
    assert_eq!(archive.codebook(4).unwrap().len(), 16);
    assert!(archive.codebook(1).is_none() && archive.codebook(5).is_none());
}

#[test]
fn every_record_matches_upstreams_directory() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    let m = manifest();
    let entries = m["tensors"].as_array().unwrap();
    assert_eq!(archive.tensors().len(), entries.len());
    for (i, (tensor, entry)) in archive.tensors().iter().zip(entries).enumerate() {
        let name = entry["name"].as_str().unwrap();
        assert_eq!(u(&entry["index"]) as usize, i);
        let dtype = match entry["dtype"].as_str().unwrap() {
            "fp16" => DType::Fp16,
            "fp32" => DType::Fp32,
            "cq" => DType::Cq,
            "raw" => DType::Raw,
            other => panic!("{name}: dtype {other}"),
        };
        assert_eq!(tensor.dtype(), dtype, "{name}");
        let shape: Vec<usize> = list(&entry["shape"]).iter().map(|&d| d as usize).collect();
        assert_eq!(tensor.shape(), shape.as_slice(), "{name}");
        assert_eq!(u64::from(tensor.group()), u(&entry["group"]), "{name}");
        assert_eq!(u64::from(tensor.bits()), u(&entry["bits"]), "{name}");
        if dtype == DType::Raw {
            assert_eq!(tensor.bytes().len() as u64, u(&entry["nbytes"]), "{name}");
        }
    }
    assert!(archive.tensor(entries.len()).is_none());
}
