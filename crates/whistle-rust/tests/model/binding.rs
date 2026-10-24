// Binding: every archive tensor gets upstream's export name (test-data/fixtures/archive/
// manifest.json, from `export._tensors` + `_audio_tensors`), every bound struct field has the
// shape and the values of the tensor upstream exports under that name, the packed Engram tables'
// rows reproduce upstream's sampled values, and archives that do not fit the geometry (including
// hostile header sizes) are errors, never panics.

use std::borrow::Cow;
use std::collections::BTreeMap;

use whistle::cact::Archive;
use whistle::model::{HadamardMlp, Matrix, Mhc, Norm, Weights};
use whistle::Error;

use crate::shared::weights;
use crate::support::archive::{build, manifest, model_bytes, Record};
use crate::support::relative_l2;

fn manifest_shapes() -> Vec<(String, Vec<usize>)> {
    manifest()["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let shape = t["shape"].as_array().unwrap();
            let shape = shape.iter().map(|v| v.as_u64().unwrap() as usize).collect();
            (t["name"].as_str().unwrap().to_string(), shape)
        })
        .collect()
}

#[test]
fn every_tensor_index_has_upstreams_export_name() {
    let names: Vec<String> = manifest_shapes().into_iter().map(|(n, _)| n).collect();
    assert_eq!(weights().tensor_names(), names.as_slice());
}

/// Each bound struct field's shape and values, keyed by the export name the field claims.
struct Shapes<'w>(BTreeMap<String, (Vec<usize>, Cow<'w, [f32]>)>);

impl<'w> Shapes<'w> {
    fn raw(&mut self, name: String, shape: Vec<usize>, data: Cow<'w, [f32]>) {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        assert!(
            self.0.insert(name.clone(), (shape, data)).is_none(),
            "{name} bound twice"
        );
    }
    fn m(&mut self, name: String, m: &'w Matrix) {
        self.raw(name, vec![m.rows(), m.cols()], m.data());
    }
    fn v(&mut self, name: String, v: &'w [f32]) {
        self.raw(name, vec![v.len()], Cow::Borrowed(v));
    }
    fn n(&mut self, name: String, n: &'w Norm) {
        self.v(name, &n.scale);
    }
    fn scalar(&mut self, name: String, value: f32) {
        self.raw(name, vec![1], Cow::Owned(vec![value]));
    }
    fn hada(&mut self, p: &str, h: &'w HadamardMlp) {
        for (k, v) in [
            ("d1", &h.d1),
            ("d2", &h.d2),
            ("b2", &h.b2),
            ("d3", &h.d3),
            ("d4", &h.d4),
        ] {
            self.v(format!("{p}.{k}"), v);
        }
        for (s, (a, b)) in h.factors.iter().enumerate() {
            self.m(format!("{p}.w{}a", s + 1), a);
            self.m(format!("{p}.w{}b", s + 1), b);
        }
        self.m(format!("{p}.cond_v"), &h.cond_v);
        self.m(format!("{p}.cond_u"), &h.cond_u);
    }
    fn mhc(&mut self, p: &str, h: &'w Mhc) {
        let (l, n) = (h.a_pre.len(), h.lanes);
        for (k, v) in [
            ("a_pre", &h.a_pre),
            ("a_post", &h.a_post),
            ("a_res", &h.a_res),
        ] {
            self.v(format!("{p}mhc_{k}"), v);
        }
        self.raw(format!("{p}mhc_b_pre"), vec![l, n], Cow::Borrowed(&h.b_pre));
        self.raw(
            format!("{p}mhc_b_post"),
            vec![l, n],
            Cow::Borrowed(&h.b_post),
        );
        self.raw(
            format!("{p}mhc_b_res"),
            vec![l, n, n],
            Cow::Borrowed(&h.b_res),
        );
        self.m(format!("{p}mhc_phi_pre"), &h.phi_pre);
        self.m(format!("{p}mhc_phi_post"), &h.phi_post);
        self.m(format!("{p}mhc_phi_res"), &h.phi_res);
    }
}

fn bound_fields(w: &Weights) -> Shapes<'_> {
    let mut s = Shapes(BTreeMap::new());
    s.m("embedding".into(), &w.embedding);
    for (i, l) in w.decoder.iter().enumerate() {
        let p = format!("layer{i:02}");
        let a = &l.self_attn;
        s.n(format!("{p}.norm_in"), &l.norm_in);
        s.m(format!("{p}.q_proj"), &a.q_proj);
        s.m(format!("{p}.k_proj"), &a.k_proj);
        s.m(format!("{p}.v_proj"), &a.v_proj);
        let taps = a
            .taps
            .as_ref()
            .expect("decoder self-attention has QKV taps");
        s.m(format!("{p}.q_taps"), &taps.q);
        s.m(format!("{p}.k_taps"), &taps.k);
        s.m(format!("{p}.v_taps"), &taps.v);
        s.n(format!("{p}.q_norm"), &a.q_norm);
        s.n(format!("{p}.k_norm"), &a.k_norm);
        s.m(format!("{p}.gate_proj"), &a.gate_proj);
        s.m(format!("{p}.out_proj"), &a.out_proj);
        s.n(format!("{p}.post_norm"), &l.post_attn_norm);
        s.scalar(format!("{p}.attn_gate"), l.attn_gate);
        s.n(format!("{p}.pre_hada"), &l.pre_hada_norm);
        s.hada(&p, &l.hadamard_mlp);
        let c = &l.cross_attn;
        s.n(format!("{p}.cross_norm"), &l.cross_norm);
        s.m(format!("{p}.cross.q_proj"), &c.q_proj);
        s.m(format!("{p}.cross.k_proj"), &c.k_proj);
        s.m(format!("{p}.cross.v_proj"), &c.v_proj);
        s.m(format!("{p}.cross.gate_proj"), &c.gate_proj);
        s.m(format!("{p}.cross.out_proj"), &c.out_proj);
        s.n(format!("{p}.cross_q_norm"), &c.q_norm);
        s.n(format!("{p}.cross_k_norm"), &c.k_norm);
        s.n(format!("{p}.post_cross_norm"), &l.post_cross_norm);
        s.scalar(format!("{p}.cross_gate"), l.cross_gate);
    }
    s.mhc("", &w.decoder_mhc);
    for (name, perm) in [("hada_p1", &w.hada_perms.p1), ("hada_p2", &w.hada_perms.p2)] {
        let values = perm.iter().map(|&i| i as f32).collect();
        s.raw(name.into(), vec![perm.len()], Cow::Owned(values));
    }
    for (i, e) in w.engrams.iter().enumerate() {
        s.raw(
            format!("engram{i}.tables"),
            vec![e.tables.rows(), e.tables.cols()],
            Cow::Owned(e.tables.to_f32().unwrap()),
        );
        s.m(format!("engram{i}.key_proj"), &e.key_proj);
        s.m(format!("engram{i}.value_proj"), &e.value_proj);
        s.m(format!("engram{i}.taps"), &e.taps);
    }
    s.n("final_norm".into(), &w.final_norm);
    let a = w.audio;
    let manifest = [
        3246,
        a.version,
        a.text,
        a.enc_layers,
        a.enc_lanes,
        a.conv_kernel,
        a.stem,
        a.stem_stages,
        a.mels,
        a.rate,
        a.fft,
        a.win,
        a.hop,
    ];
    let manifest = manifest.iter().map(|&v| v as f32).collect();
    s.raw("audio.manifest".into(), vec![13], Cow::Owned(manifest));
    s.scalar("pe_gate".into(), w.pe_gate);
    for (i, l) in w.encoder.iter().enumerate() {
        let p = format!("encoder{i:02}");
        let a = &l.self_attn;
        assert!(a.taps.is_none(), "{p}: the encoder has no QKV taps");
        for (k, n) in [
            ("pre_hada_norm_0", &l.pre_hada_norm_0),
            ("ZCRMSNorm_0", &l.attn_norm),
            ("post_attn_norm", &l.post_attn_norm),
            ("conv_norm", &l.conv_norm),
            ("conv_out_norm", &l.conv_out_norm),
            ("pre_hada_norm", &l.pre_hada_norm),
            ("q_norm", &a.q_norm),
            ("k_norm", &a.k_norm),
        ] {
            s.n(format!("{p}.{k}"), n);
        }
        for (k, m) in [
            ("q_proj", &a.q_proj),
            ("k_proj", &a.k_proj),
            ("v_proj", &a.v_proj),
            ("gate_proj", &a.gate_proj),
            ("out_proj", &a.out_proj),
            ("pw1", &l.pw1),
            ("pw2", &l.pw2),
            ("dw", &l.dw),
        ] {
            s.m(format!("{p}.{k}"), m);
        }
        s.scalar(format!("{p}.attn_gate"), l.attn_gate);
        s.hada(&format!("{p}.hadamard_mlp_0"), &l.hadamard_mlp_0);
        s.hada(&format!("{p}.hadamard_mlp"), &l.hadamard_mlp);
    }
    s.mhc("encoder.", &w.encoder_mhc);
    s.n("encoder.final_norm".into(), &w.encoder_final_norm);
    s.m("stem.w".into(), &w.stem.w);
    for (i, st) in w.stem.stages.iter().enumerate() {
        s.m(format!("stem.dw_{}", i + 1), &st.dw);
        s.m(format!("stem.pw_{}", i + 1), &st.pw);
    }
    s.m("stem.out".into(), &w.stem.out);
    s.m("mel".into(), &w.mel);
    s
}

#[test]
fn every_bound_field_has_the_exported_shape() {
    let bound: BTreeMap<String, Vec<usize>> = bound_fields(weights())
        .0
        .into_iter()
        .map(|(name, (shape, _))| (name, shape))
        .collect();
    let expected: BTreeMap<String, Vec<usize>> = manifest_shapes()
        .into_iter()
        .filter(|(n, _)| n != "tokenizer")
        .collect();
    assert_eq!(bound, expected);
}

/// Names and shapes alone would let two same-shape fields swap roles (q_norm and k_norm, the
/// encoder's six norms, phi_pre and phi_post, cross q and k): every field must hold, bit for
/// bit, the values of the archive tensor at its claimed name's index in upstream's export order.
#[test]
fn every_bound_field_holds_its_named_tensors_values() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    let bound = bound_fields(weights()).0;
    for (index, (name, _)) in manifest_shapes().iter().enumerate() {
        if name == "tokenizer" {
            continue;
        }
        let want = archive.tensor(index).unwrap().to_f32(&archive).unwrap();
        let (_, got) = &bound[name];
        let same = got.len() == want.len()
            && got
                .iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "{name} (tensor {index}) holds another tensor's values"
        );
    }
}

#[test]
fn audio_manifest_and_permutations_are_parsed() {
    let w = weights();
    let a = w.audio;
    assert_eq!(
        (a.version, a.text, a.enc_layers, a.enc_lanes),
        (1, 8192, 8, 4)
    );
    assert_eq!(
        (a.conv_kernel, a.stem, a.stem_stages, a.mels),
        (9, 128, 3, 80)
    );
    assert_eq!((a.rate, a.fft, a.win, a.hop), (16000, 512, 400, 160));
    for p in [&w.hada_perms.p1, &w.hada_perms.p2] {
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..512).collect::<Vec<_>>());
    }
    // test-data/fixtures/archive/226_hada_p1.npy holds upstream's permutation.
    let crate::support::Npy::F32 { data, .. } = crate::support::archive::array("226_hada_p1.npy")
    else {
        panic!("hada_p1 fixture is float32");
    };
    let want: Vec<usize> = data.iter().map(|&v| v as usize).collect();
    assert_eq!(w.hada_perms.p1, want);
}

/// `CqRows::row` against the oracle: the archive fixture records 64 sampled flat indices and
/// upstream's `read_export` values for every tensor, so each Engram table's sampled elements are
/// read back row by row (relative L2 1e-6, the archive area's exact bound; measured 1.8e-7 and
/// 2.0e-7). Out-of-range rows and wrong buffer sizes are errors.
#[test]
fn packed_engram_rows_match_upstreams_sampled_values() {
    let m = manifest();
    for (site, engram) in weights().engrams.iter().enumerate() {
        let entry = m["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == format!("engram{site}.tables"))
            .unwrap();
        let cols = engram.tables.cols();
        let mut row = vec![0.0f32; cols];
        let (mut got, mut want) = (Vec::new(), Vec::new());
        let indices = entry["sample_indices"].as_array().unwrap();
        for (i, v) in indices.iter().zip(entry["samples"].as_array().unwrap()) {
            let flat = i.as_u64().unwrap() as usize;
            engram.tables.row_into(flat / cols, &mut row).unwrap();
            got.push(row[flat % cols]);
            want.push(v.as_f64().unwrap() as f32);
        }
        let err = relative_l2(&got, &want);
        println!("engram{site}.tables sampled rows: relative L2 {err:.3e}");
        assert!(err <= 1e-6, "engram{site}.tables: relative L2 {err:.3e}");
        let rows = engram.tables.rows();
        assert!(matches!(engram.tables.row(rows), Err(Error::Archive(_))));
        assert!(matches!(
            engram.tables.row_into(0, &mut row[1..]),
            Err(Error::Archive(_))
        ));
    }
}

#[test]
fn an_archive_that_does_not_fit_the_geometry_is_an_error() {
    let codebook = [0.0f32; 28];
    let lone = Record {
        dtype: 1,
        shape: vec![4],
        group: 0,
        bits: 0,
        blob: vec![0; 8],
    };
    let bytes = build(&codebook, &[lone]);
    let archive = Archive::parse(&bytes).unwrap();
    let err = Weights::from_archive(&archive).unwrap_err();
    assert!(matches!(err, Error::Archive(_)), "{err}");
}

#[test]
fn probe_heads_after_the_final_norm_are_unsupported() {
    // Rewrite the directory record after final_norm (the audio manifest, fp32 [13]) as an fp16
    // vector of the same byte size, the shape a probe-head manifest (`heads.manifest`) has.
    let mut bytes = model_bytes();
    let index = weights()
        .tensor_names()
        .iter()
        .position(|n| n == "audio.manifest")
        .unwrap();
    let record = 49 * 4 + 28 * 4 + 44 * index;
    assert_eq!(bytes[record], 2, "audio.manifest is fp32");
    bytes[record] = 1;
    bytes[record + 4..record + 8].copy_from_slice(&26u32.to_le_bytes());
    let archive = Archive::parse(&bytes).unwrap();
    let err = Weights::from_archive(&archive).unwrap_err();
    assert!(
        err.to_string().contains("probe heads are not supported"),
        "{err}"
    );
}

/// The archive with header field `field` (u32 index into the 49-field header) set to `value`.
fn patched_header(fields: &[(usize, u32)]) -> Vec<u8> {
    let mut bytes = model_bytes();
    for &(field, value) in fields {
        bytes[field * 4..field * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn huge_header_geometry_is_an_error_not_a_panic() {
    // Header fields: 5 vocab, 7 d_model, 8 heads, 9 kv_heads, 10 layers, 11 qk_dim, 12 v_dim,
    // 14 hada_n, 15 mhc_lanes, 19 qkv taps, 20 slots, 21 sub_dim, 22 tables, 23 engram taps.
    // Products of these overflow a 32-bit usize (wasm32), and mhc_lanes³-sized ones a 64-bit one.
    let max = u32::MAX;
    let mut cases: Vec<Vec<(usize, u32)>> = [5, 7, 8, 9, 10, 11, 12, 14, 15, 19, 20, 21, 22, 23]
        .iter()
        .map(|&f| vec![(f, max)])
        .collect();
    cases.push(vec![(7, 0x8000_0001), (14, 0)]);
    cases.push(vec![(8, max), (11, max), (9, max), (12, max)]);
    cases.push(vec![(20, max), (22, max), (21, max)]);
    cases.push(vec![(10, max), (15, max)]);
    for fields in cases {
        let bytes = patched_header(&fields);
        let archive = Archive::parse(&bytes).unwrap();
        let err = Weights::from_archive(&archive).unwrap_err();
        assert!(matches!(err, Error::Archive(_)), "{fields:?}: {err}");
    }
}

#[test]
fn an_archive_with_an_unported_kv_width_is_rejected() {
    // Header field 4 is kv_bits; Whistle ships 8. Width 4 is upstream's codebook KV path.
    for kv_bits in [4u32, 16] {
        let bytes = patched_header(&[(4, kv_bits)]);
        let archive = Archive::parse(&bytes).unwrap();
        let err = Weights::from_archive(&archive).unwrap_err();
        assert!(
            matches!(&err, Error::Archive(m) if m.contains("kv_bits")),
            "{kv_bits}: {err}"
        );
    }
}
