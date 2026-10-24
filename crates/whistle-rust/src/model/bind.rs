//! Binding the archive's nameless tensors to their roles. The archive stores tensors in the order
//! upstream's `export._tensors` enumerates them (embedding, decoder layers, decoder mHC, Hadamard
//! permutations, Engram sites, final norm, optional probe heads), then `_audio_tensors` (the
//! audio manifest, decoder cross-attention, `pe_gate`, encoder layers, encoder mHC and final
//! norm, stem, mel matrix), then the tokenizer. [`Reader`] walks that order, checking each
//! tensor's dtype and shape against what the geometry implies and recording its upstream name;
//! `bind` assembles the [`Weights`] structs defined in `weights`.

use crate::cact::{Archive, DType, Geometry, Tensor};
use crate::error::Error;
use crate::quant::CqRows;

use super::weights::{
    AudioManifest, CrossAttention, DecoderLayer, EncoderLayer, Engram, HadaPerms, HadamardMlp,
    Matrix, Mhc, Norm, QkvTaps, SelfAttention, Stem, StemStage, Weights,
};

/// `architecture.HADA_COND_RANK`.
const HADA_COND_RANK: usize = 8;
/// `export.AUDIO_MANIFEST_MAGIC`, the audio manifest's first value.
const AUDIO_MANIFEST_MAGIC: f32 = 3246.0;
/// The audio manifest's length and the version this binding reads.
const AUDIO_MANIFEST_LEN: usize = 13;
const AUDIO_MANIFEST_VERSION: u32 = 1;

/// Walks the archive's tensors in order, checking and naming each.
struct Reader<'r, 'a> {
    archive: &'r Archive<'a>,
    names: Vec<String>,
}

impl<'r, 'a> Reader<'r, 'a> {
    /// The next tensor, which must be `dtype` with exactly `shape`.
    fn take(
        &mut self,
        name: String,
        dtype: DType,
        shape: &[usize],
    ) -> Result<&'r Tensor<'a>, Error> {
        let index = self.names.len();
        let archive: &'r Archive<'a> = self.archive;
        let tensor = archive.tensor(index).ok_or_else(|| {
            Error::Archive(format!(
                "the archive ends at tensor {index}, where the geometry expects {name}"
            ))
        })?;
        if tensor.dtype() != dtype || tensor.shape() != shape {
            return Err(Error::Archive(format!(
                "tensor {index} should be {name}, {dtype:?} {shape:?}, but is {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            )));
        }
        self.names.push(name);
        Ok(tensor)
    }

    fn fp16(&mut self, name: String, shape: &[usize]) -> Result<Vec<f32>, Error> {
        self.take(name, DType::Fp16, shape)?.to_f32(self.archive)
    }

    fn vector(&mut self, name: String, len: usize) -> Result<Vec<f32>, Error> {
        self.fp16(name, &[len])
    }

    fn norm(&mut self, name: String, len: usize) -> Result<Norm, Error> {
        Ok(Norm {
            scale: self.vector(name, len)?,
        })
    }

    fn scalar(&mut self, name: String) -> Result<f32, Error> {
        Ok(self.fp16(name, &[1])?[0])
    }

    fn fp16_matrix(&mut self, name: String, rows: usize, cols: usize) -> Result<Matrix, Error> {
        let data = self.fp16(name, &[rows, cols])?;
        Ok(Matrix { rows, cols, data })
    }

    /// A CQ matrix, dequantized.
    fn matrix(&mut self, name: String, rows: usize, cols: usize) -> Result<Matrix, Error> {
        let data = self
            .take(name, DType::Cq, &[rows, cols])?
            .to_f32(self.archive)?;
        Ok(Matrix { rows, cols, data })
    }

    /// A CQ matrix, kept packed.
    fn cq_rows(&mut self, name: String, rows: usize, cols: usize) -> Result<CqRows, Error> {
        let tensor = self.take(name, DType::Cq, &[rows, cols])?;
        CqRows::from_tensor(tensor, self.archive)
    }

    fn fp32(&mut self, name: String, shape: &[usize]) -> Result<Vec<f32>, Error> {
        self.take(name, DType::Fp32, shape)?.to_f32(self.archive)
    }

    fn permutation(&mut self, name: String, n: usize) -> Result<Vec<usize>, Error> {
        let values = self.fp32(name.clone(), &[n])?;
        let mut seen = vec![false; n];
        let mut perm = Vec::with_capacity(n);
        for v in values {
            // Every index below 2^24 is exact in f32; anything else is not a permutation.
            let ok = v.fract() == 0.0 && v >= 0.0 && (v as usize) < n;
            if !ok || seen[v as usize] {
                return Err(Error::Archive(format!(
                    "{name} is not a permutation of 0..{n}"
                )));
            }
            seen[v as usize] = true;
            perm.push(v as usize);
        }
        Ok(perm)
    }
}

/// The model widths every binder needs, from the header, each product checked: the header is
/// untrusted, and a product that overflows (easily, on wasm32's 32-bit `usize`) is an error.
struct Dims {
    d: usize,
    d2: usize,
    qk: usize,
    /// `heads · qk_dim`, `kv_heads · qk_dim`, `kv_heads · v_dim`, `heads · v_dim`.
    q_width: usize,
    kv_qk_width: usize,
    kv_v_width: usize,
    v_width: usize,
    taps: usize,
    hada_n: usize,
    hada_a: usize,
    hada_b: usize,
}

/// `a · b`, or an archive error naming what overflowed.
fn mul(a: usize, b: usize, what: &str) -> Result<usize, Error> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Archive(format!("{what} ({a} × {b}) overflows")))
}

impl Dims {
    fn new(g: &Geometry) -> Result<Dims, Error> {
        let d = g.d_model as usize;
        let padded = d.checked_next_power_of_two();
        if d == 0 || padded != Some(g.hada_n as usize) {
            return Err(Error::Archive(format!(
                "hada_n {} is not d_model {d} padded to a power of two",
                g.hada_n
            )));
        }
        let (heads, kv_heads) = (g.num_heads as usize, g.num_kv_heads as usize);
        let (qk, v) = (g.qk_head_dim as usize, g.v_head_dim as usize);
        let (hada_a, hada_b) = hada_blocks(g.hada_n as usize);
        Ok(Dims {
            d,
            d2: mul(2, d, "2 · d_model")?,
            qk,
            q_width: mul(heads, qk, "heads · qk_dim")?,
            kv_qk_width: mul(kv_heads, qk, "kv_heads · qk_dim")?,
            kv_v_width: mul(kv_heads, v, "kv_heads · v_dim")?,
            v_width: mul(heads, v, "heads · v_dim")?,
            taps: g.qkv_conv_taps as usize,
            hada_n: g.hada_n as usize,
            hada_a,
            hada_b,
        })
    }
}

/// `_hada_blocks`: `n = a · b` with `a = 2^(bit_length(n - 1) / 2)`.
fn hada_blocks(n: usize) -> (usize, usize) {
    let bit_length = usize::BITS - (n - 1).leading_zeros();
    let a = 1usize << (bit_length / 2);
    (a, n / a)
}

/// The KV-path widths the port runs: 0 (no KV quantization) and 8 (per-head A8). Upstream's
/// `maybe_quant_kv` treats any width of 8 or more as A8 and anything from 1 to 7 as its
/// codebook KV quantization (`cq_fake_quant_kv`), which is not ported; this narrows to the two
/// widths an exported archive uses (Whistle ships 8), so a wider one is rejected rather than
/// silently run as A8.
pub(crate) fn check_kv_bits(kv_bits: u32) -> Result<(), Error> {
    if kv_bits == 0 || kv_bits == 8 {
        Ok(())
    } else {
        Err(Error::Archive(format!(
            "kv_bits {kv_bits} is not supported (only 0 and 8)"
        )))
    }
}

pub(crate) fn bind(archive: &Archive<'_>) -> Result<Weights, Error> {
    let g = archive.geometry().clone();
    check_kv_bits(g.kv_bits)?;
    let dims = Dims::new(&g)?;
    let d = dims.d;
    let mut r = Reader {
        archive,
        names: Vec::with_capacity(archive.tensors().len()),
    };

    let embedding = r.matrix("embedding".into(), g.vocab as usize, d)?;
    let mut decoder_core = Vec::new();
    for i in 0..g.num_layers {
        decoder_core.push(decoder_core_layer(&mut r, &dims, &format!("layer{i:02}"))?);
    }
    let decoder_mhc = mhc(&mut r, "", g.num_layers as usize, g.mhc_lanes as usize, d)?;
    let hada_perms = HadaPerms {
        p1: r.permutation("hada_p1".into(), dims.hada_n)?,
        p2: r.permutation("hada_p2".into(), dims.hada_n)?,
    };
    let tables = g.num_engram_tables as usize;
    let sub_dim = g.engram_sub_dim as usize;
    let table_rows = mul(tables, g.engram_slots as usize, "engram tables · slots")?;
    let table_width = mul(tables, sub_dim, "engram tables · sub_dim")?;
    let mut engrams = Vec::new();
    for s in 0..g.engram_sites.len() {
        // Checked per site so an archive without Engram sites needs no hashing geometry.
        let hash = super::engram::hash_geometry(&g)?;
        let p = format!("engram{s}");
        engrams.push(Engram {
            tables: r.cq_rows(format!("{p}.tables"), table_rows, sub_dim)?,
            key_proj: r.matrix(format!("{p}.key_proj"), d, table_width)?,
            value_proj: r.matrix(format!("{p}.value_proj"), d, table_width)?,
            taps: r.fp16_matrix(format!("{p}.taps"), g.engram_conv_taps as usize, d)?,
            hash,
        });
    }
    let final_norm = r.norm("final_norm".into(), d)?;

    let audio = audio_manifest(&mut r)?;
    let mut decoder = Vec::new();
    for (i, core) in decoder_core.into_iter().enumerate() {
        decoder.push(decoder_cross(&mut r, &dims, &format!("layer{i:02}"), core)?);
    }
    let pe_gate = r.scalar("pe_gate".into())?;
    let mut encoder = Vec::new();
    for i in 0..audio.enc_layers {
        let kernel = audio.conv_kernel as usize;
        encoder.push(encoder_layer(
            &mut r,
            &dims,
            &format!("encoder{i:02}"),
            kernel,
        )?);
    }
    let encoder_mhc = mhc(
        &mut r,
        "encoder.",
        audio.enc_layers as usize,
        audio.enc_lanes as usize,
        d,
    )?;
    let encoder_final_norm = r.norm("encoder.final_norm".into(), d)?;
    let stem = stem(&mut r, &audio, d)?;
    let mel_rows = audio.fft as usize / 2 + 1;
    let mel_data = r.fp32("mel".into(), &[mel_rows, audio.mels as usize])?;
    let mel = Matrix {
        rows: mel_rows,
        cols: audio.mels as usize,
        data: mel_data,
    };
    r.take("tokenizer".into(), DType::Raw, &[])?;
    if r.names.len() != archive.tensors().len() {
        return Err(Error::Archive(format!(
            "the archive has {} tensors; the geometry accounts for {}",
            archive.tensors().len(),
            r.names.len()
        )));
    }

    Ok(Weights {
        geometry: g,
        audio,
        embedding,
        decoder,
        decoder_mhc,
        hada_perms,
        engrams,
        final_norm,
        pe_gate,
        encoder,
        encoder_mhc,
        encoder_final_norm,
        stem,
        mel,
        names: r.names,
    })
}

/// The part of a decoder layer `_tensors` writes; cross-attention comes in the audio section.
struct DecoderCore {
    norm_in: Norm,
    self_attn: SelfAttention,
    post_attn_norm: Norm,
    attn_gate: f32,
    pre_hada_norm: Norm,
    hadamard_mlp: HadamardMlp,
}

fn decoder_core_layer(r: &mut Reader<'_, '_>, k: &Dims, p: &str) -> Result<DecoderCore, Error> {
    let norm_in = r.norm(format!("{p}.norm_in"), k.d)?;
    let q_proj = r.matrix(format!("{p}.q_proj"), k.q_width, k.d)?;
    let k_proj = r.matrix(format!("{p}.k_proj"), k.kv_qk_width, k.d)?;
    let v_proj = r.matrix(format!("{p}.v_proj"), k.kv_v_width, k.d)?;
    let taps = if k.taps > 0 {
        Some(QkvTaps {
            q: r.fp16_matrix(format!("{p}.q_taps"), k.taps, k.q_width)?,
            k: r.fp16_matrix(format!("{p}.k_taps"), k.taps, k.kv_qk_width)?,
            v: r.fp16_matrix(format!("{p}.v_taps"), k.taps, k.kv_v_width)?,
        })
    } else {
        None
    };
    let q_norm = r.norm(format!("{p}.q_norm"), k.qk)?;
    let k_norm = r.norm(format!("{p}.k_norm"), k.qk)?;
    let gate_proj = r.matrix(format!("{p}.gate_proj"), k.v_width, k.d)?;
    let out_proj = r.matrix(format!("{p}.out_proj"), k.d, k.v_width)?;
    Ok(DecoderCore {
        norm_in,
        self_attn: SelfAttention {
            q_proj,
            k_proj,
            v_proj,
            taps,
            q_norm,
            k_norm,
            gate_proj,
            out_proj,
        },
        post_attn_norm: r.norm(format!("{p}.post_norm"), k.d)?,
        attn_gate: r.scalar(format!("{p}.attn_gate"))?,
        pre_hada_norm: r.norm(format!("{p}.pre_hada"), k.d)?,
        hadamard_mlp: hadamard_mlp(r, k, p)?,
    })
}

/// The 13 tensors of one Hadamard MLP, named `{p}.d1` … `{p}.cond_u` (`export.HADAMARD`).
fn hadamard_mlp(r: &mut Reader<'_, '_>, k: &Dims, p: &str) -> Result<HadamardMlp, Error> {
    let n = k.hada_n;
    let d1 = r.vector(format!("{p}.d1"), n)?;
    let d2 = r.vector(format!("{p}.d2"), n)?;
    let b2 = r.vector(format!("{p}.b2"), n)?;
    let d3 = r.vector(format!("{p}.d3"), n)?;
    let d4 = r.vector(format!("{p}.d4"), n)?;
    let mut factor = |s: usize| -> Result<(Matrix, Matrix), Error> {
        Ok((
            r.fp16_matrix(format!("{p}.w{s}a"), k.hada_a, k.hada_a)?,
            r.fp16_matrix(format!("{p}.w{s}b"), k.hada_b, k.hada_b)?,
        ))
    };
    let factors = [factor(1)?, factor(2)?, factor(3)?];
    Ok(HadamardMlp {
        d1,
        d2,
        b2,
        d3,
        d4,
        factors,
        cond_v: r.fp16_matrix(format!("{p}.cond_v"), k.d, HADA_COND_RANK)?,
        cond_u: r.fp16_matrix(format!("{p}.cond_u"), HADA_COND_RANK, n)?,
    })
}

/// A lane stack's mHC parameters, names prefixed by `p` (`""` or `"encoder."`).
fn mhc(r: &mut Reader<'_, '_>, p: &str, layers: usize, n: usize, d: usize) -> Result<Mhc, Error> {
    let ln = mul(layers, n, "mHC layers · lanes")?;
    let lnn = mul(ln, n, "mHC layers · lanes²")?;
    let nd = mul(n, d, "mHC lanes · d_model")?;
    Ok(Mhc {
        lanes: n,
        a_pre: r.vector(format!("{p}mhc_a_pre"), layers)?,
        a_post: r.vector(format!("{p}mhc_a_post"), layers)?,
        a_res: r.vector(format!("{p}mhc_a_res"), layers)?,
        b_pre: r.fp16(format!("{p}mhc_b_pre"), &[layers, n])?,
        b_post: r.fp16(format!("{p}mhc_b_post"), &[layers, n])?,
        b_res: r.fp16(format!("{p}mhc_b_res"), &[layers, n, n])?,
        phi_pre: r.matrix(format!("{p}mhc_phi_pre"), ln, nd)?,
        phi_post: r.matrix(format!("{p}mhc_phi_post"), ln, nd)?,
        phi_res: r.matrix(format!("{p}mhc_phi_res"), lnn, nd)?,
    })
}

/// The audio manifest, which must follow the final norm directly: probe heads would sit
/// between them, and Whistle has none, so their presence is unsupported.
fn audio_manifest(r: &mut Reader<'_, '_>) -> Result<AudioManifest, Error> {
    let index = r.names.len();
    let next = r.archive.tensor(index);
    if let Some(t) = next {
        let probe_heads = t.dtype() == DType::Fp16 && t.shape().len() == 1;
        if probe_heads {
            return Err(Error::Archive(format!(
                "tensor {index} after final_norm looks like a probe-head manifest; probe heads \
                 are not supported (Whistle has none)"
            )));
        }
    }
    let values = r.fp32("audio.manifest".into(), &[AUDIO_MANIFEST_LEN])?;
    if values[0] != AUDIO_MANIFEST_MAGIC {
        return Err(Error::Archive(format!(
            "tensor {index} is not the audio manifest (first value {}, expected {AUDIO_MANIFEST_MAGIC}); \
             an archive with probe heads or without the audio section is not supported",
            values[0]
        )));
    }
    let mut field = [0u32; AUDIO_MANIFEST_LEN - 1];
    for (slot, &v) in field.iter_mut().zip(&values[1..]) {
        if !(v.fract() == 0.0 && (0.0..=1e9).contains(&v)) {
            return Err(Error::Archive(format!(
                "audio manifest value {v} is not a count"
            )));
        }
        *slot = v as u32; // integral and at most 1e9, so exact
    }
    let [version, text, enc_layers, enc_lanes, conv_kernel, stem, stem_stages, mels, rate, fft, win, hop] =
        field;
    if version != AUDIO_MANIFEST_VERSION {
        return Err(Error::Archive(format!(
            "audio manifest version {version} is not supported"
        )));
    }
    if stem_stages == 0 || stem == 0 || enc_lanes == 0 || mels == 0 {
        return Err(Error::Archive(
            "the audio manifest has no stem stages, stem channels, lanes or mel bins".to_string(),
        ));
    }
    Ok(AudioManifest {
        version,
        text,
        enc_layers,
        enc_lanes,
        conv_kernel,
        stem,
        stem_stages,
        mels,
        rate,
        fft,
        win,
        hop,
    })
}

fn decoder_cross(
    r: &mut Reader<'_, '_>,
    k: &Dims,
    p: &str,
    core: DecoderCore,
) -> Result<DecoderLayer, Error> {
    let cross_norm = r.norm(format!("{p}.cross_norm"), k.d)?;
    let q_proj = r.matrix(format!("{p}.cross.q_proj"), k.q_width, k.d)?;
    let k_proj = r.matrix(format!("{p}.cross.k_proj"), k.q_width, k.d)?;
    let v_proj = r.matrix(format!("{p}.cross.v_proj"), k.v_width, k.d)?;
    let gate_proj = r.matrix(format!("{p}.cross.gate_proj"), k.v_width, k.d)?;
    let out_proj = r.matrix(format!("{p}.cross.out_proj"), k.d, k.v_width)?;
    let cross_attn = CrossAttention {
        q_proj,
        k_proj,
        v_proj,
        gate_proj,
        out_proj,
        q_norm: r.norm(format!("{p}.cross_q_norm"), k.qk)?,
        k_norm: r.norm(format!("{p}.cross_k_norm"), k.qk)?,
    };
    Ok(DecoderLayer {
        norm_in: core.norm_in,
        self_attn: core.self_attn,
        post_attn_norm: core.post_attn_norm,
        attn_gate: core.attn_gate,
        cross_norm,
        cross_attn,
        post_cross_norm: r.norm(format!("{p}.post_cross_norm"), k.d)?,
        cross_gate: r.scalar(format!("{p}.cross_gate"))?,
        pre_hada_norm: core.pre_hada_norm,
        hadamard_mlp: core.hadamard_mlp,
    })
}

fn encoder_layer(
    r: &mut Reader<'_, '_>,
    k: &Dims,
    p: &str,
    kernel: usize,
) -> Result<EncoderLayer, Error> {
    let pre_hada_norm_0 = r.norm(format!("{p}.pre_hada_norm_0"), k.d)?;
    let attn_norm = r.norm(format!("{p}.ZCRMSNorm_0"), k.d)?;
    let post_attn_norm = r.norm(format!("{p}.post_attn_norm"), k.d)?;
    let conv_norm = r.norm(format!("{p}.conv_norm"), k.d)?;
    let conv_out_norm = r.norm(format!("{p}.conv_out_norm"), k.d)?;
    let pre_hada_norm = r.norm(format!("{p}.pre_hada_norm"), k.d)?;
    let q_proj = r.matrix(format!("{p}.q_proj"), k.q_width, k.d)?;
    let k_proj = r.matrix(format!("{p}.k_proj"), k.kv_qk_width, k.d)?;
    let v_proj = r.matrix(format!("{p}.v_proj"), k.kv_v_width, k.d)?;
    let gate_proj = r.matrix(format!("{p}.gate_proj"), k.v_width, k.d)?;
    let out_proj = r.matrix(format!("{p}.out_proj"), k.d, k.v_width)?;
    let q_norm = r.norm(format!("{p}.q_norm"), k.qk)?;
    let k_norm = r.norm(format!("{p}.k_norm"), k.qk)?;
    let attn_gate = r.scalar(format!("{p}.attn_gate"))?;
    let pw1 = r.matrix(format!("{p}.pw1"), k.d2, k.d)?;
    let pw2 = r.matrix(format!("{p}.pw2"), k.d, k.d)?;
    let dw = r.fp16_matrix(format!("{p}.dw"), kernel, k.d)?;
    let hadamard_mlp_0 = hadamard_mlp(r, k, &format!("{p}.hadamard_mlp_0"))?;
    let hadamard_mlp = hadamard_mlp(r, k, &format!("{p}.hadamard_mlp"))?;
    Ok(EncoderLayer {
        pre_hada_norm_0,
        hadamard_mlp_0,
        attn_norm,
        self_attn: SelfAttention {
            q_proj,
            k_proj,
            v_proj,
            taps: None,
            q_norm,
            k_norm,
            gate_proj,
            out_proj,
        },
        post_attn_norm,
        attn_gate,
        conv_norm,
        pw1,
        dw,
        conv_out_norm,
        pw2,
        pre_hada_norm,
        hadamard_mlp,
    })
}

fn stem(r: &mut Reader<'_, '_>, audio: &AudioManifest, d: usize) -> Result<Stem, Error> {
    let c = audio.stem as usize;
    let w = r.fp16_matrix("stem.w".into(), 9, c)?;
    let mut stages = Vec::new();
    let mut mels = audio.mels as usize;
    mels = mels.div_ceil(2);
    for i in 1..audio.stem_stages {
        stages.push(StemStage {
            dw: r.fp16_matrix(format!("stem.dw_{i}"), 9, c)?,
            pw: r.matrix(format!("stem.pw_{i}"), c, c)?,
        });
        mels = mels.div_ceil(2);
    }
    let out = r.matrix(
        "stem.out".into(),
        d,
        mul(c, mels, "stem channels · mel bins")?,
    )?;
    Ok(Stem {
        w,
        stages,
        out,
        mels: audio.mels as usize,
    })
}
