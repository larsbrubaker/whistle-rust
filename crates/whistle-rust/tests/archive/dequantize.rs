// Every tensor of whistle.cact dequantizes to upstream's `read_export` values (exact stage,
// CLAUDE.md "Tolerance policy"): float64 sum (relative to the larger of |sum| and the norm) and
// L2 norm within 1e-6 relative, the 64 sampled values within relative L2 1e-6, and the committed
// full arrays within relative L2 1e-6.
//
// Each single sample is also held within 1e-5 of the tensor's RMS, a guard against one wrong
// element rather than a precision bound: the reference's own f32 matmul `rot @ H` differs from
// the exact (f64) product by up to 5.8e-6 of the RMS (layer00.out_proj, measured with numpy),
// so no f32 evaluation order can promise 1e-6 per element.
//
// Measured when this landed: sum 7.5e-7, norm 1.3e-8, sampled relative L2 4.1e-7, sample/RMS
// 2.8e-6; full arrays 0 (fp16, fp32) and 1.4e-7 (CQ2), 2.2e-7 (CQ4) relative L2
// (`cargo test --test archive -- --nocapture` prints them).

use whistle::cact::{Archive, DType};

use crate::support::archive::{array, manifest, model_bytes};
use crate::support::{relative_l2, Npy};

const BOUND: f64 = 1e-6;
const SAMPLE_BOUND: f64 = 1e-5;

#[test]
fn every_tensor_dequantizes_to_upstreams_values() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    let m = manifest();
    let (mut worst_sum, mut worst_l2, mut worst_sample, mut worst_sampled) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (tensor, entry) in archive
        .tensors()
        .iter()
        .zip(m["tensors"].as_array().unwrap())
    {
        let name = entry["name"].as_str().unwrap();
        if tensor.dtype() == DType::Raw {
            assert!(
                tensor.to_f32(&archive).is_err(),
                "{name}: RAW has no values"
            );
            continue;
        }
        let values = tensor
            .to_f32(&archive)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            values.len() as u64,
            entry["count"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(values.len(), tensor.element_count(), "{name}");
        let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let l2 = values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        let (want_sum, want_l2) = (
            entry["sum"].as_f64().unwrap(),
            entry["l2"].as_f64().unwrap(),
        );
        // A sum near zero is compared against the norm, the scale of its rounding error.
        let sum_err = (sum - want_sum).abs() / want_sum.abs().max(want_l2).max(f64::MIN_POSITIVE);
        let l2_err = (l2 - want_l2).abs() / want_l2.max(f64::MIN_POSITIVE);
        assert!(
            sum_err <= BOUND,
            "{name}: sum {sum} vs {want_sum} ({sum_err:e})"
        );
        assert!(l2_err <= BOUND, "{name}: l2 {l2} vs {want_l2} ({l2_err:e})");
        let rms = (want_l2 / (values.len() as f64).sqrt()).max(f64::MIN_POSITIVE);
        let indices = entry["sample_indices"].as_array().unwrap();
        let samples = entry["samples"].as_array().unwrap();
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for (i, expected) in indices.iter().zip(samples) {
            let i = i.as_u64().unwrap() as usize;
            let expected = expected.as_f64().unwrap() as f32;
            let err = f64::from(values[i] - expected).abs() / rms;
            assert!(
                err <= SAMPLE_BOUND,
                "{name}[{i}]: {} vs {expected} ({err:e})",
                values[i]
            );
            worst_sample = worst_sample.max(err);
            got.push(values[i]);
            want.push(expected);
        }
        let sampled = relative_l2(&got, &want);
        assert!(sampled <= BOUND, "{name}: sampled relative L2 {sampled:e}");
        worst_sampled = worst_sampled.max(sampled);
        worst_sum = worst_sum.max(sum_err);
        worst_l2 = worst_l2.max(l2_err);
    }
    eprintln!(
        "max errors: sum {worst_sum:e}, l2 {worst_l2:e}, sampled relative L2 {worst_sampled:e}, \
         sample/rms {worst_sample:e}"
    );
}

#[test]
fn committed_full_arrays_match_within_relative_l2() {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    let m = manifest();
    let mut kinds = Vec::new();
    for (tensor, entry) in archive
        .tensors()
        .iter()
        .zip(m["tensors"].as_array().unwrap())
    {
        let Some(file) = entry["file"].as_str() else {
            continue;
        };
        let name = entry["name"].as_str().unwrap();
        let Npy::F32 { shape, data } = array(file) else {
            panic!("{file} is not float32");
        };
        assert_eq!(shape, tensor.shape(), "{name}");
        let err = relative_l2(&tensor.to_f32(&archive).unwrap(), &data);
        eprintln!("{name}: relative L2 {err:e}");
        assert!(err <= BOUND, "{name}: relative L2 {err:e}");
        kinds.push((tensor.dtype(), tensor.bits()));
    }
    for kind in [
        (DType::Cq, 2),
        (DType::Cq, 4),
        (DType::Fp16, 0),
        (DType::Fp32, 0),
    ] {
        assert!(kinds.contains(&kind), "no full array of {kind:?}");
    }
}
