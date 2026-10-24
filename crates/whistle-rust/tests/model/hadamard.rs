// ZC-RMSNorm and the Hadamard MLP against en_samantha's committed block internals: encoder
// layers 0 and 7 (both MLPs of the block) and decoder layers 0 and 3. Neither contains an A8
// step, so both are exact stages (CLAUDE.md "Tolerance policy", relative L2 <= 1e-6); each
// test starts from the fixture's own input to the stage.

use whistle::model::HadamardMlp;

use crate::shared::weights;
use crate::support::{relative_l2, Fixture};

const EXACT: f64 = 1e-6;

#[test]
fn zc_rms_norm_reproduces_the_encoder_pre_hada_norm() {
    let f = Fixture::load("en_samantha");
    for layer in [0usize, 7] {
        let p = format!("encoder/layer{layer:02}");
        let (_, x) = f.f32(&format!("{p}/block_in"));
        let (_, want) = f.f32(&format!("{p}/block/pre_hada_norm_0/0"));
        let got = weights().encoder[layer].pre_hada_norm_0.forward(&x);
        let err = relative_l2(&got, &want);
        println!("{p} pre_hada_norm_0: relative L2 {err:.3e}");
        assert!(err <= EXACT, "{p}: relative L2 {err:.3e}");
    }
}

fn check_mlp(f: &Fixture, prefix: &str, norm: &str, mlp_name: &str, mlp: &HadamardMlp) {
    let (_, x) = f.f32(&format!("{prefix}/block/{norm}/0"));
    let (_, want) = f.f32(&format!("{prefix}/block/{mlp_name}/0"));
    let got = mlp.forward(&x, &weights().hada_perms);
    let err = relative_l2(&got, &want);
    println!("{prefix} {mlp_name}: relative L2 {err:.3e}");
    assert!(err <= EXACT, "{prefix} {mlp_name}: relative L2 {err:.3e}");
}

#[test]
fn hadamard_mlp_reproduces_the_encoder_layers() {
    let f = Fixture::load("en_samantha");
    for layer in [0usize, 7] {
        let p = format!("encoder/layer{layer:02}");
        let l = &weights().encoder[layer];
        check_mlp(
            &f,
            &p,
            "pre_hada_norm_0",
            "hadamard_mlp_0",
            &l.hadamard_mlp_0,
        );
        check_mlp(&f, &p, "pre_hada_norm", "hadamard_mlp", &l.hadamard_mlp);
    }
}

#[test]
fn hadamard_mlp_reproduces_the_decoder_layers() {
    let f = Fixture::load("en_samantha");
    for layer in [0usize, 3] {
        let p = format!("decoder/layer{layer:02}");
        let l = &weights().decoder[layer];
        check_mlp(&f, &p, "pre_hada_norm", "hadamard_mlp", &l.hadamard_mlp);
    }
}
