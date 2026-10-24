//! The Whistle model: its weights bound by role from the archive (`weights`, built by `bind`),
//! the shared numeric primitives (`ops`), and the forward passes ported so far: the
//! convolutional stem with the encoder's input step (`stem`), ZC-RMSNorm and the Hadamard MLP
//! (`hadamard`). Each follows upstream's `whistle.py` and `architecture.py`.

mod bind;
mod hadamard;
pub(crate) mod ops;
mod stem;
mod weights;

pub use stem::{Lanes, StemOutput};
pub use weights::{
    AudioManifest, CrossAttention, DecoderLayer, EncoderLayer, Engram, HadaPerms, HadamardMlp,
    Matrix, Mhc, Norm, QkvTaps, SelfAttention, Stem, StemStage, Weights,
};
