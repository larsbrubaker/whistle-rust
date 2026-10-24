//! The Whistle model: its weights bound by role from the archive (`weights`, built by `bind`),
//! the shared numeric primitives (`ops`), and the forward passes ported so far: the
//! convolutional stem with the encoder's input step (`stem`), ZC-RMSNorm and the Hadamard MLP
//! (`hadamard`), gated self-attention (`attention`), the mHC lane stack step (`mhc`), and the
//! encoder (`encoder`). Each follows upstream's `whistle.py` and `architecture.py`.

mod attention;
mod bind;
mod encoder;
mod hadamard;
mod mhc;
pub(crate) mod ops;
mod stem;
mod weights;

pub use attention::Projections;
pub use encoder::{Encoder, EncoderContext, EncoderOutput};
pub use stem::{Lanes, StemOutput};
pub use weights::{
    AudioManifest, CrossAttention, DecoderLayer, EncoderLayer, Engram, HadaPerms, HadamardMlp,
    Matrix, Mhc, Norm, QkvTaps, SelfAttention, Stem, StemStage, Weights,
};
