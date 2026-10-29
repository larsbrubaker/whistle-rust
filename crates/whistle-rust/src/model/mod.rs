//! The Whistle model: its weights bound by role from the archive (`weights`, built by `bind`),
//! the shared numeric primitives (`ops`) over the SIMD dense kernels and thread setting
//! (`kernels`), and the forward passes ported so far: the
//! convolutional stem with the encoder's input step (`stem`), ZC-RMSNorm and the Hadamard MLP
//! (`hadamard`), gated self-attention (`attention`), the mHC lane stack step (`mhc`), and the
//! encoder (`encoder`). Each follows upstream's `whistle.py` and `architecture.py`.
//! The decoder's inputs (`decoder_input`: embedding, memory) and its Engram sites (`engram`)
//! feed the decoder (`decoder`: causal self-attention, `cross_attention`, the stack and the
//! output head); `decoder_step` runs it one token at a time over a KV cache (`DecoderState`).
//!
//! Whistle's geometry, read from the archive header (`cact::Geometry`) and checked by `bind`:
//! a text vocabulary of 8192 ids then the 7 language tags (`en de fr es it nl pl`, 8192–8198);
//! width 512; 8 query heads over 2 KV heads (8 over 8 in cross-attention), q and k 48 wide,
//! v 64; 8 encoder and 8 decoder layers in 4 mHC lanes; encoder conv kernel 9; 3 decoder QKV
//! taps; Engram at decoder layers 3 and 7 (n-gram orders 2 and 3, two hash heads each, 18,432
//! slots of 128-wide rows, 4 value taps at dilation 3); a 128-channel, 3-stage stem; RoPE θ 1e5;
//! 80 mels at 16 kHz; at most 320 decoder tokens; pad 0, EOS 1, BOS 2.
//!
//! Every pass runs in the reference's deployment numerics: weights dequantized from the
//! archive, activations fake-quantized to A8 (`ops::a8`) and the attention KV path at the
//! archive's `kv_bits`.

mod attention;
mod bind;
mod cross_attention;
mod decoder;
mod decoder_input;
mod decoder_step;
mod encoder;
mod engram;
mod hadamard;
mod kernels;
mod mhc;
pub(crate) mod ops;
mod stem;
mod weights;

pub use attention::Projections;
pub use bind::MAX_SEQ_LEN;
pub use cross_attention::CrossMemory;
pub use decoder::{CrossAttentionWeights, Decoder, DecoderContext, HeadFilter, Logits};
pub use decoder_input::DecoderMemory;
pub use decoder_step::DecoderState;
pub use encoder::{Encoder, EncoderContext, EncoderOutput};
pub use engram::{engram_add, engram_indices, ngram_in_sequence, tap_in_sequence, MAX_NGRAM_ORDER};
pub use kernels::{set_threads, threads, with_threads, MAX_THREADS};
pub use stem::{Lanes, StemOutput};
pub use weights::{
    AudioManifest, CrossAttention, DecoderLayer, EncoderLayer, Engram, EngramHash, HadaPerms,
    HadamardMlp, Matrix, Mhc, Norm, QkvTaps, SelfAttention, Stem, StemStage, Weights,
};
