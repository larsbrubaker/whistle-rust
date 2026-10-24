# The Whistle model, the archive, and where each piece comes from

## Sources

| Piece | Where | License | Role here |
|---|---|---|---|
| Model definition (JAX/Flax): `needle/model/whistle.py`, `architecture.py`, `quantize.py` | github.com/cactus-compute/needle, pinned in `REFERENCE` | Apache-2.0 | The spec the port follows; the oracle runs it. |
| `.cact` format: `needle/model/export.py` | same | Apache-2.0 | Byte layout, `read_export` (a dequantizing reader) and `RefTokenizer`. |
| Greedy decode: `whistle_finetune._score_quantised` | same | Apache-2.0 | The decode prompt: BOS, language tag, text, EOS. |
| `whistle.cact` (16,919,407 B) and `checkpoints/whistle.safetensors` (fp32, 220 MB), weights 2.0.0 | huggingface.co/Cactus-Compute/whistle | Apache-2.0 | What the library loads; the oracle's cross-check. |
| Needle 3 engine binaries (`needle` CLI, `libneedle.a`, `needle.wasm`) | huggingface.co/Cactus-Compute/needle3 | model card: Apache-2.0 | Black-box transcripts only. |
| `cactus` engine and kernel source | github.com/cactus-compute/cactus | non-commercial custom license | Never used (CLAUDE.md contract 2). It holds no Whistle code. |

Behaviour that exists only inside the closed engine, and is therefore designed here from its
documented behaviour: beam search (the vendor benchmarks at 5 beams), keyword biasing, word
timestamps (aligned from decoder cross-attention), windowing past 30 s, streaming (it commits
words two passes agree on), the silence gate (silence and steady noise return an empty
transcript), and the KV cache.

## Configuration (`WhistleConfig`, from the checkpoint)

text vocabulary 8192 + 7 language tags (`en de fr es it nl pl`, ids 8192–8198); `dim` 512;
8 heads (`dim / v_dim`), 2 KV heads, `qk_dim` 48, `v_dim` 64; 8 encoder and 8 decoder layers,
4 mHC lanes each; encoder conv kernel 9; decoder QKV conv taps 3; Engram at decoder layers 3 and
7, n-gram orders 2 and 3, 2 hash heads per order (4 tables), 18,432 slots, 128-wide sub-vectors,
4 value taps at dilation 3; stem 128 channels, 3 stages; RoPE θ 1e5; 80 mels, 16 kHz, FFT 512,
window 400, hop 160; at most 320 decoder tokens; pad 0, EOS 1, BOS 2.

## Front end (`whistle.features`, `mel_matrix`)

Samples beyond the clip length are zeroed. STFT with the symmetric `np.hanning(400)` window,
`nperseg` 400, overlap 240, `nfft` 512, zero boundary padding of 200 each side, no end padding,
so a clip of N samples gives `1 + N // 160` frames, of which `N // 160` are valid. Power is
`|z · Σwindow|²` (undoing scipy's window normalization), times an 80-filter mel bank on a
piecewise mel scale (linear below 1 kHz, log above), then `log(x + 2⁻²⁴)`. Each mel bin is
normalized over the valid frames: mean, and the sample standard deviation (n − 1), `(x − mean) /
(std + 1e-5)`, then masked. The mel matrix is also stored in the archive (`mel`, fp32).

## Stem

Three stages of 3×3 depthwise 2-D convolution, stride 2, padding 1, treating (time, mel) as an
image with one input channel per output channel: stage 1 maps the single mel plane to 128
channels; stages 2 and 3 apply SiLU, a depthwise conv and a 128×128 pointwise dense. Valid
lengths halve with rounding up at each stage and frames past them are zeroed. Then SiLU, the
128 × 10 features flattened per frame channel-major (index `c · 10 + f`), A8, and a dense
layer from 1280 to 512: one frame per 80 ms.

## Encoder (`EncoderBlock` in a `LaneStack`)

The frame count is padded to even. Each block: `x + 0.5 · HadamardMLP(norm(x))`; gated
self-attention `x + σ(attn_gate) · norm(attn(norm(x)))` (bidirectional over valid frames, RoPE);
a conv module `x + pw2(silu(norm(depthwise_k9(glu(pw1(norm(x))) · valid))))`; then another
half-step Hadamard MLP. All norms are ZC-RMSNorm: `(1 + scale) · x / sqrt(mean(x²) + 1e-6)`.

## Attention (`MultiHeadAttention`, `CrossAttention`)

Input A8; Q, K, V projections; in the decoder a causal 3-tap depthwise conv on each of Q, K, V;
ZC-RMSNorm on q and k per head; RoPE on q and k (self-attention only); A8 on q, and the KV path
at `kv_bits` (8: per-head symmetric A8). The fused kernel pads qk (48) and v (64) to 64 and
scales q by `sqrt(64/48)` so the softmax scale is `1/sqrt(48)`. Output times
`σ(gate_proj(x))`, A8, then `out_proj`. Cross-attention keys and values come from the decoder
memory (encoder output plus `σ(pe_gate)` × sinusoidal positions, A8), masked to valid frames.

## mHC lane stacks (`LaneStack`, `_LaneBody`)

The residual stream is 4 lanes. Per layer and token: `nx = A8(rms_unit(lanes flattened))`;
pre-mix weights `σ(a_pre · nx·φ_pre + b_pre + pre_off)` pick the block input from the lanes; the
block's update `y = block(u) − u` is added back with post weights `2σ(a_post · nx·φ_post + b_post
+ post_off)`; lanes mix through a doubly stochastic matrix, 20 Sinkhorn iterations of
`a_res · nx·φ_res + b_res`. Layer i's offsets favour lane `i mod 4`. The stack ends with the lane
mean and ZC-RMSNorm.

## Hadamard MLP

On a width padded to the next power of two (512): `d4 · K3(d3 · P2(K2(silu(d2 · cond · P1(K1(d1 ·
x)) + b2))))`, where each `K` is a Kronecker product of two learned square factors (`w1a ⊗ w1b`,
…), `P1`, `P2` are fixed permutations (stored in the archive as `hada_p1`, `hada_p2`), and
`cond = 1 + softmax(x · cond_v) · cond_u` is a rank-8 per-token gain.

## Decoder (`DecoderBlock` in a `LaneStack`)

Tokens are embedded and scaled by `sqrt(512)`. At the Engram sites, each position hashes the
last 2 and 3 tokens (two seeds per order: `acc = seed; acc = (acc ^ token) · 0x01000193` over the
n-gram, wrapping u32, then `acc ^ (acc >> 15)`, modulo 18,432), gathers four 128-wide table rows
(zeroed where the n-gram reaches before the sequence), projects them (A8) to keys and values,
applies a 4-tap value conv at dilation 3, and adds `σ(⟨rms(x), rms(k)⟩ / sqrt(512)) · v` before the
block. Each block: gated causal self-attention, gated cross-attention, Hadamard MLP. Logits are
the A8 output times the tied embedding.

## Archive (`.cact`)

All little-endian. A 49-field header (48 u32, `rope_theta` f32) carrying the geometry and
`kv_bits`, the CQ codebooks (`cb2[4] | cb3[8] | cb4[16]` f32), a nameless tensor directory
(`u8 dtype, u8 ndim, u16 pad, u32 shape[4], u64 offset, u64 nbytes, u32 group, u32 bits`), then
64-byte-aligned blobs in upstream's canonical order, ending with the raw SentencePiece tokenizer
(BPE pieces with scores and types, plus the 7 language tags as user-defined pieces).

CQ ("Cactus Quants") stores a logical `[out, in]` matrix as packed indices (LSB-first bitstream
per row, `bits` per index, `in` padded to the group size 128) followed by per-group fp16 L2 norms.
A group dequantizes to `(codebook[idx] · norm) · H`, with `H` the orthonormal 128-point
Walsh–Hadamard matrix. `whistle.cact` uses 2-bit for most matrices and 4-bit for the embedding
and the mHC projections.
