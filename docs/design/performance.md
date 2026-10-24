# Performance

How fast a transcription runs, what the dense kernels do, and what is left. Measured with the
timing tests in `crates/whistle-rust/tests/transcribe/timing.rs`:

```bash
cargo test --release -p whistle-rust --test transcribe timing -- --include-ignored --nocapture --test-threads=1
```

## Machine

Apple M5 (10 cores: 4 performance, 6 efficiency), 24 GB, macOS, release build, aarch64 (NEON
path). The 2026-10-24 "after" numbers were taken while another build and test run shared the
machine (load average 9–10), so they are noisy, and the thread scaling especially is
understated; rerun on an idle machine before quoting a ratio.

## Summary

Two sets of runs, not to be mixed. When the SIMD kernels and threads landed (load average 9–10),
with the default 4 threads the encoder ran 12–16× faster than before (567 ms to 36–44 ms on
3 s; about 6.8 s to 0.35–0.43 s on 30 s), the decoder about 2× per token (10.8 ms to about
4.5–5 ms), and the whole 3 s call about 4.3–5.8× (661 ms to 114–155 ms). The few-row kernel
path (below) was measured later as its own before/after pair on a less loaded machine (load
average 7–13): the decoder step went from 4.03–4.09 ms to 2.62–2.82 ms per token and the whole
3 s call from 111.5 ms to 81–87 ms. The before of that pair (111.5 ms) is faster than the
earlier 114–155 ms because the machine was quieter, not because of a change in between.

## Numbers

en_samantha (3.09 s, 18 tokens) and 30 s (en_samantha repeated; encoder only, cancellation
gaps):

| Stage | Before (1 thread) | After, 1 thread | After, 4 threads |
|---|---|---|---|
| 3 s: encoder (features to encoder output) | 567 ms | 46–62 ms | 36–44 ms |
| 3 s: decoder setup (cross-attention memory) | 95 ms | 6–9 ms | 4 ms |
| 3 s: decoder, per token | 10.8 ms | 4.6–5.3 ms | 4.5 ms (5.3–6.6 ms before rows < 16 stayed on one thread) |
| 3 s: whole default call | 661 ms | (includes binding; see note) | 114–155 ms |
| 30 s: features | 73 ms | 48–68 ms | 45–52 ms |
| 30 s: VAD | 18 ms | 13–20 ms | 12–16 ms |
| 30 s: stem | 719 ms | 46–53 ms | 28–36 ms |
| 30 s: each encoder layer | 494–847 ms | 47–64 ms | 38–48 ms |
| 30 s: features to encoder output | about 6.8 s | 510–568 ms | 400–425 ms |

The "whole call" of the one-thread run is the first call in the process, so it also pays for
binding the archive (`Whistle::new`); the four-thread run after it is the real per-call cost.

## What was done

- **`kernels::linear_into`, the dense `x · wᵀ`.** It keeps the port's accumulation order
  exactly: each output is the sequential sum from `-0.0` in input order, product rounded then
  added, no FMA. It vectorizes across outputs instead: a panel of 16 weight rows is transposed
  once into `[n, 16]` scratch, and each block of 4 input rows broadcasts `x[i]` against panel
  row `i`, one SIMD lane per output (16 independent accumulator vectors on NEON). Paths: NEON
  (aarch64), AVX2 (x86_64, `is_x86_feature_detected!`, mul then add, not FMA), wasm `simd128`
  (`cfg(target_feature = "simd128")`), and the portable scalar loop; the unit tests pin scalar
  and SIMD to the old sequential sum bit for bit, on shapes that are not multiples of the panel
  or row block. Because the bits did not change, no stage bound, fixture comparison or token
  stream moved.
- **Callers routed through it:** `Matrix::project` (every Dense layer), the mHC mixing logits
  (`mhc::mix_logits`, a band of `phi`'s rows), and attention's logits (`attention::attend`
  computes every query of a head against every key as one projection, then scales and masks;
  the full pass and the cached step share it, so the KV cache stays bit-identical).
- **The stem's 3×3 convolutions** run all channels of a cell together per tap (same tap order
  per channel, so the same bits), which vectorizes; the closure per element is gone.
- **Threads** (`whistle::set_threads(n)`, default `available_parallelism().min(4)`, capped at
  `MAX_THREADS` = 16, process-wide; wasm32 always 1; tests set a count per calling thread with
  `model::with_threads`): a projection of at least 2²⁰ multiply-adds splits across scoped
  threads by input rows, at least 16 rows per worker. 2²⁰ is where 4 threads first beat 1 (86
  vs 58 µs; 2¹⁹ is 52 vs 54 µs; the measurement is beside `PARALLEL_MIN_MACS`). A split never
  divides a sum, so 1 and N threads are bit-identical (`tests/transcribe/threads.rs`).
- **The decode step is single-threaded.** Its projections have one row (a cached step) and
  splitting them by output columns paid a thread spawn per projection per token: 6.01 ms per
  token on 4 threads vs 5.03 ms on 1. Projections with fewer than 16 rows now stay on the
  caller's thread, so the thread count does not change the decoder's cost (4.5–4.6 ms per token
  on both).

## The few-row path (the decoder step)

A cached decoder step projects one row per matrix. For fewer rows than the panel path's block
of 4, `kernels::linear_into` now runs `kernels/small.rs`: it loads 4 inputs of each of 4 weight
rows, transposes that 4×4 block in registers (NEON `trn1`/`trn2`; SSE unpacks and moves inside
the AVX2 path; simd128 `i32x4_shuffle`) and adds each input's products to one lane per output,
in input order from `-0.0`, mul then add. Inputs past a multiple of 4 and outputs past a multiple
of 4 finish by the scalar formula. So the bits are the panel path's (and the sequential sum's:
`the_few_row_path_reproduces_the_sequential_sum_on_every_tail_shape`), and no stage bound,
fixture comparison, KV-cache identity or token stream moved. No weight layout changed and no
storage was added: resident memory is unchanged (the per-call `[n, 16]` scratch is gone for
these rows). Peak resident set of the `timing_of_one_clip` test process (which binds the model
twice, as a `Whistle` and as bare `Weights`; `/usr/bin/time -l`): 339.2 MB before, 340.2 MB
after.

**Why one group of 4 outputs.** A step streams about 72 MB of f32 weights per token (per layer
1632 self-attention, 1408 cross-attention rows of 512, the mHC `phi` bands of 2048; the Engram
projections; the 8199×512 head), more than the caches hold, and how many weight rows the kernel
reads at once decided the speed out of cache (`few_row_timing`, best of 10, release):

| | panels (scratch transpose) | in-register, 1 group (4 rows) | 2 groups | 4 groups |
|---|---|---|---|---|
| 64 distinct 512×512, 1 row (64 MB) | 3245 µs | 2168 µs | 2674 µs | 3567 µs |
| one 512×512, 1 row (in cache) | 49 µs | 32 µs | 20 µs | 16 µs |
| 8199×512 head, 1 row | 791 µs | 540 µs | 360 µs | 341 µs |

Out of cache one group wins, so that is the setting (`small::GROUPS`); in the real step it gave
2.62–2.82 ms per token against 3.66–3.88 ms with 4 groups.

| en_samantha (3.09 s, 18 tokens), 2026-10-24, load average 7–13 | Before | After |
|---|---|---|
| decoder, per token, 1 thread | 4.03 ms | 2.62–2.70 ms |
| decoder, per token, 4 threads | 4.09 ms | 2.81–2.82 ms |
| whole default call, 4 threads | 111.5 ms | 81.4–86.5 ms |
| encoder, 4 threads (unchanged path) | 31.1 ms | 30.7–31.5 ms |

A sampled profile (`sample`, 150 decodes of the clip) before the group setting put about 90% of
the step in the few-row kernel, spread over cross-attention (about 30%), self-attention's
projections and gate/out (about 37%), the head (about 23%) and the Engram projections (5%);
norms, A8, Hadamard, Sinkhorn and attention itself are each under 2%.

## Packed CQ weights

The archive has 152 CQ records (145 at 2 bits, 7 at 4 bits, groups of 128). The two Engram
tables were already kept packed (`CqRows`, gathered by row). The other 150, every CQ matrix the
model projects, are now packed too instead of dequantized: about 36 M weights, about 130 MB
less per binding (measured below). Each keeps its record: the packed indices
(10.2 MB for the 150) and the fp16 norms in the blob (0.56 MB), plus an f32 copy of the
norms (1.12 MB, the kernels' per-group scale). Packed matrices need a power-of-two group of at
least 16 (the band decode's butterflies rely on it; `cact` admits no other) and 2 or 4 bits;
any other record would stay dense. The fp16 matrices (stem taps, depthwise kernels, Hadamard
factors, the mel filterbank) stay dense. `Matrix::data()` returns a `Cow`: borrowed for a dense
matrix; for a packed one, a new dequantized vector (exactly `Tensor::to_f32`) on every call,
with nothing kept, so it costs `rows · cols` f32 and about 0.4 ns per weight each time. Only
tests and inspection call it; the model never does. Two packed matrices compare equal by their
records, without dequantizing. The embedding is gathered row by row (`Matrix::row_into`,
padded rows included) and the mHC mixing projects a band of `phi`'s rows
(`Matrix::project_rows`). `kernels/packed.rs` has two orders:

- **Rotated (the decoder's matrices, the head and the Engram projections).** A group's row
  dequantizes to `(cb[idx] · norm_g) @ H`, and `H` is symmetric, so `w_g · x_g = norm_g ·
  (cb[idx] · H x_g)`. Each input row is padded and rotated once per group by the existing
  FWHT, and each output is `Σ_g norm_g · Σ_k cb[idx] · xr_k`. There are 16 lanes per group
  (lane `j` takes `k ≡ j mod 16`), mul then add, a fixed reduction tree, and the groups are
  summed in order. NEON decodes 16 indices per step: a `vqtbl1q_u8` byte spread, a shift and
  a mask give `4 · code`, then `vqtbl1q_u8` (2-bit) or `vqtbl4q_u8` (4-bit) into the codebook's
  bytes. AVX2 and simd128 have their own paths (below). The unit tests pin NEON to scalar
  bit for bit on every archive matrix at 1, 4 and 5 rows and at both widths. The full pass and
  the cached step use the same kernel, so the KV-cache identity still holds. So does the thread
  identity: a split never divides a sum.
- **Dense (the encoder's and the stem's matrices, and the decoder's cross-attention keys and
  values).** These project many frames (the keys and values only ever project the encoder's
  frames, once per call, shared by every decode pass) and have no cached step. The rotated order
  decodes the weights again for every 4 input rows, which made the 30 s encoder 3–4× slower
  (2.07 s against 0.55 s on 1 thread under the same load). So these matrices run
  `kernels/band.rs`: workers own contiguous slices of the outputs, and each decodes 64 rows at a
  time and multiplies them by every input row while they are in cache, storing straight into
  the output (`linear_band_into`). Results do not depend on the thread count, and no full
  dequantized copy is made. The decode gives `Tensor::to_f32`'s bits exactly (a test pins it on
  every archive matrix and path). NEON folds `(cb[c] · norm) · scale` into a per-group lookup
  table, decodes 16 indices with the same byte shuffles, does the butterfly levels 1–8 in
  registers and the higher ones as vector add and subtract: 0.37–0.58 ns per weight against
  1.9–2.0 scalar (`band_decode_timing`, load 10–12). The dense kernel then gives the same bits
  as before, so no encoder or stem bound moved.

**SIMD by target.** Every path keeps the scalar order's lanes, reduction and butterfly
order, mul then add with no FMA, and unaligned loads. The lane mappings are worked out in each
module's header.

| Kernel | aarch64 | x86_64 | wasm32 |
|---|---|---|---|
| rotated projection (decoder) | NEON: `vqtbl` index and codebook lookups | AVX2 (`Avx2Detected`): `_mm256_srlv_epi32` index decode, `_mm256_permutevar8x32_ps` codebook lookup (`packed_x86.rs`) | simd128: nibble split with `i8x16_swizzle` and `v128_bitselect`, codebook as byte planes through `i8x16_swizzle` (`packed_wasm.rs`) |
| band decode (encoder, stem, cross K/V) | NEON | AVX2: the same decode, butterflies by `_mm256_shuffle_ps` / `_mm256_blend_ps` / `_mm256_permute2f128_ps` | simd128: the same decode, butterflies by `i32x4_shuffle` |
| anything else | scalar order | scalar order (no AVX2) | scalar order (no simd128) |

What has run and what has not:
- **NEON** runs here: the unit tests pin it to scalar bit for bit.
- **simd128** ran in wasmtime through a scratch harness outside the repo. A wasm32 build with
  simd128 and one without hashed the same encoder output and decoder logits bits on a 2 s
  chirp (`6644388670678719336` both); the simd128 build took 0.62 s against 1.22 s.
- **AVX2 has not run.** There is no x86_64 machine or Rosetta here. It is checked by
  construction and by `--emit asm` for x86_64-unknown-linux-gnu:
  - `rows_into_avx2` and `band_into_avx2` hold their whole loops, with no calls except
    memcpy and panic paths, so there is no per-group call.
  - There is no FMA anywhere in the crate.
  - The expected instructions are present: `vpsrlvd`, `vpermd` (the integer form of the f32
    permute, the same bits), `vblendvps` and blends 0xAA/0xCC/0xF0. LLVM chose
    `vmovsldup`/`vmovshdup`/`vmovddup` and `vinsertf128` for the same element moves, and
    `vhaddps` for the exact pairwise final adds.
- The wasm asm (`--emit asm`, +simd128) shows both kernels inlined into `rows_into` and
  `band_into`, with `i8x16.swizzle`, `i8x16.shuffle`, `f32x4.mul` and `f32x4.add`, no calls
  and no FMA.
- The same unit tests compare `best_path` against scalar, so they cover AVX2 when run on
  x86_64 hardware.

**Numerics.** The rotated order rounds differently from dequantize-then-sum. Measured
2026-10-24 against `main` (e78c65a), same fixtures:

| Stage (worst) | main | packed |
|---|---|---|
| decoder layer from previous lanes (`LAYER_BOUND`, was 5e-4, now 2e-3) | 1.39e-4 (de_anna L2) | 1.08e-3 (en_samantha L3) |
| decoder/out (bound 3e-2) | 1.65e-2 (pl_zosia); silence 6.0e-7 | 1.76e-2 (pl_zosia); silence 1.30e-2 |
| decoder/logits (bound 1.5e-2) | 7.17e-3 | 5.40e-3 |
| decoder block layers 0, 3 (bound 7e-3) | 3.49e-3 | 3.49e-3 |
| decoder self/cross projections, norms, outputs (≤1e-6) | 4.7e-7 / 6.3e-7 | 6.6e-7 / 7.4e-7 |
| Engram keys / values (5e-4) | 2.04e-4 / 2.31e-4 | unchanged |
| encoder layer, encoder end to end, stem | 2.28e-3, 2.74e-2, 1.32e-4 | unchanged (dense order) |

The per-layer decoder measurement moved past its bound, from one flipped A8 step on
en_samantha layer 3. 1.08e-3 is under the reference's own 2.6–2.8e-3 for one block between
execution modes (`oracle.md`), so the bound was re-set to 2e-3 with the measurement beside it.
That move was pre-approved for a contract that holds. The exact-stage projection tests stay
≤ 1e-6 on the packed path. **Token contract:** identical tokens, text and language on all 8
speech clips with the VAD off and with the default gate and trim. Silence and empty audio give
an empty transcript. The cached steps score exactly as the full recompute, and 1 and 4 threads
are bit-identical.

**Why silence's `decoder/out` moved** (6.0e-7 to 1.30e-2). It is one flipped A8 step, not a
zero-row asymmetry. Run layer by layer, the rotated order against the dense bits on the silence
clip's own lanes (7 positions, "Thank you."), the difference is f32 rounding at every position
in layer 0 (3–6e-7). Then a single jump appears at layer 1, position 0 (the start token, 4.8e-7
to 1.2e-4), and self-attention spreads it to every later position from layer 2 on (1.4e-3 to
1.5e-2 by layer 6). Silence's zero features never reach the decoder as zero rows: its memory
matches the reference to 4.5e-8, and no layer-0 row matches exactly. On `main` this clip
happened to have no flip; the speech clips already carried flips at the same 1e-2 level.

**Memory and time.** Apple M5, release, 2026-10-24, the machine shared with other builds. Main
(e78c65a) and packed were run back to back, alternating. Peak resident set of the `transcribe`
timing process (`/usr/bin/time -l`; it binds the model twice, as a `Whistle` and as bare
`Weights`): 399.5–402.9 MB on main, 163.3–179.4 MB packed. The resident set right after
binding (`from_cact`, measured in review) is 171 MB on main against 40.8 MB packed: about 130 MB
less per binding. The quietest set (load average 2.9, three rounds each; before the strided store and the
folded lookup table, which only made the band path cheaper):

| | main | packed |
|---|---|---|
| decoder, per token, 1 thread | 2.46–2.50 ms | 2.09–2.31 ms |
| decoder, per token, 4 threads | 2.54–2.56 ms | 2.15–2.16 ms |
| 3 s encoder, 1 thread | 36.9–37.4 ms | 41.6–41.8 ms |
| 3 s encoder, 4 threads | 29.8–30.1 ms | 32.0–32.2 ms |
| 3 s decoder setup (cross keys and values then rotated), 1 / 4 threads | 5.1 / 3.5 ms | 6.2–6.5 / 3.9 ms |
| 30 s features to encoder output, 1 thread | 385–392 ms | 397–406 ms |
| 30 s features to encoder output, 4 threads | 257–263 ms | 274–300 ms |
| 3 s whole default call, 4 threads | 78.1–78.8 ms | 73.4–74.5 ms |

The final code, 30 s only, four alternating rounds at load average 9.6–10: 1 thread 462–640
ms on main (median about 540) against 483–573 ms packed (about 523); 4 threads 394–518 ms (about
439) against 410–502 ms (about 471): within the noise. The 3 s clip in the same conditions:
whole call 126–174 ms on main against 114–150 ms packed (4 threads); encoder 56–60 ms against
61–83 ms on 4 threads and 58–69 against 61–69 ms on 1.

So a binding holds about 130 MB less and the decoder step is faster: 2 bits per weight to
stream instead of 32. The whole call is no slower. The encoder pays one decode of its matrices
per call (about 19M weights at 0.4–0.6 ns): within the noise on 30 s, but a few ms (5–12%) on a
3 s clip, which has only about 39 frames to amortize it over. That is the one stage still
measurably behind `main` on a quiet machine.

## What is left

- **The decoder step** (about 2.7 ms per token) runs on one thread and is now bound by how the
  weights stream from memory: at one group the kernel reads 4 rows at once, about 30 GB/s,
  latency-bound on one accumulator chain per 4 outputs; more groups compute faster but read more
  rows at once and lose out of cache. Weights stored once in the transposed panel layout
  (`[n, 16]` per panel, built at bind time) would read as one sequential stream with 4
  accumulators, likely the next large cut; it adds the decoder's projected matrices again
  (about 72 MB on top of the dequantized weights) unless `Matrix::data()` callers move to it,
  so it is a memory trade-off to decide. A persistent pool would let single-row projections
  split. A 30 s clip end to end is the encoder (about 0.42 s) plus the decode, so the per-token
  cost decides whether it lands well under 2 s.
- **Spawn cost:** threads are spawned per projection (`std::thread::scope`); a persistent pool
  would cut the overhead that keeps small and single-row projections on one thread.
- **Packed encoder on short clips:** the per-call band decode costs a few ms on a 3 s clip
  (above).
- **Run the AVX2 packed paths** (and the dense AVX2 kernels) on an x86_64 machine: the unit
  tests pin them to scalar, but they have only been checked by construction and asm.
