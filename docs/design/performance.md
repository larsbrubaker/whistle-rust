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
- **Packed CQ weights** and the fast Walsh–Hadamard rotation (PORTING_PLAN item 6) are the
  next step; these kernels work on the dense f32 weights.
