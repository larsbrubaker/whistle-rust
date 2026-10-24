# Summarizes a V8 .cpuprofile from `node --cpu-prof` on the wasm bench build
# (scripts/wasm-bench.sh profile): self time per Rust function (demangled with c++filt when it
# is on PATH), top N, and the time grouped into stage buckets (front end and VAD, encoder,
# decoder setup, decoder step, everything else), each split by what the innermost recognisable
# frame is doing. Std only; run with `python3 -I`.
#
#   wasm_profile.py PROFILE.cpuprofile [--under REGEX] [--top N]
#
# --under keeps only samples with a frame matching REGEX on the stack (default: the timed
# `Whistle::transcribe_wav` calls, so the bench example's separate encoder-only timing and the
# model load are left out).
import collections
import json
import re
import shutil
import subprocess
import sys

# Stage of a sample: the first rule whose pattern matches any frame on its stack, checked in
# this order (the decoder step's head runs under `Decoder::logits`, its setup under
# `cross_memories`/`DecoderState::new`).
STAGES = [
    ("front end + VAD", r"audio::mel::features|audio::fft|whistle::vad::|audio::wav"),
    ("decoder setup", r"cross_memories|memory_kv|decoder_memory|DecoderState>::new"),
    ("decoder step", r"DecoderState>::step|Decoder>::logits|Decoder>::finish|transcribe::search"),
    ("encoder", r"Encoder>::|Stem>::forward"),
]

# What a sample is doing: the innermost frame matching one of these, leaf first.
KINDS = [
    ("exp/ln/libm", r"^(expf?|logf?|log1pf?|tanhf?|powf?|sinf?|cosf?|__math_\w+|__rem_pio2\w*|"
                    r"__sindf|__cosdf|__expo2f?|erff?|sqrtf?)$"),
    ("alloc/memcpy", r"dlmalloc|dlfree|dlrealloc|calloc|^malloc$|^free$|memcpy|memset|memmove|"
                     r"__rust_alloc|__rust_dealloc|__rust_realloc|__rdl_|RawVec|"
                     r"finish_grow|::drop$|Vec<f32>>::clone"),
    ("band decode (packed CQ)", r"band_into|CqLayout>::decode|slice_into"),
    ("rotated packed kernel", r"PackedMatrix>::rows_into|PackedMatrix>::row_into"),
    ("dense kernel", r"linear_panels_into|linear_into|kernels::linear|Matrix>::project|"
                     r"PackedMatrix>::project"),
    ("fake quant A8", r"fake_quant"),
    ("Hadamard MLP", r"hadamard|HadamardMlp"),
    ("mHC/sinkhorn", r"sinkhorn|Mhc>|mhc::"),
    ("Engram", r"Engram|engram"),
    ("cross-attention", r"CrossAttention|cross_attention"),
    ("self-attention", r"attention::|SelfAttention|Rope"),
    ("conv", r"conv_in|conv_mix|conv_out|causal_taps|value_conv"),
    ("norms", r"rms_norm|rms_unit|layer_norm"),
    ("stem", r"Stem>::|model::stem::"),
    ("FFT", r"audio::fft"),
    ("mel/log", r"audio::mel"),
    ("VAD", r"whistle::vad::"),
    ("head/logits", r"Decoder>::logits|embed|Tokenizer"),
    ("beam search", r"transcribe::search"),
]


def demangle(names):
    names = list(names)
    if not names or not shutil.which("c++filt"):
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return {n: re.sub(r"\.llvm\.\d+", "", d).replace(" ()", "") for n, d in zip(names, out)}


def classify(stack, rules, default):
    for name, pattern in rules:
        if any(re.search(pattern, frame) for frame in stack):
            return name
    return default


def kind_of(stack):
    for frame in stack:  # leaf first
        for name, pattern in KINDS:
            if re.search(pattern, frame):
                return name
    return "other"


def main():
    args = sys.argv[1:]
    under, top = r"Whistle>::transcribe_wav", 25
    if "--under" in args:
        i = args.index("--under")
        under = args[i + 1]
        del args[i:i + 2]
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    with open(args[0], encoding="utf-8") as f:
        prof = json.load(f)
    nodes = {n["id"]: n for n in prof["nodes"]}
    parent = {}
    for n in prof["nodes"]:
        for c in n.get("children", []):
            parent[c] = n["id"]
    names = demangle({n["callFrame"]["functionName"] for n in prof["nodes"]})

    def stack(node_id):
        out = []
        while node_id is not None:
            out.append(names[nodes[node_id]["callFrame"]["functionName"]])
            node_id = parent.get(node_id)
        return out  # leaf first

    per_node = collections.Counter()
    for sample, delta in zip(prof["samples"], prof["timeDeltas"]):
        per_node[sample] += delta
    selfs, stages, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    total = 0
    for node_id, us in per_node.items():
        st = stack(node_id)
        if not any(re.search(under, f) for f in st):
            continue
        total += us
        selfs[st[0]] += us
        stage = classify(st, STAGES, "everything else")
        stages[stage] += us
        kinds[(stage, kind_of(st))] += us
    if total == 0:
        sys.exit(f"no samples under {under!r}")
    print(f"{total / 1e3:.0f} ms sampled under {under!r}\n")
    print(f"top {top} functions by self time")
    for name, us in selfs.most_common(top):
        short = name if len(name) <= 110 else name[:107] + "..."
        print(f"  {us / total * 100:5.1f}%  {us / 1e3:7.1f} ms  {short}")
    print("\nbuckets (stage, then what the innermost recognisable frame does)")
    for stage, us in stages.most_common():
        print(f"  {us / total * 100:5.1f}%  {stage}")
        for (s, kind), kus in kinds.most_common():
            if s == stage and kus / total >= 0.002:
                print(f"           {kus / total * 100:5.1f}%  {kind}")


if __name__ == "__main__":
    main()
