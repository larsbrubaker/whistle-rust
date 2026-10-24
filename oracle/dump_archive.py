# The archive and tokenizer oracle fixture (test-data/fixtures/archive/).
"""Writes what the Rust `.cact` reader, CQ dequantizer and tokenizer are tested against, from
upstream's own `export.read_export`, `parse_tokenizer_blob` and `RefTokenizer` (needle-reference,
pinned in REFERENCE) run on models/whistle.cact:

- manifest.json: the header geometry, the stored codebook, and for every tensor in file order its
  upstream export name, dtype, shape, group and bits plus checksums of its dequantized values
  (element count, float64 sum and L2 norm, and up to 64 values at evenly spaced flat indices);
- full .npy arrays for a few tensors (the smallest CQ2 and CQ4 matrices, an fp16 vector, `mel`,
  `hada_p1`);
- tokenizer cases: `encode` ids and `decode(encode(s))` for every clip's text and some extra
  strings, and every clip's reference text tokens decoded.

    oracle/.venv/bin/python -I oracle/dump_archive.py

Regenerate only when REFERENCE changes (CLAUDE.md "The contract").
"""
import glob
import json
import os
import sys

import numpy as np

# `python -I` leaves the script's folder off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cact_params as cp  # noqa: E402 (also puts needle-reference on sys.path)

from needle.model import export  # noqa: E402
from needle.model.checkpoints import read_checkpoint  # noqa: E402
from needle.model.export import RefTokenizer, parse_tokenizer_blob, read_export  # noqa: E402
from needle.model.whistle import WhistleConfig  # noqa: E402

ROOT = cp.ROOT
CACT = os.path.join(ROOT, "models", "whistle.cact")
CHECKPOINT = os.path.join(ROOT, "models", "whistle.safetensors")
OUT = os.path.join(ROOT, "test-data", "fixtures", "archive")
SAMPLES = 64
DTYPES = {export.FP16: "fp16", export.FP32: "fp32", export.CQ: "cq", export.RAW: "raw"}

EXTRA_TEXTS = [
    "",
    " ",
    "a",
    "  leading spaces",
    "trailing spaces  ",
    "double  space  inside",
    "Größe über Straße",
    "café à la crème, s'il vous plaît",
    "¿Dónde está la estación? ¡Vamos!",
    "Zażółć gęślą jaźń",
    "Łódź, Kraków i Gdańsk",
    "3.14159 + 2,718 = 5.86 (approx.) 100% #1",
    "Bitte <|de|> mach den Würfel größer",
    "<|en|><|fr|>",
    "emoji 🎉🚀 and CJK 漢字かな",
    "tab\tnewline\nend",
    "Mixed-Case HTTP://example.com/path?q=1&r=2",
]


def _header_geometry(raw):
    hdr = export.struct.unpack_from(export._HDR_FMT, raw, 0)
    names = ("tag", "num_tensors", "codebook_len", "kv_window", "kv_bits", "vocab", "out_vocab",
             "d_model", "num_heads", "num_kv_heads", "num_layers", "qk_head_dim", "v_head_dim",
             "max_seq_len", "hada_n", "mhc_lanes", "sliding_window", "global_mask_lo",
             "global_mask_hi", "qkv_conv_taps", "engram_slots", "engram_sub_dim",
             "num_engram_tables", "engram_conv_taps", "engram_conv_dilation", "engram_seed_heads")
    geometry = {name: int(v) for name, v in zip(names, hdr[:26])}
    geometry["global_mask"] = geometry.pop("global_mask_lo") | (geometry.pop("global_mask_hi") << 32)
    geometry["engram_orders"] = [int(v) for v in hdr[27:27 + hdr[26]]]
    geometry["engram_sites"] = [int(v) for v in hdr[32:32 + hdr[31]]]
    geometry["rope_theta"] = float(hdr[48])
    return geometry


def _checksums(value):
    flat = np.asarray(value, np.float32).reshape(-1)
    n = flat.size
    idx = np.unique(np.linspace(0, n - 1, min(SAMPLES, n)).round().astype(np.int64)) if n else []
    f64 = flat.astype(np.float64)
    return {"count": int(n), "sum": float(f64.sum()), "l2": float(np.sqrt((f64 * f64).sum())),
            "sample_indices": [int(i) for i in idx], "samples": [float(flat[i]) for i in idx]}


def _full_arrays(names, records):
    def smallest(pred):
        found = [(int(np.prod(r[1])), n) for n, r in zip(names, records) if pred(r)]
        return min(found)[1] if found else None

    chosen = [smallest(lambda r: r[0] == export.CQ and r[3] == 2),
              smallest(lambda r: r[0] == export.CQ and r[3] == 4),
              "final_norm", "mel", "hada_p1"]
    return [n for n in chosen if n is not None]


def _tokenizer_cases(tok):
    fixtures = os.path.join(ROOT, "test-data", "fixtures")
    with open(os.path.join(fixtures, "index.json"), encoding="utf-8") as f:
        names = json.load(f)["clips"]
    clips = []
    for name in names:
        with open(os.path.join(fixtures, name, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        clips.append((m["clip"], m["text"], m["tokens"]))
    encode = []
    for text in [t for _, t, _ in clips] + EXTRA_TEXTS:
        ids = tok.encode(text)
        encode.append({"text": text, "ids": [int(i) for i in ids], "decoded": tok.decode(ids)})
    decode = []
    for clip, text, tokens in clips:
        ids = [int(i) for i in tokens[2:-1]]
        decoded = tok.decode(ids)
        if decoded != text:
            raise ValueError(f"{clip}: reference tokens decode to {decoded!r}, manifest says {text!r}")
        decode.append({"clip": clip, "ids": ids, "text": decoded})
    return encode, decode


def main():
    with open(CACT, "rb") as f:
        raw = f.read()
    meta, tensors = read_export(CACT)
    records = cp.directory(CACT)
    ckpt = read_checkpoint(CHECKPOINT)
    names = cp.tensor_names(ckpt["params"], WhistleConfig.from_saved(ckpt["config"])) + ["tokenizer"]
    if len(names) != len(tensors):
        raise ValueError(f"{len(tensors)} archive tensors, {len(names)} export names")
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.npy")):
        os.remove(old)
    full = _full_arrays(names, records)
    entries = []
    for i, (name, (dtype, shape, group, bits), value) in enumerate(zip(names, records, tensors)):
        entry = {"index": i, "name": name, "dtype": DTYPES[dtype], "shape": [int(d) for d in shape],
                 "group": int(group), "bits": int(bits)}
        if dtype == export.RAW:
            entry["nbytes"] = len(value)
        else:
            entry.update(_checksums(value))
            if name in full:
                file = f"{i:03d}_{name.replace('.', '_')}.npy"
                np.save(os.path.join(OUT, file), np.ascontiguousarray(value, np.float32))
                entry["file"] = file
        entries.append(entry)
    blob = next(t for t in tensors if isinstance(t, (bytes, bytearray)))
    tok_meta = parse_tokenizer_blob(blob)
    encode, decode = _tokenizer_cases(RefTokenizer(tok_meta))
    manifest = {
        "geometry": _header_geometry(raw),
        "codebook": [float(v) for v in meta["codebook"]],
        "tensors": entries,
        "tokenizer": {
            "pieces": len(tok_meta["pieces"]), "pad_id": tok_meta["pad_id"],
            "eos_id": tok_meta["eos_id"], "bos_id": tok_meta["bos_id"], "unk_id": tok_meta["unk_id"],
            "add_dummy_prefix": tok_meta["add_dummy_prefix"],
            "byte_fallback": tok_meta["byte_fallback"],
            "user_defined": {p: i for i, (p, t) in enumerate(zip(tok_meta["pieces"], tok_meta["types"]))
                             if t == export.TK_USER_DEFINED},
            "encode": encode, "decode": decode,
        },
    }
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"{len(entries)} tensors, full arrays: {full}")


if __name__ == "__main__":
    main()
