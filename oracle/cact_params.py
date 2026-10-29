# whistle.cact -> the Flax parameter tree the JAX reference runs on, with its checks.
"""Rebuilds the Flax parameter tree from a `.cact` archive, so the JAX reference runs on exactly
the weights the Rust port loads (dequantized CQ matrices, fp16 vectors), not on the fp32
checkpoint they were quantized from.

The tensor order and each tensor's name come from upstream's own `export._tensors` (run with
its packers stubbed out), so this file only supplies the inverse of each export transform.
`verify` checks the rebuild against the fp32 checkpoint: every fp16 tensor must equal the
checkpoint value rounded to fp16, and every CQ tensor must equal upstream's `cq_quantize` of the
checkpoint value at the width the archive records, except nearest-codeword ties that float
noise flips (at most 0.02% of 128-wide groups may differ; a wrong transpose makes nearly all
differ); and the Hadamard permutations and mel matrix stored in the archive must equal
upstream's. A wrong transpose or reshape fails loudly.

The archive mixes widths: 145 CQ2 matrices, 7 CQ4 (the embedding and the mHC `phi`
projections), 524 fp16 vectors, 4 fp32 (permutations, mel) and the raw tokenizer. Upstream's
public exporter writes only CQ4, so it did not produce this file; the format and `read_export`
handle every width, and the checkpoint supplies only the tree's shape and the config.
"""
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "needle-reference"))
os.environ["NEEDLE_TELEMETRY"] = "0"

from needle.model import export  # noqa: E402
from needle.model.checkpoints import flatten, read_checkpoint, unflatten  # noqa: E402
from needle.model.whistle import WhistleConfig, mel_matrix  # noqa: E402

FP16, FP32, CQ, RAW = export.FP16, export.FP32, export.CQ, export.RAW


def directory(path):
    """The archive's tensor records: (dtype, shape, group, bits) in file order."""
    with open(path, "rb") as f:
        raw = f.read()
    _, num_tensors, cb_n = struct.unpack_from(export._HDR_FMT, raw, 0)[:3]
    off = struct.calcsize(export._HDR_FMT) + cb_n * 4
    records = []
    for _ in range(num_tensors):
        rec = struct.unpack_from(export._REC_FMT, raw, off)
        off += export.REC_SIZE
        records.append((rec[0], tuple(rec[3:3 + rec[1]]), rec[9], rec[10]))
    return records


def tensor_names(params, cfg):
    """Upstream's export order for this config, without packing anything."""
    saved = export._q, export._fp16

    def name_only(name, *_args, **_kwargs):
        return export._Tensor(name, 0, (), b"")

    export._q, export._fp16 = name_only, name_only
    try:
        return [t.name for t in export._tensors(params, cfg.needle(), export.WEIGHT_BITS, 128, audio=cfg)]
    finally:
        export._q, export._fp16 = saved


def _leaf(tree, path):
    node = tree
    for key in path:
        node = node[key]
    return node


_WRITES = []


def _set(tree, path, value, layer=None):
    _WRITES.append("/".join(path))
    parent = _leaf(tree, path[:-1])
    if layer is None:
        parent[path[-1]] = np.asarray(value, np.float32).reshape(np.shape(parent[path[-1]]))
    else:
        target = parent[path[-1]]
        target[layer] = np.asarray(value, np.float32).reshape(target.shape[1:])


PROJ = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")
DEC_SCALES = {"norm_in": "ZCRMSNorm_0", "post_norm": "post_attn_norm", "pre_hada": "pre_hada_norm",
              "cross_norm": "cross_norm", "post_cross_norm": "post_cross_norm"}


def _place(tree, name, value, cfg):
    """Writes one dequantized archive tensor back where export took it from. Returns False for
    tensors that are not parameters (permutations, the mel matrix, the manifest)."""
    parts = name.split(".")
    if name in ("hada_p1", "hada_p2", "mel", "audio.manifest"):
        return False
    if name == "embedding":
        _set(tree, ("embedding", "embedding"), value)
    elif name == "final_norm":
        _set(tree, ("stack", "final_norm", "scale"), value)
    elif name == "pe_gate":
        _set(tree, ("pe_gate",), value)
    elif name.startswith("mhc_"):
        _place_mhc(tree, ("stack",), name, value, cfg.dec_layers)
    elif name.startswith("encoder.") and len(parts) == 2:
        if parts[1] == "final_norm":
            _set(tree, ("encoder", "final_norm", "scale"), value)
        else:
            _place_mhc(tree, ("encoder",), parts[1], value, cfg.enc_layers)
    elif name.startswith("layer"):
        _place_decoder(tree, int(parts[0][5:]), parts[1:], value)
    elif name.startswith("encoder"):
        _place_encoder(tree, int(parts[0][7:]), parts[1:], value, cfg)
    elif name.startswith("engram"):
        e = ("engrams_" + parts[0][6:],)
        if parts[1] == "tables":
            _set(tree, e + ("embedding",), value)
        elif parts[1] == "taps":
            _set(tree, e + ("taps",), value)
        else:
            _set(tree, e + (parts[1], "kernel"), np.asarray(value).T)
    elif name.startswith("stem."):
        key = parts[1]
        if key in ("w",) or key.startswith("dw_"):
            _set(tree, ("stem", key), value)
        else:
            _set(tree, ("stem", key, "kernel"), np.asarray(value).T)
    else:
        raise KeyError(f"no inverse for archive tensor {name}")
    return True


def _place_mhc(tree, base, key, value, layers):
    if key.startswith("mhc_phi"):
        target = _leaf(tree, base + (key,))
        L, nC, lanes = target.shape
        value = np.asarray(value).reshape(L, lanes, nC).transpose(0, 2, 1)
    _set(tree, base + (key,), value)


def _place_decoder(tree, i, rest, value):
    b = ("stack", "layers", "block")
    key = rest[0]
    if key in DEC_SCALES:
        _set(tree, b + (DEC_SCALES[key], "scale"), value, i)
    elif key in PROJ:
        _set(tree, b + ("self_attn", key, "kernel"), np.asarray(value).T, i)
    elif key in ("q_taps", "k_taps", "v_taps"):
        _set(tree, b + ("self_attn", key), value, i)
    elif key in ("q_norm", "k_norm"):
        _set(tree, b + ("self_attn", key, "scale"), value, i)
    elif key in ("attn_gate", "cross_gate"):
        _set(tree, b + (key,), value, i)
    elif key in export.HADAMARD:
        _set(tree, b + ("hadamard_mlp", key), value, i)
    elif key == "cross":
        _set(tree, b + ("cross_attn", rest[1], "kernel"), np.asarray(value).T, i)
    elif key in ("cross_q_norm", "cross_k_norm"):
        _set(tree, b + ("cross_attn", key[6:], "scale"), value, i)
    else:
        raise KeyError(f"no inverse for decoder tensor {key}")


def _place_encoder(tree, i, rest, value, cfg):
    b = ("encoder", "layers", "block")
    key = rest[0]
    if key in ("pre_hada_norm_0", "ZCRMSNorm_0", "post_attn_norm", "conv_norm", "conv_out_norm", "pre_hada_norm"):
        _set(tree, b + (key, "scale"), value, i)
    elif key in PROJ:
        _set(tree, b + ("self_attn", key, "kernel"), np.asarray(value).T, i)
    elif key in ("q_norm", "k_norm"):
        _set(tree, b + ("self_attn", key, "scale"), value, i)
    elif key == "attn_gate":
        _set(tree, b + ("attn_gate",), value, i)
    elif key in ("pw1", "pw2"):
        _set(tree, b + (key, "kernel"), np.asarray(value).T, i)
    elif key == "dw":
        _set(tree, b + ("dw",), value, i)
    elif key in ("hadamard_mlp_0", "hadamard_mlp"):
        _set(tree, b + (key, rest[1]), value, i)
    else:
        raise KeyError(f"no inverse for encoder tensor {key}")


def load(cact_path, checkpoint_path):
    """(params, cfg, names, records, tensors, checkpoint params, source): the archive's weights
    in the checkpoint's tree shape. The checkpoint supplies the tree structure and config only;
    `source` maps each checkpoint leaf to the archive tensor that replaced it."""
    ckpt = read_checkpoint(checkpoint_path)
    cfg = WhistleConfig.from_saved(ckpt["config"])
    template = ckpt["params"]
    names = tensor_names(template, cfg)
    records = directory(cact_path)
    _, tensors = export.read_export(cact_path)
    if len(names) + 1 != len(tensors) or records[-1][0] != RAW:
        raise ValueError(f"{len(tensors)} archive tensors, export order has {len(names)} + tokenizer")
    flat = {k: np.array(v, np.float32, copy=True) for k, v in flatten(template).items()}
    params = unflatten(flat)
    source = {}
    for name, value in zip(names, tensors):
        _WRITES.clear()
        _place(params, name, value, cfg)
        for key in _WRITES:
            source.setdefault(key, name)
    return params, cfg, names, records, tensors, ckpt["params"], source


def verify(params, cfg, names, records, tensors, reference, source):
    """Checks the rebuild leaf by leaf against the checkpoint. Returns a list of report lines;
    raises on any mismatch."""
    import jax.numpy as jnp
    from needle.model.architecture import _hada_perms
    from needle.model.quantize import cq_quantize

    rebuilt, ref = flatten(params), flatten(reference)
    missing = [k for k in ref if k not in source]
    if missing:
        raise ValueError(f"archive never wrote checkpoint leaves: {missing}")
    lines, widths = [], {}
    for (dtype, shape, group, bits), name in zip(records, names):
        widths.setdefault((dtype, bits), []).append(name)
    for (dtype, bits), group_names in sorted(widths.items()):
        label = {FP16: "fp16", FP32: "fp32", CQ: f"cq{bits}"}.get(dtype, str(dtype))
        lines.append(f"{label:5} {len(group_names):4} tensors")

    # Each leaf: the archive record it came from decides the expected value.
    width_of = {}
    for (dtype, _shape, _group, bits), name in zip(records, names):
        width_of[name] = (dtype, bits)
    worst = {}
    for key, have in rebuilt.items():
        want = ref[key]
        kind = width_of[source[key]]
        if kind[0] == FP16:
            expect = np.asarray(want, np.float16).astype(np.float32)
            err = float(np.max(np.abs(have - expect)))
            if err != 0.0:
                raise ValueError(f"{key}: fp16 leaf differs from the checkpoint by {err}")
        else:
            bits = kind[1]
            expect = _cq_reference(key, np.asarray(want, np.float32), bits, cq_quantize, jnp)
            flipped = _differing_groups(key, have, expect)
            worst[key] = (bits, flipped)
            # Float noise can flip a nearest-codeword tie, so a few groups may differ; a wrong
            # transpose or reshape makes nearly all of them differ.
            if flipped > 1e-3:
                raise ValueError(f"{key}: {flipped:.2%} of CQ{bits} groups differ from cq_quantize(checkpoint)")
    for key, (bits, flipped) in sorted(worst.items(), key=lambda kv: -kv[1][1])[:3]:
        lines.append(f"most CQ{bits} tie flips {key}: {flipped:.4%} of groups")

    p1, p2 = _hada_perms(1 << (cfg.dim - 1).bit_length())
    for name, perm in (("hada_p1", p1), ("hada_p2", p2)):
        if not np.array_equal(tensors[names.index(name)], np.asarray(perm, np.float32)):
            raise ValueError(f"{name} differs from architecture._hada_perms")
    if not np.array_equal(tensors[names.index("mel")], mel_matrix(cfg)):
        raise ValueError("archive mel matrix differs from whistle.mel_matrix")
    lines.append("hada_p1/hada_p2 and mel match upstream")
    return lines


def _reduces_rows(key):
    return key.endswith("/kernel") or key.split("/")[-1].startswith("mhc_phi")


def _differing_groups(key, have, expect):
    """The fraction of 128-wide quant groups (along the axis export reduces over) that differ."""
    if _reduces_rows(key):
        have, expect = np.swapaxes(have, -1, -2), np.swapaxes(expect, -1, -2)
    pad = (-have.shape[-1]) % 128
    if pad:
        widths = [(0, 0)] * (have.ndim - 1) + [(0, pad)]
        have, expect = np.pad(have, widths), np.pad(expect, widths)
    have, expect = have.reshape(-1, 128), expect.reshape(-1, 128)
    scale = np.abs(expect).max(-1, keepdims=True) + 1e-12
    return float(np.mean(np.any(np.abs(have - expect) > 1e-5 * scale, axis=-1)))


def _cq_reference(key, w, bits, cq_quantize, jnp):
    """upstream cq_quantize along the axis export reduces over."""
    if _reduces_rows(key):
        moved = np.swapaxes(w, -1, -2)
        return np.swapaxes(np.asarray(cq_quantize(jnp.asarray(moved), bits)), -1, -2)
    return np.asarray(cq_quantize(jnp.asarray(w), bits))


if __name__ == "__main__":
    cact = os.path.join(ROOT, "models", "whistle.cact")
    ckpt = os.path.join(ROOT, "models", "whistle.safetensors")
    for line in verify(*load(cact, ckpt)):
        print(line)
