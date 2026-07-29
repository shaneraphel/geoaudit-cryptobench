#!/usr/bin/env python3
"""Read the weights of CryptoBench's pLM-NN baseline out of its published
SavedModel, without TensorFlow.

The baseline is a Keras SavedModel: a graph in ``saved_model.pb`` and a
TensorFlow v2 checkpoint in ``variables/``. Loading it the intended way needs
TensorFlow plus ``tensorflow_addons`` for the metric the authors compiled it
with, which is a large dependency to install in order to read six arrays, and it
would make the baseline's weights arrive through a path nothing else in this
repository can check.

So the checkpoint is read directly. Both formats involved are specified:

  ``variables.index`` is a LevelDB table. Its footer gives the offset of the
  index block, whose entries point at data blocks, whose entries are
  prefix-compressed key/value pairs. The keys are variable names and the values
  are ``BundleEntryProto`` messages, which are parsed field by field for the
  dtype, shape, and the byte range of the tensor.

  ``variables.data-00000-of-00001`` is those byte ranges, little-endian.

Nothing is inferred from the layout. Every array arrives at an offset the index
stated, with a shape the index stated, and the shapes are then required to
compose into one network: 2560 in, two hidden widths, two classes out. The
checkpoint also holds the Adam moments, three times the size of the weights, and
naming a moment tensor by accident would give a network that runs and predicts
noise -- so the names are matched exactly rather than by position or by size.

The activations are read from the graph too, not assumed. A dense stack is only
determined once you know what sits between the layers, and ``relu, relu,
softmax`` is a guess until the op names confirm it.

Usage: PYTHONPATH=src:tools python3.12 tools/export_plmnn_weights.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct

import numpy as np

from pocket_bench.paths import ROOT

OSF = ROOT / "data/cryptobench_apo/_osf/cryptobench/benchmark/best_trained"
GRAPH = OSF / "saved_model.pb"
INDEX = OSF / "variables/variables.index"
DATA = OSF / "variables/variables.data-00000-of-00001"
NPZ = ROOT / "results/baselines/PLMNN_WEIGHTS.npz"
OUT = ROOT / "results/baselines/PLMNN_NETWORK.json"
SCHEMA = "geoaudit.plmnn_network.v1"

# The digests OSF reports for the three files, already verified on download by
# tools/fetch_official_data.py. Repeated here so that this tool refuses to read a
# checkpoint that is not the published one.
EXPECTED = {
    "saved_model.pb":
        "7e21232169b9a66fd17f7fafdf533dc1d7c5d05a022cacba842e91ae2fb15a6e",
    "variables/variables.index":
        "f37c8b7d24ff47ad74a37a9cbf0cee218ace5d03ad66450583017624a4f897cb",
    "variables/variables.data-00000-of-00001":
        "a004b4f7fd928caaf0efbadd7f0197a077fd51119115d5dd238481713f47ff02",
}
# LevelDB table footer, and the DT_FLOAT enum value in TensorFlow's DataType.
TABLE_MAGIC = 0xDB4775248B80FB57
FOOTER_BYTES = 48
DT_FLOAT = 1
LAYERS = ("dense_3", "dense_4", "dense_5")


# --- the two published formats ----------------------------------------------- #
def _varint(buf: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = buf[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _block(buf: bytes, off: int, size: int) -> list[tuple[bytes, bytes]]:
    """Entries of one LevelDB block, undoing the prefix compression of the keys."""
    blk = buf[off:off + size]
    if len(blk) != size:
        raise SystemExit(f"index truncated: wanted {size} bytes at {off}")
    if buf[off + size] != 0:
        raise SystemExit("index block is compressed; only stored blocks are read")
    n_restart = struct.unpack("<I", blk[-4:])[0]
    end = len(blk) - 4 - 4 * n_restart
    i, key, out = 0, b"", []
    while i < end:
        shared, i = _varint(blk, i)
        fresh, i = _varint(blk, i)
        vlen, i = _varint(blk, i)
        key = key[:shared] + blk[i:i + fresh]
        i += fresh
        out.append((key, blk[i:i + vlen]))
        i += vlen
    return out


def _table_entries(buf: bytes) -> list[tuple[bytes, bytes]]:
    if struct.unpack("<Q", buf[-8:])[0] != TABLE_MAGIC:
        raise SystemExit(f"{INDEX.name} is not a LevelDB table")
    foot = buf[-FOOTER_BYTES:]
    i = 0
    _, i = _varint(foot, i)  # the metaindex handle, which holds no variables
    _, i = _varint(foot, i)
    off, i = _varint(foot, i)
    size, i = _varint(foot, i)
    out: list[tuple[bytes, bytes]] = []
    for _, handle in _block(buf, off, size):
        j = 0
        o, j = _varint(handle, j)
        s, j = _varint(handle, j)
        out += _block(buf, o, s)
    return out


def _message(buf: bytes) -> dict[int, object]:
    """Protobuf fields by number, enough of the wire format for BundleEntryProto."""
    out: dict[int, object] = {}
    i = 0
    while i < len(buf):
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            v, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wire == 5:
            v, i = struct.unpack("<I", buf[i:i + 4])[0], i + 4
        elif wire == 1:
            v, i = struct.unpack("<Q", buf[i:i + 8])[0], i + 8
        else:
            raise SystemExit(f"unhandled protobuf wire type {wire}")
        out[field] = v
    return out


def _shape(buf: bytes) -> list[int]:
    """TensorShapeProto: repeated Dim dim = 2, each with int64 size = 1."""
    dims, i = [], 0
    while i < len(buf):
        _, i = _varint(buf, i)
        n, i = _varint(buf, i)
        dims.append(_message(buf[i:i + n]).get(1))
        i += n
    return [int(d) for d in dims]


# --- the network ------------------------------------------------------------- #
def _digest(path: str) -> str:
    return hashlib.sha256((OSF / path).read_bytes()).hexdigest()


def _activations(graph: bytes) -> dict[str, list[str]]:
    """The ops each dense layer's namespace contains, in the graph's own naming."""
    seen: dict[str, list[str]] = {}
    for name in LAYERS:
        ops = {m.group(1).decode() for m in re.finditer(
            rb"/" + name.encode() + rb"/([A-Za-z][A-Za-z_0-9]*)", graph)}
        seen[name] = sorted(ops)
    return seen


def build() -> tuple[dict, dict[str, np.ndarray]]:
    for path, want in EXPECTED.items():
        if not (OSF / path).exists():
            raise SystemExit(
                f"missing {(OSF / path).relative_to(ROOT)}; fetch it with\n"
                f"  python3.12 tools/fetch_official_data.py --fetch "
                f"/cryptobench/benchmark/best_trained/{path}")
        got = _digest(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got[:12]} is not the published "
                             f"{want[:12]}; refusing to read it")

    entries = dict(_table_entries(INDEX.read_bytes()))
    raw = DATA.read_bytes()
    weights: dict[str, np.ndarray] = {}
    described = []
    for k, (name, role) in enumerate(
            [(n, r) for n in LAYERS for r in ("kernel", "bias")]):
        layer = k // 2
        key = (f"layer_with_weights-{layer}/{role}"
               f"/.ATTRIBUTES/VARIABLE_VALUE").encode()
        if key not in entries:
            raise SystemExit(f"the checkpoint has no {key.decode()}")
        e = _message(entries[key])
        if e.get(1) != DT_FLOAT:
            raise SystemExit(f"{key.decode()} is dtype {e.get(1)}, not float32")
        shape = _shape(e.get(2, b""))
        off, size = int(e.get(4, 0)), int(e.get(5, 0))
        n = int(np.prod(shape)) if shape else 1
        if size != 4 * n:
            raise SystemExit(f"{key.decode()}: {size} bytes for {n} float32")
        weights[f"{name}_{role}"] = np.frombuffer(
            raw[off:off + size], dtype="<f4").reshape(shape).copy()
        described.append({"array": f"{name}_{role}",
                          "checkpoint_key": key.decode(),
                          "shape": shape, "byte_offset": off,
                          "byte_length": size})

    widths = [weights["dense_3_kernel"].shape[0]] + [
        weights[f"{n}_kernel"].shape[1] for n in LAYERS]
    for a, b in zip(LAYERS, LAYERS[1:]):
        if weights[f"{a}_kernel"].shape[1] != weights[f"{b}_kernel"].shape[0]:
            raise SystemExit(f"{a} does not feed {b}")
    for name in LAYERS:
        if weights[f"{name}_kernel"].shape[1] != weights[f"{name}_bias"].shape[0]:
            raise SystemExit(f"{name}: bias does not match its kernel")
    if widths[0] != 2560:
        raise SystemExit(f"the input width is {widths[0]}, not the 2560 of "
                         "ESM2-3B; this is not the published baseline")
    if widths[-1] != 2:
        raise SystemExit(f"the output width is {widths[-1]}, not 2")

    ops = _activations(GRAPH.read_bytes())
    expected_act = {"dense_3": "Relu", "dense_4": "Relu", "dense_5": "Softmax"}
    for name, act in expected_act.items():
        if act not in ops[name]:
            raise SystemExit(f"{name} has ops {ops[name]}, with no {act}; the "
                             "forward pass this repository implements would be "
                             "the wrong network")
    # A dense layer is a matmul, a bias add and an activation, over the two
    # variables it owns. Anything else in the namespace -- a dropout that is live
    # at inference, a normalisation, a second activation -- would mean the
    # forward pass here is not the published one.
    benign = ("MatMul", "BiasAdd", "ReadVariableOp", "kernel", "bias")
    stray = {n: [o for o in ops[n] if o not in benign + (expected_act[n],)]
             for n in LAYERS}
    if any(stray.values()):
        raise SystemExit(f"unexpected ops inside the dense layers: {stray}")

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "what_this_is": (
            "the weights and the forward pass of CryptoBench's own pLM-NN "
            "baseline, read out of the SavedModel the authors published"),
        "why_not_tensorflow": (
            "the SavedModel was compiled with a tensorflow_addons metric, so "
            "loading it the intended way needs TensorFlow and that package "
            "installed to read six arrays. The checkpoint format is specified, "
            "so it is parsed instead, and every array is taken at the offset "
            "and shape the checkpoint's own index states"),
        "source": "osf.io/pz4a9, /cryptobench/benchmark/best_trained",
        "source_sha256": {k: _digest(k) for k in EXPECTED},
        "architecture": {
            "widths": widths,
            "activations": [expected_act[n] for n in LAYERS],
            "n_parameters": int(sum(a.size for a in weights.values())),
            "forward_pass": ("p = softmax(relu(relu(x @ W0 + b0) @ W1 + b1) "
                             "@ W2 + b2), and column 1 is the binding class, "
                             "which is how the authors' own example reads it"),
        },
        "ops_found_in_the_graph": ops,
        "arrays": described,
        "adam_moments_in_the_checkpoint_were_not_read": (
            "the checkpoint is 8.67 MB for 2.89 MB of weights because it also "
            "stores the optimizer's two moments per parameter. Those keys are "
            "not named here, and a network built from a moment tensor would "
            "run and predict nothing"),
        "weights_sha256": hashlib.sha256(
            b"".join(weights[f"{n}_{r}"].tobytes()
                     for n in LAYERS for r in ("kernel", "bias"))).hexdigest(),
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "a published baseline's weights are somebody else's fitted model. "
            "No label, prediction or metric of the test fold is opened here"),
    }
    return doc, weights


def forward(x: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """The published network: relu, relu, softmax. Returns both class columns."""
    h = np.maximum(x @ w["dense_3_kernel"] + w["dense_3_bias"], 0.0)
    h = np.maximum(h @ w["dense_4_kernel"] + w["dense_4_bias"], 0.0)
    z = h @ w["dense_5_kernel"] + w["dense_5_bias"]
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load() -> dict[str, np.ndarray]:
    """The exported weights, checked against the digest the export recorded."""
    if not NPZ.exists() or not OUT.exists():
        raise SystemExit(f"missing {NPZ.relative_to(ROOT)}; run "
                         "tools/export_plmnn_weights.py")
    w = {k: v for k, v in np.load(NPZ).items()}
    doc = json.loads(OUT.read_text())
    got = hashlib.sha256(
        b"".join(w[f"{n}_{r}"].tobytes()
                 for n in LAYERS for r in ("kernel", "bias"))).hexdigest()
    if got != doc["weights_sha256"]:
        raise SystemExit(f"{NPZ.name} does not match the digest recorded in "
                         f"{OUT.name}")
    return w


def _report(d: dict) -> None:
    a = d["architecture"]
    print("  ".join(f"{w}" for w in a["widths"]) +
          f"   activations {', '.join(a['activations'])}")
    print(f"  {a['n_parameters']} parameters, "
          f"digest {d['weights_sha256'][:16]}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") is not None:
        bad.append("reading published weights must not claim a read index")

    # The checkpoint and the exported arrays are both excluded from the tree,
    # being reproducible from the digests recorded here. So the gate checks as far
    # as the clone allows and says how far that was, rather than failing a machine
    # that has not fetched 12 MB from OSF.
    depth = "the recorded provenance only"
    if all((OSF / k).exists() for k in EXPECTED):
        depth = "a full re-extraction from the published checkpoint"
        try:
            live, w = build()
        except SystemExit as e:
            print(f"FAIL {OUT.relative_to(ROOT)}: {e}")
            return 1
        if d.get("weights_sha256") != live["weights_sha256"]:
            bad.append("the weights no longer follow from the published "
                       "checkpoint")
        if d.get("architecture") != live["architecture"]:
            bad.append("the architecture read from the checkpoint changed")
        if NPZ.exists():
            have = dict(np.load(NPZ).items())
            if set(have) != set(w):
                bad.append("the exported npz holds a different set of arrays")
            else:
                moved = sorted(k for k in w
                               if not np.array_equal(have[k], w[k]))
                if moved:
                    bad.append(f"exported arrays differ from the checkpoint: "
                               f"{moved}")
    elif NPZ.exists():
        depth = "the exported arrays against their recorded digest"
        try:
            load()
        except SystemExit as e:
            bad.append(str(e))
    if not d.get("weights_sha256"):
        bad.append("no weights digest is recorded, so nothing can be checked")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"  checked against {depth}")
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    doc, weights = build()
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **weights)
    OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} and {NPZ.relative_to(ROOT)}\n")
    _report(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
