"""Versioned parameter files for the portable C kernel.

A parameter file holds everything `embedded/gru.c` needs besides code: GRU weights,
input normalization and the feature mask. It carries a version and a SHA-256 of its
contents, is validated before use, and is the single source for the generated
`weights.h` and for the PyTorch reference used in back-to-back tests.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SCHEMA = "edge-soc-params/1"
INPUTS = ["voltage_v", "current_a", "temp_c"]
UNITS = {"voltage_v": "V", "current_a": "A, discharge positive", "temp_c": "degC", "output": "SoC ratio"}
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def shapes(hidden):
    """Tensor names as used in gru.c, with their shapes for a given hidden size."""
    return {
        "input_weight": [3, 3],
        "input_bias": [3],
        "gru_weight_ih_l0": [3 * hidden, 3],
        "gru_weight_hh_l0": [3 * hidden, hidden],
        "gru_bias_ih_l0": [3 * hidden],
        "gru_bias_hh_l0": [3 * hidden],
        "head_weight": [1, hidden],
        "head_bias": [1],
        "feature_mean": [3],
        "feature_std": [3],
        "feature_mask": [3],
    }


def _float32_list(values):
    # float(np.float32(v)) is the exact float32 value; JSON round-trips it without loss.
    return [float(v) for v in np.asarray(values, dtype=np.float32).ravel()]


def digest(params):
    """SHA-256 over the canonical form of everything the kernel consumes."""
    payload = dict(hidden=params["hidden"], inputs=params["inputs"], tensors=params["tensors"])
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def from_state_dict(state_dict, hidden, mean, std, mask=(1, 1, 1), name="unnamed", version="0.0.0",
                    provenance=None):
    tensors = {key.replace(".", "_"): _float32_list(value.detach().cpu().numpy())
               for key, value in state_dict.items()}
    tensors.update(feature_mean=_float32_list(mean), feature_std=_float32_list(std),
                   feature_mask=_float32_list(mask))
    ordered = {key: tensors[key] for key in shapes(hidden) if key in tensors}
    ordered.update({key: value for key, value in tensors.items() if key not in ordered})
    params = dict(
        schema=SCHEMA,
        name=name,
        version=version,
        hidden=int(hidden),
        inputs=list(INPUTS),
        units=dict(UNITS),
        provenance=dict(provenance or {}),
        tensors=ordered,
    )
    params["sha256"] = digest(params)
    validate(params)
    return params


def validate(params):
    """Raise ValueError on anything the C kernel could not use safely."""
    if params.get("schema") != SCHEMA:
        raise ValueError(f"Unknown schema: {params.get('schema')!r}")
    if not isinstance(params.get("name"), str) or not NAME.match(params["name"]):
        raise ValueError("Name must be letters, digits, '.', '_' or '-'")
    if not isinstance(params.get("version"), str) or not VERSION.match(params["version"]):
        raise ValueError("Version must look like 1.2.3")
    hidden = params.get("hidden")
    if not isinstance(hidden, int) or isinstance(hidden, bool) or not 1 <= hidden <= 1024:
        raise ValueError("Hidden size must be an integer from 1 to 1024")
    if params.get("inputs") != INPUTS:
        raise ValueError(f"Inputs must be {INPUTS}")
    tensors = params.get("tensors", {})
    expected = shapes(hidden)
    if set(tensors) != set(expected):
        missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        raise ValueError(f"Tensor names differ from the kernel: missing {missing}, unexpected {extra}")
    for key, shape in expected.items():
        values = tensors[key]
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise ValueError(f"{key} must hold {math.prod(shape)} values for shape {shape}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in values):
            raise ValueError(f"{key} holds a non-finite or non-numeric value")
    if any(v <= 0 for v in tensors["feature_std"]):
        raise ValueError("feature_std must be positive")
    if any(v not in (0.0, 1.0) for v in tensors["feature_mask"]):
        raise ValueError("feature_mask must be 0 or 1")
    if params.get("sha256") != digest(params):
        raise ValueError("Checksum mismatch: the tensors changed without a new sha256")


def load(path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(params)
    return params


def save(params, path):
    validate(params)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readable layout: metadata indented, one line per tensor.
    marker = "__TENSORS__"
    text = json.dumps({key: value for key, value in params.items() if key != "tensors"} | {"tensors": marker},
                      indent=2)
    body = ",\n".join(f"    {json.dumps(key)}: {json.dumps(values)}" for key, values in params["tensors"].items())
    text = text.replace(json.dumps(marker), "{\n" + body + "\n  }")
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    if load(path) != params:
        raise AssertionError("Parameter file did not round-trip")


def identity(params):
    return f"{params['name']} {params['version']} sha256:{params['sha256'][:16]}"


def render_header(params):
    validate(params)
    lines = [
        f"/* Generated from parameter file {params['name']} {params['version']}. Do not edit. */",
        f"#define HIDDEN {params['hidden']}",
        f"#define PARAMS_ID \"{identity(params)}\"",
    ]
    for key in shapes(params["hidden"]):
        values = params["tensors"][key]
        lines.append(f"static const float {key}[{len(values)}] = {{"
                     + ",".join(f"{v:.9e}f" for v in values) + "};")
    return "\n".join(lines) + "\n"


def float_count(params):
    return sum(len(values) for values in params["tensors"].values())


def to_model(params):
    """PyTorch reference built from the same file the C header is generated from."""
    import torch

    from battery_shift.models import GRU

    validate(params)
    model = GRU(params["hidden"]).eval()
    state = {}
    for key, tensor in model.state_dict().items():
        values = params["tensors"][key.replace(".", "_")]
        state[key] = torch.tensor(values, dtype=torch.float32).reshape(tensor.shape)
    model.load_state_dict(state)
    return model


def normalize(params, raw):
    t = params["tensors"]
    mean, std, mask = (np.asarray(t[k], dtype=np.float32) for k in ("feature_mean", "feature_std",
                                                                       "feature_mask"))
    return ((np.asarray(raw, dtype=np.float32) - mean) / std * mask).astype(np.float32)
