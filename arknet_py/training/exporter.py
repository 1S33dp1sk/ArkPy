# training/exporter.py — canonical weight export (FP16 RNE) → .safetensors
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from ..utils.hashing import sha256_hex, sha256_domain_hex
from ..utils.iohelpers import ensure_parent_dir
from ..utils.json_canon import dumps_canonical

# We prefer numpy backend for explicit FP16 conversion (RNE via NumPy)
try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    np = None  # type: ignore

# Support both numpy and torch interfaces for safetensors
try:
    from safetensors.numpy import save_file as save_file_np  # type: ignore
except Exception:  # pragma: no cover
    save_file_np = None  # type: ignore

try:
    import torch  # type: ignore
    from safetensors.torch import save_file as save_file_torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    save_file_torch = None  # type: ignore


# ------------------------------ config / metadata ---------------------------

_EXPORT_DOMAIN = b"ARK/MODEL/EXPORT/v1\n"  # domain tag for file-level hash

DEFAULT_EXCLUDE_SUBSTR = (
    "num_batches_tracked",     # common BatchNorm counter
    "_non_persistent_buffers", # PyTorch impl detail
)

@dataclass(frozen=True)
class ExportResult:
    path: str
    size_bytes: int
    sha256: str             # hash over the final .safetensors bytes (domain separated)
    n_tensors: int
    keys: Tuple[str, ...]   # sorted keys
    meta_json: str          # canonical JSON metadata used in file


# ------------------------------ helpers ------------------------------------

def _to_numpy_fp16(x) -> "np.ndarray":
    """
    Convert a tensor/array to contiguous NumPy float16 on CPU using
    IEEE-754 round-to-nearest-even (NumPy does that by default).
    """
    if np is None:
        raise RuntimeError("NumPy is required for canonical FP16 export")

    # torch.Tensor → numpy
    if torch is not None and isinstance(x, torch.Tensor):  # type: ignore[misc]
        with torch.no_grad():  # type: ignore[union-attr]
            arr = x.detach().cpu().contiguous().numpy()
        return arr.astype(np.float16, copy=False)  # RNE via NumPy
    # numpy → numpy
    if isinstance(x, np.ndarray):
        return x.astype(np.float16, copy=False)
    # Python scalar?
    raise TypeError(f"Unsupported tensor type for export: {type(x)}")


def _filter_state_dict(
    state_dict: Dict[str, object],
    exclude_substrings: Iterable[str],
) -> Dict[str, object]:
    excl = tuple(exclude_substrings or ())
    out = {}
    for k, v in state_dict.items():
        if any(s in k for s in excl):
            continue
        out[k] = v
    return out


def _sorted_keys(d: Dict[str, object]) -> List[str]:
    ks = list(d.keys())
    ks.sort()
    return ks


def _stable_metadata(manifest: Dict[str, object]) -> Dict[str, str]:
    """
    Deterministic metadata for safetensors:
      - JSON-canonicalize the manifest we embed
      - store only a single key "arknet_meta" with canonical JSON text
    """
    j = dumps_canonical(manifest).decode("utf-8")
    return {"arknet_meta": j}


def _state_dict_from_model(model) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch not available; pass a state_dict instead of a model")
    if not hasattr(model, "state_dict"):
        raise TypeError("model has no .state_dict()")
    sd = model.state_dict()
    # Eagerly materialize to plain dict (avoid weight sharing views surprises)
    return {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


# ------------------------------ public API ---------------------------------

def export_state_dict_to_safetensors(
    state_dict: Dict[str, object],
    out_path: str,
    *,
    exclude_substrings: Iterable[str] = DEFAULT_EXCLUDE_SUBSTR,
    extra_manifest: Optional[Dict[str, object]] = None,
) -> ExportResult:
    """
    Canonical export:
      1) filter keys (exclude_substrings)
      2) sort keys lexicographically
      3) convert each tensor → FP16 (RNE) NumPy array
      4) write .safetensors with deterministic metadata
      5) hash the resulting bytes with domain separation
    """
    if np is None or save_file_np is None:
        raise RuntimeError("safetensors[numpy] and NumPy are required for canonical export")

    filtered = _filter_state_dict(state_dict, exclude_substrings)
    keys = _sorted_keys(filtered)

    # Build ordered mapping of numpy arrays
    arrays: "OrderedDict[str, np.ndarray]" = OrderedDict()
    shapes_manifest = {}
    dtypes_manifest = {}

    for k in keys:
        arr = _to_numpy_fp16(filtered[k])
        arrays[k] = arr
        shapes_manifest[k] = tuple(int(x) for x in arr.shape)
        dtypes_manifest[k] = "float16"

    manifest: Dict[str, object] = {
        "format": "safetensors",
        "dtype": "float16",
        "shapes": shapes_manifest,
        "keys": keys,
        "arknet_export_version": 1,
    }
    if extra_manifest:
        manifest["extra"] = extra_manifest

    meta = _stable_metadata(manifest)

    ensure_parent_dir(out_path)
    # Save deterministically (save_file_np respects dict insertion order)
    save_file_np(arrays, out_path, metadata=meta)

    # Read back file bytes to hash (domain-separated)
    with open(out_path, "rb") as f:
        data = f.read()
    digest = sha256_domain_hex(_EXPORT_DOMAIN, data)
    return ExportResult(
        path=out_path,
        size_bytes=len(data),
        sha256=digest,
        n_tensors=len(keys),
        keys=tuple(keys),
        meta_json=meta["arknet_meta"],
    )


def export_model_to_safetensors(
    model,
    out_path: str,
    *,
    exclude_substrings: Iterable[str] = DEFAULT_EXCLUDE_SUBSTR,
    extra_manifest: Optional[Dict[str, object]] = None,
) -> ExportResult:
    """
    Convenience wrapper: pull state_dict from a torch model and export it.
    """
    sd = _state_dict_from_model(model)
    return export_state_dict_to_safetensors(
        sd, out_path, exclude_substrings=exclude_substrings, extra_manifest=extra_manifest
    )
