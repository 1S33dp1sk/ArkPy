# arknet_py/training/exporter.py
"""
Canonical weight export (safetensors) and weight commit helpers.

Consensus rule (defaults; overridable):
- Float tensors are bucketized BEFORE export:
  * mode="fp16_rne"  -> cast to IEEE-754 float16 (round-to-nearest-even)
  * mode="grid"      -> snap to a uniform grid with step `grid_eps` (float32)

Why: this collapses tiny GPU/parallelism-induced numeric drift so that the
exported bytes – and therefore the commit hash – are stable across hardware.

Public API:
- export_safetensors(out_path, tensors=None, *, bucket_mode=None, grid_eps=None)
    Writes a valid .safetensors file. If `tensors` is falsy, writes a
    deterministic empty archive (header-only).

- weight_commit_hex(tensors=None, *, path=None, bucket_mode=None, grid_eps=None) -> str
    Returns sha256 hex of the canonical safetensors bytes (if `tensors`
    provided) or of the file at `path` (if provided).

- tensors_to_safetensors_bytes(tensors, *, bucket_mode=None, grid_eps=None) -> bytes
    Build canonical safetensors bytes from a name->ndarray/array-like dict.

Format emitted:
  <u64 little-endian header_len> <header_json_bytes> <raw_tensor_bytes...>

Header JSON is canonicalized (sorted keys, compact) via utils.json_canon.dumps_canon.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional, Tuple

from ..utils.json_canon import dumps_canon
from ..utils.hashing import sha256_hex
from ..utils.iohelpers import atomic_write
from ..constants import (
    BUCKET_MODE_DEFAULT,
    BUCKET_MODE_FP16_RNE,
    BUCKET_MODE_GRID,
    GRID_EPS_DEFAULT,
)

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # Export of real tensors requires NumPy; empty archives still work.


__all__ = [
    "export_safetensors",
    "weight_commit_hex",
    "tensors_to_safetensors_bytes",
]


# ---------------- dtype & normalization helpers -----------------------------

def _dtype_code(arr) -> str:
    """
    Map numpy dtype -> safetensors dtype code.
    """
    if np is None:
        raise TypeError("NumPy is required to export tensors; install numpy")

    dt = np.dtype(arr.dtype)
    if dt == np.float16:
        return "F16"
    if dt == np.float32:
        return "F32"
    if dt == np.float64:
        return "F64"
    # BF16 if available
    try:
        if dt == np.dtype("bfloat16"):
            return "BF16"
    except Exception:
        pass
    if dt == np.int8:
        return "I8"
    if dt == np.int16:
        return "I16"
    if dt == np.int32:
        return "I32"
    if dt == np.int64:
        return "I64"
    if dt == np.uint8:
        return "U8"
    if dt == np.uint16:
        return "U16"
    if dt == np.uint32:
        return "U32"
    if dt == np.uint64:
        return "U64"
    if dt == np.bool_:
        return "BOOL"
    raise TypeError(f"unsupported dtype for safetensors: {dt!r}")


def _ensure_le_c_contig(a: "np.ndarray") -> "np.ndarray":
    """Ensure little-endian, C-contiguous array (copy if needed)."""
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not np.little_endian):
        a = a.byteswap().newbyteorder("<")
    return a


def _bucketize_array(
    x,
    *,
    bucket_mode: Optional[str],
    grid_eps: Optional[float],
) -> "np.ndarray":
    """
    Apply consensus bucketization to float arrays; leave non-floats unchanged.

    - bucket_mode="fp16_rne": cast to float16 (NumPy uses RNE for cast)
    - bucket_mode="grid":     round to nearest multiple of grid_eps in float32
    - bucket_mode=None:       use BUCKET_MODE_DEFAULT
    """
    if np is None:
        raise TypeError("NumPy is required to export tensors; install numpy")

    mode = bucket_mode or BUCKET_MODE_DEFAULT
    eps = GRID_EPS_DEFAULT if grid_eps is None else float(grid_eps)

    a = x if isinstance(x, np.ndarray) else np.array(x)
    kind = np.dtype(a.dtype).kind

    if kind != "f":
        # ints/uints/bool: keep as-is (but normalize layout/endianness)
        return _ensure_le_c_contig(a)

    if mode == BUCKET_MODE_FP16_RNE:
        # Cast to float16 (RNE). Keep as float16 to make bytes canonical.
        a16 = a.astype(np.float16, copy=False)
        return _ensure_le_c_contig(a16)

    if mode == BUCKET_MODE_GRID:
        if eps <= 0.0 or not np.isfinite(eps):
            raise ValueError(f"grid_eps must be positive finite, got {eps}")
        # Work in float32 for stable bytes; bankers rounding (ties to even)
        a32 = a.astype(np.float32, copy=False)
        snapped = np.rint(a32 / eps) * eps
        snapped = snapped.astype(np.float32, copy=False)
        return _ensure_le_c_contig(snapped)

    raise ValueError(f"unknown bucket_mode: {mode!r}")


# ---------------- core building blocks --------------------------------------

def _build_header_and_body(
    tensors: Dict[str, Any],
    *,
    bucket_mode: Optional[str],
    grid_eps: Optional[float],
) -> Tuple[bytes, bytes]:
    """
    Produce (<header_prefix+json>, <raw_concat_bytes>) for safetensors.
    Header prefix is 8-byte LE length as per spec. Tensor names are sorted.
    """
    if not tensors:
        hdr_json = dumps_canon({}).encode("utf-8")
        return struct.pack("<Q", len(hdr_json)) + hdr_json, b""

    names = sorted(tensors.keys())
    np_tensors: Dict[str, "np.ndarray"] = {}

    if np is None:
        raise TypeError("NumPy is required to export tensors; install numpy")

    # Normalize + bucketize per-tensor
    for name in names:
        np_tensors[name] = _bucketize_array(
            tensors[name],
            bucket_mode=bucket_mode,
            grid_eps=grid_eps,
        )

    # Build header and concatenate body bytes in order
    offset = 0
    header: Dict[str, Any] = {}
    body_parts = []
    for name in names:
        a = np_tensors[name]
        a = _ensure_le_c_contig(a)
        code = _dtype_code(a)
        nbytes = int(a.nbytes)
        header[name] = {
            "dtype": code,
            "shape": [int(d) for d in a.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        body_parts.append(a.tobytes(order="C"))
        offset += nbytes

    hdr_json = dumps_canon(header).encode("utf-8")
    hdr_prefix = struct.pack("<Q", len(hdr_json))
    body = b"".join(body_parts)
    return hdr_prefix + hdr_json, body


def tensors_to_safetensors_bytes(
    tensors: Optional[Dict[str, Any]],
    *,
    bucket_mode: Optional[str] = None,
    grid_eps: Optional[float] = None,
) -> bytes:
    """
    Return full .safetensors bytes for a dict of name->array-like.
    If tensors is falsy, returns a valid empty-archive safetensors.
    """
    hdr, body = _build_header_and_body(tensors or {}, bucket_mode=bucket_mode, grid_eps=grid_eps)
    return hdr + body


# ---------------- public API -------------------------------------------------

def export_safetensors(
    out_path: str,
    tensors: Optional[Dict[str, Any]] = None,
    *,
    bucket_mode: Optional[str] = None,
    grid_eps: Optional[float] = None,
) -> None:
    """
    Write a canonical .safetensors file.

    Args:
      out_path: destination path
      tensors:  dict of name->array-like; if falsy, writes empty archive
      bucket_mode: "fp16_rne" (default) or "grid"
      grid_eps: grid step for bucket_mode="grid" (default from constants)
    """
    data = tensors_to_safetensors_bytes(
        tensors or {},
        bucket_mode=bucket_mode,
        grid_eps=grid_eps,
    )
    atomic_write(out_path, data, binary=True)


def weight_commit_hex(
    tensors: Optional[Dict[str, Any]] = None,
    *,
    path: Optional[str] = None,
    bucket_mode: Optional[str] = None,
    grid_eps: Optional[float] = None,
) -> str:
    """
    Produce a sha256 hex for weights either from:
      - a live tensor dict (canonical, bucketized safetensors first), or
      - an existing .safetensors file (`path`).
    """
    if path is not None:
        # Hash file bytes directly (assumed already bucketized per consensus)
        from ..utils.iohelpers import read_bytes  # local import to avoid cycles
        return sha256_hex(read_bytes(path))

    data = tensors_to_safetensors_bytes(
        tensors or {},
        bucket_mode=bucket_mode,
        grid_eps=grid_eps,
    )
    return sha256_hex(data)
