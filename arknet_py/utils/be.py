# be.py — tiny big-endian helpers (C parity)
from __future__ import annotations

from typing import Union

__all__ = [
    "be32", "be64", "u32", "u64",
    "put_be32", "put_be64", "get_be32", "get_be64",
]

BytesLike = Union[bytes, bytearray, memoryview]


def be32(v: int) -> bytes:
    """Encode unsigned 32-bit int to 4 big-endian bytes."""
    if not (0 <= v <= 0xFFFFFFFF):
        raise ValueError("be32 out of range (0..2^32-1)")
    return v.to_bytes(4, "big", signed=False)


def be64(v: int) -> bytes:
    """Encode unsigned 64-bit int to 8 big-endian bytes."""
    if not (0 <= v <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError("be64 out of range (0..2^64-1)")
    return v.to_bytes(8, "big", signed=False)


def u32(b: BytesLike) -> int:
    """Decode 4 big-endian bytes to unsigned 32-bit int."""
    bv = bytes(b)
    if len(bv) != 4:
        raise ValueError("u32 expects exactly 4 bytes")
    return int.from_bytes(bv, "big", signed=False)


def u64(b: BytesLike) -> int:
    """Decode 8 big-endian bytes to unsigned 64-bit int."""
    bv = bytes(b)
    if len(bv) != 8:
        raise ValueError("u64 expects exactly 8 bytes")
    return int.from_bytes(bv, "big", signed=False)


def put_be32(buf: bytearray, off: int, v: int) -> None:
    """Write be32 at offset into a mutable buffer."""
    b = be32(v)
    end = off + 4
    if end > len(buf) or off < 0:
        raise IndexError("put_be32 out of bounds")
    buf[off:end] = b


def put_be64(buf: bytearray, off: int, v: int) -> None:
    """Write be64 at offset into a mutable buffer."""
    b = be64(v)
    end = off + 8
    if end > len(buf) or off < 0:
        raise IndexError("put_be64 out of bounds")
    buf[off:end] = b


def get_be32(buf: BytesLike, off: int) -> int:
    """Read be32 at offset from a bytes-like."""
    mv = memoryview(buf)
    end = off + 4
    if end > len(mv) or off < 0:
        raise IndexError("get_be32 out of bounds")
    return u32(mv[off:end])


def get_be64(buf: BytesLike, off: int) -> int:
    """Read be64 at offset from a bytes-like."""
    mv = memoryview(buf)
    end = off + 8
    if end > len(mv) or off < 0:
        raise IndexError("get_be64 out of bounds")
    return u64(mv[off:end])
