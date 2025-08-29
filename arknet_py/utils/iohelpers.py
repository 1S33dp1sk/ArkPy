# iohelpers.py — small, safe, deterministic file helpers
from __future__ import annotations

import io
import os
import tempfile
from typing import Iterable, Optional

__all__ = [
    "read_bytes",
    "read_text",
    "write_bytes_atomic",
    "write_text_atomic",
    "mkdirs",
    "file_sha256",
]

_CHUNK = 1024 * 1024  # 1 MiB


def mkdirs(path: str) -> None:
    """mkdir -p path"""
    os.makedirs(path, exist_ok=True)


def _fsync_dir(path: str) -> None:
    """Best-effort fsync on the directory that contains a recently renamed file."""
    try:
        dfd = os.open(path, os.O_DIRECTORY)
    except Exception:
        return
    try:
        os.fsync(dfd)
    finally:
        try:
            os.close(dfd)
        except Exception:
            pass


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, newline="") as f:
        return f.read()


def write_bytes_atomic(path: str, data: bytes, mode: int = 0o644) -> None:
    """
    Write bytes atomically:
      - create temp in same dir
      - fsync file
      - rename over target
      - fsync directory
    """
    d = os.path.dirname(os.path.realpath(path)) or "."
    mkdirs(d)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d)
    try:
        with os.fdopen(fd, "wb", closefd=True) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
        try:
            os.chmod(path, mode)
        except Exception:
            pass
        _fsync_dir(d)
    finally:
        # If something exploded before replace(), clean up
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def write_text_atomic(path: str, text: str, encoding: str = "utf-8", mode: int = 0o644) -> None:
    data = text.encode(encoding)
    write_bytes_atomic(path, data, mode=mode)


def file_sha256(path: str) -> bytes:
    """SHA-256 of a file (streamed)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.digest()
