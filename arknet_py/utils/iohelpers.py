# arknet_py/utils/iohelpers.py — small, safe, deterministic file helpers
from __future__ import annotations

import os
import tempfile
from typing import Optional, Union

__all__ = [
    "read_bytes",
    "read_text",
    "write_bytes_atomic",
    "write_text_atomic",
    "atomic_write",
    "mkdirs",
    "ensure_dir",
    "ensure_parent_dir",
    "file_sha256",
]

_CHUNK = 1024 * 1024  # 1 MiB
BytesLike = Union[bytes, bytearray, memoryview]


def mkdirs(path: str) -> None:
    """mkdir -p path"""
    os.makedirs(path, exist_ok=True)


# Alias used throughout the codebase
def ensure_dir(path: str) -> None:
    """Alias for mkdirs()."""
    mkdirs(path)


def ensure_parent_dir(path: str) -> str:
    """
    Ensure the parent directory of a file path exists.
    Returns the resolved parent directory path.

    Examples:
      ensure_parent_dir("/tmp/foo/bar.txt")  -> ensures "/tmp/foo"
      ensure_parent_dir("rel/dir/file.bin")  -> ensures "<cwd>/rel/dir"
    """
    parent = os.path.dirname(os.path.realpath(path)) or "."
    mkdirs(parent)
    return parent


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


def write_bytes_atomic(path: str, data: BytesLike, mode: int = 0o644) -> None:
    """
    Write bytes atomically:
      - create temp in same dir
      - fsync file
      - rename over target
      - fsync directory
    """
    d = ensure_parent_dir(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d)
    try:
        with os.fdopen(fd, "wb", closefd=True) as f:
            f.write(bytes(data))
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
    write_bytes_atomic(path, text.encode(encoding), mode=mode)


def atomic_write(
    path: str,
    data: Union[str, BytesLike],
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
    binary: Optional[bool] = None,  # compatibility kwarg (ignored for bytes)
) -> None:
    """
    Convenience wrapper that dispatches to bytes/text atomic writers.
    Accepts either bytes-like or str.

    - If `data` is bytes-like, `binary` is ignored.
    - If `data` is str and `binary=True`, we raise (to prevent accidental text→bytes misuse).
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        write_bytes_atomic(path, data, mode=mode)
    else:
        if binary:
            raise ValueError("atomic_write: got text with binary=True (pass bytes instead)")
        write_text_atomic(path, str(data), encoding=encoding, mode=mode)


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
