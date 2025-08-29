# arknet_py/utils/hashing.py — SHA-256 helpers, domain-tagged hashes, hex utils
from __future__ import annotations

import hashlib
from typing import Iterable, Union

__all__ = [
    "sha256", "sha256_hex", "sha256_stream", "sha256_multi",
    "domain_hash", "domain_hash_hex",
    # Compatibility wrappers expected by other modules:
    "sha256_domain", "sha256_domain_hex",
    "hex_to_bytes", "bytes_to_hex",
]

BytesLike = Union[bytes, bytearray, memoryview]


def sha256(data: BytesLike) -> bytes:
    """Return raw 32-byte SHA-256 digest of data."""
    h = hashlib.sha256()
    h.update(bytes(data))
    return h.digest()


def sha256_hex(data: BytesLike) -> str:
    """Return lower-case hex64 SHA-256 of data."""
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_stream(chunks: Iterable[BytesLike]) -> bytes:
    """SHA-256 over an iterator of chunks."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(bytes(c))
    return h.digest()


def sha256_multi(*parts: BytesLike) -> bytes:
    """SHA-256 over multiple parts (in-order)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(bytes(p))
    return h.digest()


def _normalize_domain(dom: Union[str, bytes]) -> bytes:
    """
    Domain separation helper.
    Convention: ASCII string tag with trailing newline, e.g. "ARK/MODEL-COMMIT/v1\n".
    If dom is bytes and already ends with b'\\n', it's used verbatim.
    """
    if isinstance(dom, str):
        db = dom.encode("ascii")
    else:
        db = bytes(dom)
    return db if db.endswith(b"\n") else (db + b"\n")


def domain_hash(data: BytesLike, domain: Union[str, bytes]) -> bytes:
    """
    Domain-tagged SHA-256: H( domain||data ), where domain ends with '\\n'.
    Keep this aligned cross-language.
    """
    tag = _normalize_domain(domain)
    return sha256_multi(tag, bytes(data))


def domain_hash_hex(data: BytesLike, domain: Union[str, bytes]) -> str:
    """Hex64 of domain_hash()."""
    return domain_hash(data, domain).hex()


# --------- Compatibility wrappers (used by other modules) ------------------

def sha256_domain(data: BytesLike, domain: Union[str, bytes]) -> bytes:
    """Alias of domain_hash(data, domain)."""
    return domain_hash(data, domain)


def sha256_domain_hex(domain: Union[str, bytes], data: BytesLike, raw: bool = False):
    """
    Compatibility signature: (domain, data, raw=False)
    - If raw=False (default): return hex string (len 64)
    - If raw=True: return raw 32-byte digest
    """
    d = domain_hash(data, domain)
    return d if raw else d.hex()


# --------- Hex helpers -----------------------------------------------------

def hex_to_bytes(hx: str) -> bytes:
    """Lenient lower/upper hex → bytes; raises ValueError on bad input."""
    s = hx.strip()
    if s.startswith(("0x", "0X")):
        s = s[2:]
    try:
        return bytes.fromhex(s)
    except ValueError as e:
        raise ValueError(f"invalid hex: {e}") from e


def bytes_to_hex(b: BytesLike) -> str:
    """Lower-case hex (no 0x) for bytes-like."""
    return bytes(b).hex()
