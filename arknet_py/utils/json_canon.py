# json_canon.py — Ark Canonical JSON (ACJ v1)
# Deterministic JSON bytes for hashing & cross-process comparisons.
#
# Rules:
# - UTF-8 bytes output
# - Objects sorted by key (Unicode code point order)
# - Minimal separators ("," and ":") — no extra whitespace
# - No NaN/Infinity/-Infinity
# - By default, floats are FORBIDDEN (for cross-lang stability). You can allow
#   them with allow_floats=True; representation follows Python 3.11+ shortest
#   roundtrip, which is deterministic within Python but not guaranteed across
#   languages. Prefer integers/strings for stable hashing across toolchains.

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Tuple

__all__ = [
    "dumps",
    "dumps_str",
    "loads",
    "normalize_bytes",
    "assert_no_floats",
]

_JSON_KW = dict(
    sort_keys=True,
    ensure_ascii=False,   # keep UTF-8; escapes only control chars
    allow_nan=False,      # disallow NaN/Inf
    separators=(",", ":"),# no spaces
)


def assert_no_floats(obj: Any) -> None:
    """Raise TypeError if a float is found anywhere in the structure."""
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, float):
            raise TypeError("floats are not allowed in canonical JSON (use int/str)")
        if isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            # keys must be strings for canonical JSON
            for k, v in x.items():
                if not isinstance(k, str):
                    raise TypeError("object keys must be strings in canonical JSON")
                stack.append(v)
        elif isinstance(x, (str, int, bool)) or x is None:
            continue
        elif isinstance(x, bytes):
            raise TypeError("bytes not allowed; decode to UTF-8 str first")
        else:
            raise TypeError(f"unsupported type for canonical JSON: {type(x).__name__}")


def dumps(obj: Any, *, allow_floats: bool = False) -> bytes:
    """
    Return canonical JSON as UTF-8 BYTES.
    Use allow_floats=True only if you accept Python-specific float formatting.
    """
    if not allow_floats:
        assert_no_floats(obj)
    s = json.dumps(obj, **_JSON_KW)
    return s.encode("utf-8")


def dumps_str(obj: Any, *, allow_floats: bool = False) -> str:
    """Return canonical JSON as STR (UTF-8 text)."""
    if not allow_floats:
        assert_no_floats(obj)
    return json.dumps(obj, **_JSON_KW)


def loads(data: str | bytes) -> Any:
    """Standard JSON load (useful for normalize_bytes)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def normalize_bytes(data: str | bytes, *, allow_floats: bool = False) -> bytes:
    """
    Parse arbitrary JSON then re-emit canonical bytes.
    Handy for “canonicalizing” JSON files before hashing.
    """
    obj = loads(data)
    return dumps(obj, allow_floats=allow_floats)
