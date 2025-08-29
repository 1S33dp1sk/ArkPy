# arknet_py/utils/json_canon.py
"""
Canonical JSON helpers for Arknet.

Goals:
- Deterministic text form for the same semantic object.
- No NaN/Infinity (reject them) to keep cross-runtime parity.
- Sorted keys, minimal whitespace, UTF-8 (no ASCII escaping).

Public API:
- dumps_canon(obj) -> str
- loads_canon(text) -> Any
- (compat) dumps_canonical, loads_canonical
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Mapping

__all__ = [
    "dumps_canon",
    "loads_canon",
    # compatibility aliases
    "dumps_canonical",
    "loads_canonical",
]


def _canonize(x: Any) -> Any:
    """
    Recursively normalize Python structures for canonical dumping:
    - dicts: sort keys lexicographically; values canonized
    - lists/tuples: canonize each element (order preserved)
    - bool/None/int/str: kept as-is
    - float: kept as float but NaN/Inf rejected at dump time (allow_nan=False)
    - Decimal: converted to str to avoid binary float drift
    """
    if isinstance(x, Mapping):
        # sort keys by their string form for stability
        return {str(k): _canonize(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canonize(v) for v in x]
    if isinstance(x, Decimal):
        # lossless textual form
        return format(x, "f")
    return x  # bool, None, int, float, str, etc.


def dumps_canon(obj: Any) -> str:
    """
    Canonical JSON string:
      - sorted keys
      - separators=(',', ':')  (no extra spaces)
      - ensure_ascii=False (emit UTF-8)
      - allow_nan=False (reject NaN/Inf)
    """
    canon = _canonize(obj)
    return json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def loads_canon(text: str) -> Any:
    """Parse JSON text into Python objects (no normalization)."""
    return json.loads(text)


# ----------------- compatibility aliases -----------------

def dumps_canonical(obj: Any) -> str:
    """Alias of dumps_canon()."""
    return dumps_canon(obj)


def loads_canonical(text: str) -> Any:
    """Alias of loads_canon()."""
    return loads_canon(text)
