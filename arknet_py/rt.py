# arknet_py/rt.py
from __future__ import annotations
from typing import Optional, Union

_ARTIFACT: Optional[str] = None

def init(artifact_dir: Optional[str]) -> None:
    global _ARTIFACT
    _ARTIFACT = artifact_dir

def train_step(state: Union[bytes, bytearray], batch: Union[bytes, bytearray]) -> bytes:
    if not isinstance(state, (bytes, bytearray)) or not isinstance(batch, (bytes, bytearray)):
        raise TypeError("train_step expects bytes")
    # wiring stub: echo state unchanged (useful for server e2e tests)
    return bytes(state)
