"""
arknet_py.constants — Python mirror of Arknet domains & defaults.

Kept intentionally simple: module-level constants plus a tiny ENV
override helper for tunables you might want to tweak when running
Python tools in isolation (does *not* change the on-chain/C values).

These match `include/core/constants.h` where applicable.
"""

from __future__ import annotations
import os
from typing import Optional

# -------------------- sizes (mirrors C; mostly informational) ------------------

HASH32_LEN: int = 32
ADDR32_LEN: int = 32
NODEID_LEN: int = 32
MODELID_LEN: int = 32
ED25519_PUB_LEN: int = 32
ED25519_SIG_LEN: int = 64

# -------------------- header / tx (wire sizes; informational) ------------------

HDR_WIRE_LEN: int = 432  # ArkHeader on-wire size
# TX_WIRE_LEN is not strictly needed in Python, but keep for parity if desired.
# TX_WIRE_LEN: int = 200

# -------------------- timing (safe defaults matching C constants) --------------

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

SLOT_MS: int = _env_int("ARK_SLOT_MS", 800)
SLOTS_PER_EPOCH: int = _env_int("ARK_SLOTS_PER_EPOCH", 7200)
GOV_WINDOW_LEN: int = _env_int("ARK_GOV_WINDOW_LEN", 256)
TRAIN_CADENCE_SLOTS: int = _env_int("ARK_TRAIN_CADENCE_SLOTS", 7200)

# -------------------- domains (must match C for cross-lang hashing) ------------

DOM_HDR: str     = "ARK/HDR/v1"
DOM_QC: str      = "ARK/QC/v1"
DOM_MODEL: str   = "ARK/MODEL/v1"
DOM_TRAIN: str   = "ARK/TRAIN/v1"
DOM_VOTE: str    = "ARK/VOTE/v1"
DOM_TX: str      = "ARK/TX/v1"
DOM_MERKLE: str  = "ARK/MERKLE/v1"

# Python-side additions used by training/audit/export tooling:
DOM_MODEL_COMMIT: str   = "ARK/MODEL-COMMIT/v1"
DOM_TRAIN_STEP: str     = "ARK/TRAIN/STEP/v1"
DOM_TRAIN_TRANSCRIPT: str = "ARK/TRAIN/TRANSCRIPT/v1"
DOM_LEADER_SEED: str    = "ARK/LEADER/SEED/v1"
DOM_LEADER_PERM: str    = "ARK/LEADER/PERM/v1"

def domain_bytes(name: str) -> bytes:
    """
    Return domain bytes with a trailing newline (common convention we use
    in Python tools to avoid accidental concatenation collisions).
    """
    return (name + "\n").encode("utf-8")

__all__ = [
    # sizes
    "HASH32_LEN","ADDR32_LEN","NODEID_LEN","MODELID_LEN",
    "ED25519_PUB_LEN","ED25519_SIG_LEN","HDR_WIRE_LEN",
    # timing
    "SLOT_MS","SLOTS_PER_EPOCH","GOV_WINDOW_LEN","TRAIN_CADENCE_SLOTS",
    # domains
    "DOM_HDR","DOM_QC","DOM_MODEL","DOM_TRAIN","DOM_VOTE","DOM_TX","DOM_MERKLE",
    "DOM_MODEL_COMMIT","DOM_TRAIN_STEP","DOM_TRAIN_TRANSCRIPT",
    "DOM_LEADER_SEED","DOM_LEADER_PERM","domain_bytes",
]
