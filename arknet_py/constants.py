"""
arknet_py.constants — Python mirror of Arknet domains & defaults.

Kept intentionally simple: module-level constants plus tiny ENV override
helpers for tunables you might want to tweak when running Python tools
in isolation (does *not* change on-chain/C values).

These should mirror `include/core/constants.h` (domains, sizes) and add
Python-only consensus defaults used by training/export/audit tooling.
"""

from __future__ import annotations
import os

# -------------------- sizes (mirrors C; mostly informational) ------------------

HASH32_LEN: int = 32
ADDR32_LEN: int = 32
NODEID_LEN: int = 32
MODELID_LEN: int = 32
ED25519_PUB_LEN: int = 32
ED25519_SIG_LEN: int = 64

# -------------------- header / tx (wire sizes; informational) ------------------

HDR_WIRE_LEN: int = 432  # ArkHeader on-wire size
# TX_WIRE_LEN is not strictly needed in Python.

# -------------------- ENV helpers ---------------------------------------------

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

# -------------------- timing (safe defaults; mirror C where relevant) ---------

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
DOM_MODEL_COMMIT: str      = "ARK/MODEL-COMMIT/v1"
DOM_TRAIN_STEP: str        = "ARK/TRAIN/STEP/v1"
DOM_TRAIN_TRANSCRIPT: str  = "ARK/TRAIN/TRANSCRIPT/v1"
DOM_TRAIN_SPEC: str        = "ARK/TRAIN/SPEC/v1"
DOM_ENV_SNAPSHOT: str      = "ARK/ENV/SNAPSHOT/v1"
DOM_DATASET_TAR: str       = "ARK/DATASET/TAR/v1"
DOM_DATASET_JSONL: str     = "ARK/DATASET/JSONL/v1"
DOM_DATA_PERM: str         = "ARK/DATA/PERM/v1"
DOM_WEIGHT_BUCKET: str     = "ARK/WEIGHT/BUCKET/v1"

def domain_bytes(name: str) -> bytes:
    """
    Return domain bytes with a trailing newline (convention to avoid
    accidental concatenation collisions).
    """
    return (name + "\n").encode("utf-8")

# -------------------- consensus (Proof-of-Training) defaults -------------------

# Bucket modes for weight consensus
BUCKET_MODE_FP16_RNE: str = "fp16_rne"  # cast to IEEE fp16 round-to-nearest-even
BUCKET_MODE_GRID: str     = "grid"      # uniform grid quantization by GRID_EPS
BUCKET_MODES = (BUCKET_MODE_FP16_RNE, BUCKET_MODE_GRID)

# Tunables (ENV-overridable)
BUCKET_MODE_DEFAULT: str  = _env_str("ARK_BUCKET_MODE", BUCKET_MODE_FP16_RNE)
GRID_EPS_DEFAULT: float   = _env_float("ARK_GRID_EPS", 1e-6)
LOSS_DECIMALS_DEFAULT: int = _env_int("ARK_LOSS_DECIMALS", 1)
ANCHOR_STRIDE_DEFAULT: int = _env_int("ARK_ANCHOR_STRIDE", 1000)

__all__ = [
    # sizes
    "HASH32_LEN","ADDR32_LEN","NODEID_LEN","MODELID_LEN",
    "ED25519_PUB_LEN","ED25519_SIG_LEN","HDR_WIRE_LEN",
    # timing
    "SLOT_MS","SLOTS_PER_EPOCH","GOV_WINDOW_LEN","TRAIN_CADENCE_SLOTS",
    # domains
    "DOM_HDR","DOM_QC","DOM_MODEL","DOM_TRAIN","DOM_VOTE","DOM_TX","DOM_MERKLE",
    "DOM_MODEL_COMMIT","DOM_TRAIN_STEP","DOM_TRAIN_TRANSCRIPT","DOM_TRAIN_SPEC",
    "DOM_ENV_SNAPSHOT","DOM_DATASET_TAR","DOM_DATASET_JSONL","DOM_DATA_PERM",
    "DOM_WEIGHT_BUCKET","domain_bytes",
    # consensus defaults
    "BUCKET_MODE_FP16_RNE","BUCKET_MODE_GRID","BUCKET_MODES",
    "BUCKET_MODE_DEFAULT","GRID_EPS_DEFAULT","LOSS_DECIMALS_DEFAULT","ANCHOR_STRIDE_DEFAULT",
]
