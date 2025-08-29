# arknet_py/training/container.py — environment snapshot + deterministic digest
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:    # optional deps
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..utils.json_canon import dumps_canon
from ..utils.hashing import sha256_domain_hex
from ..utils.iohelpers import atomic_write, ensure_dir

__all__ = ["write_environment_snapshot", "snapshot_environment"]

_ENV_DOMAIN = b"ARK/ENV/SNAPSHOT/v1\n"

# Determinism-relevant env keys we record verbatim (if set)
_DETERMINISM_ENV_KEYS = [
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
]

@dataclass
class GpuInfo:
    name: str
    capability: Optional[str]
    total_memory_mb: Optional[int]

@dataclass
class TorchInfo:
    version: Optional[str]
    cuda_version: Optional[str]
    cudnn: Optional[Dict[str, Any]]
    gpus: List[GpuInfo]

@dataclass
class EnvSnapshot:
    python: str
    platform: Dict[str, str]               # {system, release, machine}
    numpy: Optional[str]
    torch: Optional[TorchInfo]
    determinism_env: Dict[str, str]
    allow_tf32: bool

    def to_canon_json(self) -> str:
        # dataclasses → dict → canonical JSON (sorted keys, minimal whitespace)
        return dumps_canon(asdict(self))

    def commit_hex(self) -> str:
        text = self.to_canon_json()
        # Domain-separated digest; keep arg order consistent with repo usage
        return sha256_domain_hex(_ENV_DOMAIN, text.encode("utf-8"))

# -------- collectors (stable only; no paths/hostnames/timestamps) ----------

def _collect_platform() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }

def _collect_torch(allow_tf32: bool) -> Optional[TorchInfo]:
    if torch is None:
        return None

    version = getattr(torch, "__version__", None)
    cuda_ver = getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None

    cudnn_info: Optional[Dict[str, Any]] = None
    try:
        if hasattr(torch.backends, "cudnn"):
            cudnn_info = {
                "enabled": bool(torch.backends.cudnn.enabled),
                "version": getattr(torch.backends.cudnn, "version", None),
                "deterministic": getattr(torch.backends.cudnn, "deterministic", None),
                "benchmark": getattr(torch.backends.cudnn, "benchmark", None),
            }
    except Exception:
        cudnn_info = None

    gpus: List[GpuInfo] = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    name = str(torch.cuda.get_device_name(i))
                except Exception:
                    name = f"cuda:{i}"
                capability: Optional[str] = None
                try:
                    maj, minr = torch.cuda.get_device_capability(i)
                    capability = f"{maj}.{minr}"
                except Exception:
                    pass
                mem_mb: Optional[int] = None
                try:
                    props = torch.cuda.get_device_properties(i)
                    mem_mb = int(getattr(props, "total_memory", 0) // (1024 * 1024))
                except Exception:
                    pass
                gpus.append(GpuInfo(name=name, capability=capability, total_memory_mb=mem_mb))
    except Exception:
        # leave gpus empty
        pass

    return TorchInfo(
        version=version,
        cuda_version=cuda_ver,
        cudnn=cudnn_info,
        gpus=sorted(gpus, key=lambda x: (x.name or "", x.capability or "", x.total_memory_mb or -1)),
    )

def _collect_determinism_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in _DETERMINISM_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            out[k] = str(v)
    return out

# -------- public API -------------------------------------------------------

def snapshot_environment(*, allow_tf32: bool = False) -> EnvSnapshot:
    return EnvSnapshot(
        python=".".join(map(str, sys.version_info[:3])),
        platform=_collect_platform(),
        numpy=(getattr(np, "__version__", None) if np is not None else None),
        torch=_collect_torch(allow_tf32),
        determinism_env=_collect_determinism_env(),
        allow_tf32=bool(allow_tf32),
    )

def write_environment_snapshot(out_dir: str, *, allow_tf32: bool = False) -> str:
    """
    Write env.json (canonical JSON) into `out_dir` and return its commit hex.
    This snapshot intentionally excludes volatile fields (paths, cwd, hostnames, time).
    """
    ensure_dir(out_dir)
    snap = snapshot_environment(allow_tf32=allow_tf32)
    text = snap.to_canon_json()
    atomic_write(os.path.join(out_dir, "env.json"), text.encode("utf-8"))
    return snap.commit_hex()
