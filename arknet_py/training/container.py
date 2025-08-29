# training/container.py — environment snapshot + deterministic digest
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..utils.json_canon import dumps_canon
from ..utils.hashing import sha256_domain_hex


_ENV_DOMAIN = b"ARK/ENV/SNAPSHOT/v1\n"

# Determinism-relevant env keys we record verbatim (if set)
DETERMINISM_ENV_KEYS = [
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TF32",  # generic flag some users set
]

# Minimal pinned-package list (add more if you like)
PINNED_PKGS = ["torch", "numpy", "safetensors"]


@dataclass
class GpuInfo:
    index: int
    name: str
    capability: Optional[str]
    total_memory_mb: Optional[int]


@dataclass
class TorchInfo:
    version: Optional[str]
    cuda: Optional[str]
    cudnn: Optional[str]
    gpu_list: List[GpuInfo]


@dataclass
class EnvSnapshot:
    python: str
    python_exe: str
    platform: str
    uname: str
    pip_freeze: Dict[str, str]
    torch: TorchInfo
    determinism_env: Dict[str, str]
    nvidia_smi: Optional[str]  # first line parse or None

    def to_json(self) -> str:
        # Ensure dataclasses nested conversion
        d = asdict(self)
        return dumps_canon(d)

    def commit_hex(self) -> str:
        return sha256_domain_hex(_ENV_DOMAIN, self.to_json().encode("utf-8"))


# ---- collectors -----------------------------------------------------------

def _collect_pip_freeze(target: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in target:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", None)
            if ver:
                out[name] = str(ver)
        except Exception:
            pass
    return out


def _collect_torch() -> TorchInfo:
    gpu_list: List[GpuInfo] = []
    version = getattr(torch, "__version__", None) if torch else None
    cuda = torch.version.cuda if (torch and hasattr(torch, "version") and hasattr(torch.version, "cuda")) else None
    cudnn = None
    if torch and hasattr(torch.backends, "cudnn"):
        try:
            cudnn = str(torch.backends.cudnn.version())  # type: ignore
        except Exception:
            cudnn = None

    if torch and torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = None
                try:
                    major, minor = torch.cuda.get_device_capability(i)
                    cap = f"{major}.{minor}"
                except Exception:
                    pass
                mem_mb = None
                try:
                    props = torch.cuda.get_device_properties(i)
                    mem_mb = int(getattr(props, "total_memory", 0) // (1024 * 1024))
                except Exception:
                    pass
                gpu_list.append(GpuInfo(index=i, name=name, capability=cap, total_memory_mb=mem_mb))
        except Exception:
            pass

    return TorchInfo(version=version, cuda=cuda, cudnn=cudnn, gpu_list=gpu_list)


def _collect_nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, "-q"], stderr=subprocess.STDOUT, timeout=2)
        # Return just the first non-empty line for stability
        for line in out.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                return line[:200]
        return None
    except Exception:
        return None


def snapshot_environment() -> EnvSnapshot:
    determinism_env: Dict[str, str] = {}
    for k in DETERMINISM_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            determinism_env[k] = str(v)

    snap = EnvSnapshot(
        python=sys.version.split()[0],
        python_exe=sys.executable,
        platform=platform.platform(),
        uname=" ".join(platform.uname()),
        pip_freeze=_collect_pip_freeze(PINNED_PKGS),
        torch=_collect_torch(),
        determinism_env=determinism_env,
        nvidia_smi=_collect_nvidia_smi(),
    )
    return snap


# ---- convenience ----------------------------------------------------------

def write_environment_snapshot(out_dir: str) -> str:
    """
    Write env.json to `out_dir` (created if needed). Return commit hex.
    """
    os.makedirs(out_dir, exist_ok=True)
    snap = snapshot_environment()
    j = snap.to_json()
    with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as f:
        f.write(j)
        f.write("\n")
    return snap.commit_hex()
