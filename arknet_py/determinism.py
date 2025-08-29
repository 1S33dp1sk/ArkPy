"""
arknet_py.determinism — environment/profile knobs for reproducible runs.

Goals:
- Provide one call to set a *sane* deterministic baseline across common stacks.
- Keep it tolerant: if a library (NumPy, PyTorch, TensorFlow) isn't present,
  we just skip its section.
- Offer a small report() helper for provenance/audits.

Notes:
- Some frameworks read env flags at import time; call apply_determinism_profile()
  as early as possible in your process.
- Strict deterministic algorithms can reduce performance and may raise if an op
  lacks a deterministic implementation. Use strict_ops=False to soften this.
"""

from __future__ import annotations

import os
import platform
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


# --------------------------- internals --------------------------------------


def _setenv_default(name: str, value: Any) -> None:
    """Set env var only if it's not already set."""
    os.environ.setdefault(name, str(value))


def _setenv_force(name: str, value: Any) -> None:
    """Force-set env var."""
    os.environ[name] = str(value)


# --------------------------- public API -------------------------------------


def apply_determinism_profile(
    seed: Optional[int] = 0,
    *,
    allow_tf32: bool = False,
    strict_ops: bool = True,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Make common frameworks run deterministically *as far as practically possible*.

    seed:
      If not None, seeds Python, NumPy, and PyTorch RNGs (best effort).
    allow_tf32:
      If False (default), attempts to disable TF32 math for cross-device parity.
    strict_ops:
      If True (default), requests deterministic algorithms (may be slower/stricter).
    env_overrides:
      Optional dict of env→value to apply after defaults (e.g., {"OMP_NUM_THREADS": 1}).
    """
    # --- process & BLAS/GPU env gates ---
    if seed is not None:
        _setenv_default("PYTHONHASHSEED", seed)
    # Deterministic GEMM workspace for cuBLAS
    _setenv_default("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # TensorFlow deterministic ops hint (respected by tf+addons on many ops)
    _setenv_default("TF_DETERMINISTIC_OPS", "1")
    # CuDNN autotune off → more stable kernels
    _setenv_default("TF_CUDNN_USE_AUTOTUNE", "0")

    if not allow_tf32:
        # NVIDIA global TF32 override (checked by some libraries)
        _setenv_default("NVIDIA_TF32_OVERRIDE", "0")

    # Apply any user-provided environment last
    if env_overrides:
        for k, v in env_overrides.items():
            _setenv_force(k, v)

    # --- Python stdlib RNG ---
    if seed is not None:
        random.seed(int(seed))

    # --- NumPy ---
    try:
        import numpy as np  # type: ignore
        if seed is not None:
            np.random.seed(int(seed))
    except Exception:
        pass

    # --- PyTorch ---
    try:
        import torch  # type: ignore

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # CuDNN determinism toggles
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        if not allow_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

        # Enforce deterministic algorithms when available
        if strict_ops:
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                # Older torch versions: fallback to warn_only behavior
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
    except Exception:
        pass

    # --- TensorFlow (optional) ---
    try:
        import tensorflow as tf  # type: ignore

        if seed is not None:
            try:
                tf.random.set_seed(int(seed))
            except Exception:
                pass

        if not allow_tf32:
            try:
                # Disable mixed/TF32-like fast math where possible
                from tensorflow.python.framework import config as tf_cfg  # type: ignore
                tf_cfg.enable_op_determinism()  # TF 2.13+
            except Exception:
                # Older TF: rely on TF_DETERMINISTIC_OPS env var
                pass
    except Exception:
        pass


def determinism_report() -> Dict[str, Any]:
    """
    Capture a small provenance blob: versions, CUDA status, and key envs.
    """
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_USE_AUTOTUNE",
            ]
        },
        "libs": {},
    }

    # NumPy
    try:
        import numpy as np  # type: ignore
        info["libs"]["numpy"] = {"version": np.__version__}
    except Exception:
        info["libs"]["numpy"] = None

    # PyTorch
    try:
        import torch  # type: ignore
        cuda = {
            "available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
            "device_count": int(getattr(torch.cuda, "device_count", lambda: 0)() or 0),
            "version_cuda": getattr(torch.version, "cuda", None),
            "version_cudnn": getattr(torch.backends, "cudnn", None)
            and getattr(torch.backends.cudnn, "version", lambda: None)(),
        }
        info["libs"]["torch"] = {"version": torch.__version__, "cuda": cuda}
    except Exception:
        info["libs"]["torch"] = None

    # TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        info["libs"]["tensorflow"] = {"version": tf.__version__}
    except Exception:
        info["libs"]["tensorflow"] = None

    return info


@contextmanager
def deterministic_session(
    seed: Optional[int] = 0,
    *,
    allow_tf32: bool = False,
    strict_ops: bool = True,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """
    Context helper that applies the profile once before the block.
    (We do not attempt to restore env afterwards—most frameworks sample flags at import.)
    """
    apply_determinism_profile(seed, allow_tf32=allow_tf32, strict_ops=strict_ops, env_overrides=env_overrides)
    yield
