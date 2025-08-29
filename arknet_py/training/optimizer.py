# training/optimizer.py — deterministic, reference optimizers + canonical state commit
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, OrderedDict as TOrderedDict
from collections import OrderedDict
import math
import re

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import numpy as np  # type: ignore

from ..utils.hashing import sha256_domain_hex
from ..utils.be import be32


# --------- shared helpers ---------------------------------------------------

_OPT_DOMAIN_ADAMW = b"ARK/OPTIMIZER/ADAMW/v1\n"
_OPT_DOMAIN_SGD   = b"ARK/OPTIMIZER/SGD/v1\n"

_NAME_RE = re.compile(r"[^ -~]")  # keep names printable; otherwise we still hash bytes

def _named_leaf_parameters(model) -> "OrderedDict[str, 'torch.Tensor']":
    """
    Deterministically collect leaf parameters from a torch module.
    Sorted by fully-qualified name (lexicographic, case-sensitive).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for optimizer reference impls")
    if not hasattr(model, "named_parameters"):
        raise TypeError("model has no .named_parameters()")

    items = list(model.named_parameters(recurse=True))
    # filter leaves only (nn.Parameter)
    items = [(n, p) for (n, p) in items if isinstance(p, torch.nn.Parameter)]
    items.sort(key=lambda kv: kv[0])
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for n, p in items:
        out[n] = p
    return out


def _to_f32_cpu(x: "torch.Tensor") -> "torch.Tensor":
    """
    Copy to CPU float32, contiguous.
    """
    return x.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _require_grads_present(name: str, p: "torch.Tensor") -> "torch.Tensor":
    if p.grad is None:
        raise RuntimeError(f"param '{name}' has no gradient (.grad is None)")
    return p.grad


def _apply_param_update_inplace(p: "torch.Tensor", delta_cpu_f32: "torch.Tensor") -> None:
    """
    Apply delta computed in CPU-f32 to param in its native dtype/device:
      p <- p + cast(delta)
    We cast the CPU delta to p.dtype on CPU, then move to p.device to minimize drift.
    """
    # NB: do not mutate grad; only data
    dev = p.device
    dtype = p.dtype
    # cast on CPU first for stable rounding, then move to target device
    delta_native = delta_cpu_f32.to(dtype=dtype, device="cpu")
    if dev.type != "cpu":
        delta_native = delta_native.to(device=dev)
    p.data.add_(delta_native)  # in-place


# --------- AdamW (decoupled) -----------------------------------------------

@dataclass
class AdamWConfig:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0     # decoupled
    amsgrad: bool = False


class AdamWDeterministicTorch:
    """
    Deterministic AdamW (decoupled weight decay, Loshchilov & Hutter).
    - State on CPU float32 (m, v, [vmax]).
    - Strict update order: sorted(parameter_name).
    - Bias correction uses step t starting at 1 on first step().
    - Update formula (decoupled decay):
        p <- p - lr * weight_decay * p - lr * m_hat / (sqrt(v_hat) + eps)
      where m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t).
    """

    def __init__(self, model, cfg: Optional[AdamWConfig] = None):
        if torch is None:
            raise RuntimeError("PyTorch required")
        self.cfg = cfg or AdamWConfig()
        self._params = _named_leaf_parameters(model)
        self._step: int = 0
        # per-parameter state: dict[name] = {"m": Tensor, "v": Tensor, "vmax": Tensor?}
        self._state: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, p in self._params.items():
            shape = tuple(p.shape)
            self._state[name] = {
                "m": torch.zeros(shape, dtype=torch.float32, device="cpu"),
                "v": torch.zeros(shape, dtype=torch.float32, device="cpu"),
            }
            if self.cfg.amsgrad:
                self._state[name]["vmax"] = torch.zeros(shape, dtype=torch.float32, device="cpu")

    @property
    def step_count(self) -> int:
        return self._step

    def zero_grad(self) -> None:
        for p in self._params.values():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self) -> None:
        self._step += 1
        beta1, beta2 = self.cfg.betas
        t = float(self._step)

        # Precompute bias corrections
        bc1 = 1.0 - (beta1 ** t)
        bc2 = 1.0 - (beta2 ** t)

        for name, p in self._params.items():
            g = _require_grads_present(name, p)
            g32 = _to_f32_cpu(g)

            st = self._state[name]
            m = st["m"]; v = st["v"]

            # moments
            m.mul_(beta1).add_(g32, alpha=(1.0 - beta1))
            v.mul_(beta2).addcmul_(g32, g32, value=(1.0 - beta2))

            # bias-correct
            m_hat = m / bc1
            v_hat = v / bc2

            if self.cfg.amsgrad:
                vmax = st["vmax"]
                torch.maximum(vmax, v_hat, out=vmax)
                denom = torch.sqrt(vmax).add_(self.cfg.eps)
            else:
                denom = torch.sqrt(v_hat).add_(self.cfg.eps)

            # decoupled weight decay term uses original p
            # compute delta in f32 on CPU
            p32 = _to_f32_cpu(p.data)
            delta = torch.empty_like(p32)

            # total delta = -(lr*wd*p) - lr * (m_hat / denom)
            if self.cfg.weight_decay != 0.0:
                delta.copy_(p32).mul_(-self.cfg.lr * self.cfg.weight_decay)
            else:
                delta.zero_()

            adam_term = m_hat / denom
            delta.add_(adam_term, alpha=-self.cfg.lr)

            _apply_param_update_inplace(p, delta)

    # ---- state export / commit -------------------------------------------

    def state_commit_hex(self) -> str:
        """
        Canonical digest over the optimizer state and hyperparams.
        This is a compact streaming hash; no temp files.
        Layout (domain-separated):
          - "cfg": lr, betas, eps, weight_decay, amsgrad (be32-encoded scalars with IEEE754)
          - step (be32)
          - for each param in sorted(name):
              - name length (be32), name bytes (utf-8)
              - m tensor shape rank (be32) + each dim (be32)
              - raw bytes of m (f32 little-endian)
              - v tensor shape … + raw bytes
              - if amsgrad: vmax shape … + raw bytes
        """
        import struct
        import hashlib

        h = hashlib.sha256()
        h.update(_OPT_DOMAIN_ADAMW)

        # cfg
        def f32(x: float) -> bytes:
            return struct.pack("<f", float(x))

        def b32(u: int) -> bytes:
            return be32(u)

        # hyper
        h.update(f32(self.cfg.lr))
        h.update(f32(self.cfg.betas[0]))
        h.update(f32(self.cfg.betas[1]))
        h.update(f32(self.cfg.eps))
        h.update(f32(self.cfg.weight_decay))
        h.update(b32(1 if self.cfg.amsgrad else 0))

        # global step
        h.update(b32(int(self._step)))

        for name in self._params.keys():
            name_b = name.encode("utf-8")
            h.update(b32(len(name_b))); h.update(name_b)

            st = self._state[name]
            for key in ("m", "v") + (("vmax",) if self.cfg.amsgrad else ()):
                t = st[key].contiguous().to(dtype=torch.float32, device="cpu").numpy()  # type: ignore
                arr: np.ndarray = t.view()  # no copy
                # shape
                h.update(b32(arr.ndim))
                for d in arr.shape:
                    h.update(b32(int(d)))
                # bytes (little-endian f32)
                h.update(arr.astype("<f4", copy=False).tobytes(order="C"))

        return h.hexdigest()


# --------- SGD (with momentum, optional Nesterov) --------------------------

@dataclass
class SGDConfig:
    lr: float = 1e-2
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0   # decoupled if decoupled_wd=True, else L2-coupled
    nesterov: bool = False
    decoupled_wd: bool = True   # default to decoupled-by-default for parity with AdamW-style


class SGDDeterministicTorch:
    """
    Deterministic SGD with (optional) momentum, nesterov, and decoupled weight decay.
    Update order = sorted(parameter_name).

    If decoupled_wd:
        p <- p - lr*wd*p - lr * grad_term
    Else (classic L2 regularization / coupled):
        g <- g + wd * p
        p <- p - lr * g   (with momentum/nesterov handled on g)

    Momentum buffer u (f32 on CPU):
        u <- momentum * u + (1 - dampening) * g
        nesterov: grad_term = g + momentum*u
        else:     grad_term = u
    """

    def __init__(self, model, cfg: Optional[SGDConfig] = None):
        if torch is None:
            raise RuntimeError("PyTorch required")
        self.cfg = cfg or SGDConfig()
        self._params = _named_leaf_parameters(model)
        self._step: int = 0
        self._bufs: Dict[str, torch.Tensor] = {}
        if self.cfg.momentum != 0.0:
            for name, p in self._params.items():
                self._bufs[name] = torch.zeros(tuple(p.shape), dtype=torch.float32, device="cpu")

    @property
    def step_count(self) -> int:
        return self._step

    def zero_grad(self) -> None:
        for p in self._params.values():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self) -> None:
        self._step += 1

        for name, p in self._params.items():
            g = _require_grads_present(name, p)
            g32 = _to_f32_cpu(g)
            p32 = _to_f32_cpu(p.data)

            if not self.cfg.decoupled_wd and self.cfg.weight_decay != 0.0:
                # coupled L2: g <- g + wd * p
                g32 = g32 + p32.mul(self.cfg.weight_decay)

            # momentum
            if self.cfg.momentum != 0.0:
                u = self._bufs[name]
                u.mul_(self.cfg.momentum).add_(g32, alpha=(1.0 - self.cfg.dampening))
                if self.cfg.nesterov:
                    grad_term = g32 + u * self.cfg.momentum
                else:
                    grad_term = u
            else:
                grad_term = g32

            # base delta
            delta = -self.cfg.lr * grad_term

            # decoupled wd (applied off-gradient)
            if self.cfg.decoupled_wd and self.cfg.weight_decay != 0.0:
                delta = delta + (-self.cfg.lr * self.cfg.weight_decay) * p32

            _apply_param_update_inplace(p, delta)

    def state_commit_hex(self) -> str:
        """
        Canonical digest over SGD state (hyper + momentum buffers).
        """
        import struct
        import hashlib

        h = hashlib.sha256()
        h.update(_OPT_DOMAIN_SGD)

        def f32(x: float) -> bytes:
            return struct.pack("<f", float(x))

        from ..utils.be import be32

        # hyper
        h.update(f32(self.cfg.lr))
        h.update(f32(self.cfg.momentum))
        h.update(f32(self.cfg.dampening))
        h.update(f32(self.cfg.weight_decay))
        h.update(be32(1 if self.cfg.nesterov else 0))
        h.update(be32(1 if self.cfg.decoupled_wd else 0))

        h.update(be32(int(self._step)))

        names = list(self._params.keys())
        for name in names:
            name_b = name.encode("utf-8")
            h.update(be32(len(name_b))); h.update(name_b)
            if self.cfg.momentum != 0.0:
                buf = self._bufs[name].contiguous().to(dtype=torch.float32, device="cpu").numpy()  # type: ignore
                arr: np.ndarray = buf
                h.update(be32(arr.ndim))
                for d in arr.shape:
                    h.update(be32(int(d)))
                h.update(arr.astype("<f4", copy=False).tobytes(order="C"))
            else:
                h.update(be32(0))  # rank=0 signals "no buffer"

        return h.hexdigest()
