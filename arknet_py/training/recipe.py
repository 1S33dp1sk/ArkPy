# training/recipe.py — LR schedules & parameter freeze rules (deterministic)
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import fnmatch
import math


# ------------------------- learning-rate schedules --------------------------

@dataclass(frozen=True)
class ConstantLR:
    lr: float

    def at(self, step: int) -> float:
        return float(self.lr)


@dataclass(frozen=True)
class LinearWarmupDecay:
    base_lr: float
    total_steps: int
    warmup_steps: int = 0
    end_lr: float = 0.0

    def at(self, step: int) -> float:
        s = max(0, int(step))
        T = max(1, int(self.total_steps))
        W = max(0, int(self.warmup_steps))

        if s < W and W > 0:
            # linear warmup from 0 -> base_lr
            return float(self.base_lr) * (s / float(W))

        # linear decay from base_lr -> end_lr over [W .. T]
        if s >= T:
            return float(self.end_lr)
        frac = (s - W) / float(max(1, T - W))
        return float(self.base_lr) * (1.0 - frac) + float(self.end_lr) * frac


@dataclass(frozen=True)
class CosineDecay:
    base_lr: float
    total_steps: int
    min_lr: float = 0.0

    def at(self, step: int) -> float:
        s = max(0, min(int(step), int(self.total_steps)))
        if self.total_steps <= 0:
            return float(self.min_lr)
        cos = 0.5 * (1.0 + math.cos(math.pi * s / float(self.total_steps)))
        return float(self.min_lr) + (float(self.base_lr) - float(self.min_lr)) * cos


@dataclass(frozen=True)
class CosineWithWarmup:
    base_lr: float
    total_steps: int
    warmup_steps: int = 0
    min_lr: float = 0.0

    def at(self, step: int) -> float:
        s = max(0, int(step))
        W = max(0, int(self.warmup_steps))
        if s < W and W > 0:
            return float(self.base_lr) * (s / float(W))
        # shift cosine to start at s = W
        T = max(1, int(self.total_steps))
        if s >= T:
            return float(self.min_lr)
        # progress in [0..1]
        prog = (s - W) / float(max(1, T - W))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return float(self.min_lr) + (float(self.base_lr) - float(self.min_lr)) * cos


def make_lr_scheduler(kind: str, **kw) -> Callable[[int], float]:
    """
    Factory → scheduler(step)->lr.  Kind in:
      - "constant" (lr)
      - "linear_warmup_decay" (base_lr, total_steps, warmup_steps=0, end_lr=0)
      - "cosine" (base_lr, total_steps, min_lr=0)
      - "cosine_warmup" (base_lr, total_steps, warmup_steps=0, min_lr=0)
    """
    k = (kind or "").lower()
    if k == "constant":
        sch = ConstantLR(**kw)
    elif k in ("linear_warmup_decay", "linear"):
        sch = LinearWarmupDecay(**kw)
    elif k == "cosine":
        sch = CosineDecay(**kw)
    elif k in ("cosine_warmup", "cosine_with_warmup"):
        sch = CosineWithWarmup(**kw)
    else:
        raise ValueError(f"unknown scheduler kind: {kind}")

    return sch.at  # function(step)->lr


# ------------------------- parameter freezing rules -------------------------

@dataclass(frozen=True)
class FreezeRule:
    """
    Freeze parameters by matching their fully-qualified names (as in model.named_parameters()).
    - pattern: UNIX glob (e.g., "encoder.*", "lm_head.*", "*layernorm*")
    - start_step: first step (inclusive) when the rule applies
    - end_step: last step (inclusive) when the rule applies (None → infinite)
    """
    pattern: str
    start_step: int = 0
    end_step: Optional[int] = None


def frozen_names_at_step(step: int, rules: Sequence[FreezeRule], param_names: Sequence[str]) -> Set[str]:
    """
    Deterministically compute the set of frozen parameter names for the given step.
    Multiple rules union together. Inclusive ranges.
    """
    s = int(step)
    out: Set[str] = set()
    for r in rules or ():
        if s < r.start_step:
            continue
        if r.end_step is not None and s > r.end_step:
            continue
        for name in param_names:
            if fnmatch.fnmatchcase(name, r.pattern):
                out.add(name)
    return out


def apply_freezing_(model, frozen_names: Set[str]) -> None:
    """
    In-place toggle requires_grad for parameters in the supplied set.
    (We keep this separate so callers can manage optimizer param-groups as needed.)
    """
    if not hasattr(model, "named_parameters"):
        raise TypeError("model has no .named_parameters()")
    for name, p in model.named_parameters():
        want = (name not in frozen_names)
        if p.requires_grad != want:
            p.requires_grad = want
