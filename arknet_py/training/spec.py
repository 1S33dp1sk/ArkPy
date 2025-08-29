# arknet_py/training/spec.py — TrainingSpec schema + canonical hashing (spec_hash)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.json_canon import dumps_canon          # deterministic JSON string
from ..utils.hashing import domain_hash_hex, sha256_domain_hex         # domain-separated SHA-256

# Domain tag aligned with C constants style (ARK_DOM_TRAIN)
_SPEC_DOMAIN = b"ARK/TRAIN/SPEC/v1\n"

# (Optional) reference shapes for docs/tooling — not used directly by trainer
@dataclass
class DatasetSpecDoc:
    name: str
    path: str
    format: str = "auto"
    weight: float = 1.0
    commit: Optional[str] = None
    split: Optional[Dict[str, float]] = None


@dataclass
class OptimizerSpecDoc:
    name: str = "adamw"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class SchedulerSpecDoc:
    name: str = "none"  # "cosine", "linear", "none"
    warmup_steps: int = 0
    total_steps: Optional[int] = None


@dataclass
class TrainingSpec:
    """
    Canonical training job description (kept deliberately lean).

    Flexible enough to support:
      (A) richer schema (epochs/max_steps, datasets list, scheduler),
      (B) current trainer expectations (steps, schedule dict, singular dataset).

    Normalization performed by from_dict():
      - steps -> max_steps
      - scheduler -> schedule
      - dataset (dict) and/or datasets (list of dicts)
      - optimizer/schedule coerced to dicts
      - round_dp: integer decimals for rounding float commits (default=1)
    """
    # Versioning / metadata
    format_version: int = 1
    model_base_commit: Optional[str] = None
    recipe: str = "default"
    recipe_version: int = 1

    # Determinism
    seed: int = 0
    allow_tf32: bool = False
    round_dp: int = 1  # NEW: decimals to round floats for weight commit & loss

    # Loop shape
    epochs: int = 1
    max_steps: Optional[int] = None  # preferred storage; .steps property provides compat

    # Batching / seq
    batch_size: int = 32
    grad_accum: int = 1
    max_seq_len: Optional[int] = None

    # Optim/schedule (trainer expects dicts)
    optimizer: Dict[str, Any] = field(default_factory=lambda: {"name": "adamw", "lr": 1e-3})
    schedule: Dict[str, Any]  = field(default_factory=dict)

    # Data (compat: both singular and list)
    dataset: Dict[str, Any] = field(default_factory=dict)       # primary dataset (trainer uses this)
    datasets: List[Dict[str, Any]] = field(default_factory=list)

    # Misc
    notes: Optional[str] = None

    # ------------------------- constructors / normalization -----------------

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "TrainingSpec":
        if not isinstance(src, dict):
            raise ValueError("TrainingSpec.from_dict: expected dict")

        d = dict(src)  # shallow copy

        # steps -> max_steps
        if "steps" in d and d.get("max_steps") is None:
            try:
                d["max_steps"] = int(d["steps"])
            except Exception:
                d["max_steps"] = None

        # scheduler -> schedule
        if "scheduler" in d and "schedule" not in d:
            sch = d.get("scheduler")
            d["schedule"] = sch if isinstance(sch, dict) else {}

        # helpers for ints
        def _as_int(k: str, default: int) -> int:
            try:
                return int(d.get(k, default) or default)
            except Exception:
                return default

        format_version = _as_int("format_version", 1)
        recipe_version = _as_int("recipe_version", 1)
        seed          = _as_int("seed", 0)
        epochs        = _as_int("epochs", 1)
        batch_size    = _as_int("batch_size", 32)
        grad_accum    = _as_int("grad_accum", 1)
        round_dp      = _as_int("round_dp", 1)

        max_steps = d.get("max_steps")
        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except Exception:
                max_steps = None

        max_seq_len = d.get("max_seq_len")
        if max_seq_len is not None:
            try:
                max_seq_len = int(max_seq_len)
            except Exception:
                max_seq_len = None

        # optimizer/schedule -> dicts
        opt = d.get("optimizer")
        if not isinstance(opt, dict):
            try:
                opt = dict(opt)  # type: ignore[arg-type]
            except Exception:
                opt = {"name": "adamw", "lr": 1e-3}
        sch = d.get("schedule")
        if not isinstance(sch, dict):
            try:
                sch = dict(sch)  # type: ignore[arg-type]
            except Exception:
                sch = {}

        # dataset(s) normalization
        ds_single = d.get("dataset")
        if ds_single is None:
            ds_single = {}
        elif isinstance(ds_single, str):
            ds_single = {"path": ds_single}
        elif not isinstance(ds_single, dict):
            try:
                ds_single = dict(ds_single)  # type: ignore[arg-type]
            except Exception:
                ds_single = {}

        ds_list_in = d.get("datasets")
        ds_list: List[Dict[str, Any]] = []
        if isinstance(ds_list_in, list):
            for item in ds_list_in:
                if isinstance(item, dict):
                    ds_list.append(item)
                elif isinstance(item, str):
                    ds_list.append({"path": item})
                else:
                    try:
                        ds_list.append(dict(item))  # type: ignore[arg-type]
                    except Exception:
                        pass

        # If no primary dataset but list exists, pick first for MVP compat
        if not ds_single and ds_list:
            ds_single = dict(ds_list[0])

        spec = TrainingSpec(
            format_version=format_version,
            model_base_commit=d.get("model_base_commit"),
            recipe=str(d.get("recipe", "default")),
            recipe_version=recipe_version,
            seed=seed,
            allow_tf32=bool(d.get("allow_tf32", False)),
            round_dp=round_dp,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_seq_len=max_seq_len,
            optimizer=opt,
            schedule=sch,
            dataset=ds_single,
            datasets=ds_list,
            notes=d.get("notes"),
        )
        spec.validate()
        return spec

    # ------------------------- computed / helpers ---------------------------

    @property
    def steps(self) -> int:
        """Compatibility for trainer.py (prefers max_steps, else epochs)."""
        return int(self.max_steps) if self.max_steps is not None else int(self.epochs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Stable, explicit dictionary for serialization/hashing.
        (Keeps empty dicts/lists and None fields to remain fully explicit.)
        """
        return {
            "format_version": self.format_version,
            "model_base_commit": self.model_base_commit,
            "recipe": self.recipe,
            "recipe_version": self.recipe_version,
            "seed": self.seed,
            "allow_tf32": self.allow_tf32,
            "round_dp": self.round_dp,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_len": self.max_seq_len,
            "optimizer": dict(self.optimizer),
            "schedule": dict(self.schedule),
            "dataset": dict(self.dataset),
            "datasets": list(self.datasets),
            "notes": self.notes,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """Alias used by trainer/manifests; same as to_dict for now."""
        return self.to_dict()

    def validate(self) -> None:
        if self.epochs <= 0 and not self.max_steps:
            raise ValueError("epochs must be > 0 or max_steps specified")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be > 0")
        # Optimizer sanity (lr >= 0 to allow no-op)
        try:
            lr = float(self.optimizer.get("lr", 1e-3))
        except Exception:
            lr = 1e-3
        if lr < 0:
            raise ValueError("optimizer.lr must be >= 0")
        if not (self.dataset or self.datasets):
            raise ValueError("at least one dataset required")
        if self.round_dp < 0:
            raise ValueError("round_dp must be >= 0")

    def spec_hash_hex(self) -> str:
        """
        Canonical spec hash (hex-64) with domain separation.
        """
        self.validate()
        blob = dumps_canon(self.to_public_dict()).encode("utf-8")
        # IMPORTANT: domain first, then bytes
        return sha256_domain_hex(b"ARK/TRAIN/SPEC/v1\n", blob)


__all__ = ["TrainingSpec"]
