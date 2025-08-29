# training/spec.py — TrainingSpec schema + canonical hashing (spec_hash)
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ..utils.json_canon import dumps_canonical  # deterministic JSON bytes
from ..utils.hashing import sha256_domain_hex   # domain-separated SHA-256


# Domain tag aligned with C constants style (ARK_DOM_TRAIN)
_SPEC_DOMAIN = b"ARK/TRAIN/SPEC/v1\n"


@dataclass
class DatasetSpec:
    """
    A single dataset use within a training job.
    - name: logical name (e.g. "alpaca", "my-jsonl")
    - path: local/remote path (runner/trainer interprets)
    - format: "jsonl"|"hf"|"parquet"|"dir"|"auto"
    - weight: sampling weight when interleaving multiple datasets
    - commit: optional precomputed dataset commit hex (if known)
    - split: optional split ratios (e.g. {"train":0.98,"val":0.02})
    """
    name: str
    path: str
    format: str = "auto"
    weight: float = 1.0
    commit: Optional[str] = None
    split: Optional[Dict[str, float]] = None


@dataclass
class OptimizerSpec:
    name: str = "adamw"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class SchedulerSpec:
    name: str = "none"  # "cosine", "linear", "none"
    warmup_steps: int = 0
    total_steps: Optional[int] = None


@dataclass
class TrainingSpec:
    """
    Canonical training job description (kept deliberately lean).
    Hashing is over the canonical JSON of this structure (with stable key order).
    """
    format_version: int = 1

    # Model base / recipe
    model_base_commit: Optional[str] = None   # starting weights commit (hex)
    recipe: str = "default"                   # identifies trainer recipe variant
    recipe_version: int = 1

    # Determinism
    seed: int = 0

    # Loop
    epochs: int = 1
    max_steps: Optional[int] = None

    # Batching/seq
    batch_size: int = 32
    grad_accum: int = 1
    max_seq_len: Optional[int] = None

    # Optim/sched
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)

    # Data
    datasets: List[DatasetSpec] = field(default_factory=list)

    # Misc
    notes: Optional[str] = None

    # ---- helpers ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python dict (dataclasses → dict), without derived/transient fields."""
        d = asdict(self)
        return d

    def validate(self) -> None:
        """
        Minimal sanity checks (no heavy jsonschema dep).
        Raises ValueError with a readable message.
        """
        if self.epochs <= 0 and not self.max_steps:
            raise ValueError("epochs must be >0 or max_steps specified")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be >0")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be >0")
        if self.optimizer.lr <= 0:
            raise ValueError("optimizer.lr must be >0")
        if not self.datasets:
            raise ValueError("at least one dataset required")
        for ds in self.datasets:
            if not ds.name or not ds.path:
                raise ValueError("dataset.name and dataset.path are required")
            if ds.split:
                s = sum(float(v) for v in ds.split.values())
                if not (0.99 <= s <= 1.01):
                    raise ValueError(f"dataset.split must sum to ~1.0 (got {s})")

    def spec_hash(self) -> str:
        """
        Canonical spec hash (hex-64). Domain-separated to avoid collisions.
        """
        self.validate()
        blob = dumps_canonical(self.to_dict())  # bytes
        return sha256_domain_hex(_SPEC_DOMAIN, blob)


# Convenience constructors ----------------------------------------------------

def from_dict(obj: Dict[str, Any]) -> TrainingSpec:
    """
    Lenient loader from arbitrary dict (e.g., JSON). Unknown keys are ignored.
    """
    # Shallow copy first
    x = dict(obj or {})

    # Nested specs
    opt = x.get("optimizer", {}) or {}
    sch = x.get("scheduler", {}) or {}

    ds_raw = x.get("datasets", []) or []
    ds_list = []
    for d in ds_raw:
        ds_list.append(DatasetSpec(
            name=d.get("name", ""),
            path=d.get("path", ""),
            format=d.get("format", "auto"),
            weight=float(d.get("weight", 1.0)),
            commit=d.get("commit"),
            split=d.get("split"),
        ))

    spec = TrainingSpec(
        format_version=int(x.get("format_version", 1)),
        model_base_commit=x.get("model_base_commit"),
        recipe=str(x.get("recipe", "default")),
        recipe_version=int(x.get("recipe_version", 1)),
        seed=int(x.get("seed", 0)),
        epochs=int(x.get("epochs", 1)),
        max_steps=(int(x["max_steps"]) if x.get("max_steps") is not None else None),
        batch_size=int(x.get("batch_size", 32)),
        grad_accum=int(x.get("grad_accum", 1)),
        max_seq_len=(int(x["max_seq_len"]) if x.get("max_seq_len") is not None else None),
        optimizer=OptimizerSpec(
            name=str(opt.get("name", "adamw")),
            lr=float(opt.get("lr", 1e-4)),
            betas=tuple(opt.get("betas", (0.9, 0.999))),
            eps=float(opt.get("eps", 1e-8)),
            weight_decay=float(opt.get("weight_decay", 0.0)),
        ),
        scheduler=SchedulerSpec(
            name=str(sch.get("name", "none")),
            warmup_steps=int(sch.get("warmup_steps", 0)),
            total_steps=(int(sch["total_steps"]) if sch.get("total_steps") is not None else None),
        ),
        datasets=ds_list,
        notes=(str(x["notes"]) if x.get("notes") is not None else None),
    )
    # Validate early to fail fast
    spec.validate()
    return spec
