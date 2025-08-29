# training/audit.py — deterministic audit of sampled training leaves
from __future__ import annotations

import importlib.util
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..utils.json_canon import dumps_canon
from ..utils.hashing import domain_hash_hex
from ..determinism import apply_determinism_profile
from ..commit import compute_commit as compute_artifact_commit

_AUDIT_DOMAIN = b"ARK/AUDIT/REPORT/v1\n"


# --------------------------- datamodel --------------------------------------

@dataclass
class AuditSampleResult:
    index: int                # leaf index in transcript
    kind: str                 # e.g., "train_step"
    ok: bool
    checks: Dict[str, bool]   # per-check booleans (e.g., parent_commit_ok, dataset_commit_ok, pre/post)
    metrics: Dict[str, float] # numeric drift metrics (e.g., max_abs, max_rel)
    notes: List[str]


@dataclass
class AuditReport:
    transcript_path: str
    artifact_dir: str
    artifact_commit: str
    total_leaves: int
    sampled: int
    seed: int
    tolerance: float
    samples: List[AuditSampleResult]
    overall_ok: bool

    def to_json(self) -> str:
        return dumps_canon(asdict(self))

    def commit_hex(self) -> str:
        return domain_hash_hex(self.to_json().encode("utf-8"), _AUDIT_DOMAIN)


# --------------------------- helpers ----------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_trainer_recompute(artifact_dir: str):
    """
    If artifact contains trainer.py with a recompute_step() function exposing:
      recompute_step(artifact_dir:str, job_spec:dict, leaf:dict, tolerance:float) -> dict
    we use it for deep audits (forward/backward/opt).
    """
    entry = os.path.join(artifact_dir, "trainer.py")
    if not os.path.exists(entry):
        return None
    spec = importlib.util.spec_from_file_location("arknet_artifact_trainer", entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "recompute_step", None)


def _select_indices(n: int, k: int, seed: int) -> List[int]:
    k = max(0, min(k, n))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[:k])


def _within_tol(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _drift_metrics(a: "np.ndarray", b: "np.ndarray") -> Tuple[float, float]:
    if np is None:
        return (0.0, 0.0)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    diff = np.abs(aa - bb)
    max_abs = float(diff.max() if diff.size else 0.0)
    denom = np.maximum(np.abs(aa), np.abs(bb))
    denom[denom == 0.0] = 1.0
    rel = diff / denom
    max_rel = float(rel.max() if rel.size else 0.0)
    return (max_abs, max_rel)


# ----------------------------- public API -----------------------------------

def audit_transcript(
    artifact_dir: str,
    transcript_path: str,
    spec_path: Optional[str] = None,
    *,
    k: int = 8,
    seed: int = 0,
    tolerance: float = 1e-6,
    apply_determinism: bool = True,
) -> AuditReport:
    """
    Recompute K sampled leaves from `transcript_path` using artifact’s trainer if available.

    Transcript format (from new trainer):
      {
        "header": {
          "spec": {...},
          "parent_commit": "...",
          "env_commit": "...",
          "steps": N,
          "dataset_commit": "..."
        },
        "leaves": [ { "kind":"train_step", "index":0, "pre_commit":..., "post_commit":..., "loss":..., "dataset_commit":..., "batch": {...}, "tags":[...] }, ... ],
        "root": "...",
        "domain": "ARK/TRAIN/TRANSCRIPT/v1"
      }

    Deep audit path (if artifact trainer exposes recompute_step):
      - Calls trainer.recompute_step(artifact_dir, job_spec_dict, leaf_dict, tolerance)
      - Consumes booleans like pre_commit_ok/post_commit_ok and numeric metrics.

    Light audit fallback:
      - Checks parent_commit == computed artifact commit (per sample)
      - Checks leaf.dataset_commit == header.dataset_commit (if present)
      - If leaf has tag "noop", verifies pre_commit == post_commit
    """
    if apply_determinism:
        apply_determinism_profile(seed=seed)

    tr = _load_json(transcript_path)
    header: Dict[str, Any] = dict(tr.get("header") or {})
    leaves: List[Dict[str, Any]] = list(tr.get("leaves") or [])
    total = len(leaves)

    # Spec to provide to recompute (prefer explicit file; else header.spec)
    if spec_path:
        job_spec: Dict[str, Any] = _load_json(spec_path)
    else:
        job_spec = dict(header.get("spec") or {})

    # Commit of the artifact currently on disk (should match header.parent_commit)
    artifact_commit_hex, _ = compute_artifact_commit(artifact_dir)

    # Optional deep-audit hook from artifact
    recompute_step = _import_trainer_recompute(artifact_dir)

    # Which leaves to audit
    chosen = _select_indices(total, k, seed)

    results: List[AuditSampleResult] = []
    for idx in chosen:
        leaf = leaves[idx]
        kind = str(leaf.get("kind") or "unknown")
        checks: Dict[str, bool] = {}
        metrics: Dict[str, float] = {}
        notes: List[str] = []
        ok = True

        # Parent commit (header vs computed)
        parent_commit = header.get("parent_commit")
        if parent_commit is not None:
            same_parent = (str(parent_commit) == str(artifact_commit_hex))
            checks["parent_commit_ok"] = same_parent
            ok = ok and same_parent
            if not same_parent:
                notes.append("parent_commit != compute_commit(artifact_dir)")

        # Dataset commit (leaf vs header)
        if "dataset_commit" in leaf and "dataset_commit" in header:
            dc_same = (str(leaf["dataset_commit"]) == str(header["dataset_commit"]))
            checks["dataset_commit_ok"] = dc_same
            ok = ok and dc_same
            if not dc_same:
                notes.append("leaf.dataset_commit != header.dataset_commit")

        if recompute_step and kind in ("train_step", "opt_step", "sgd_step", "adamw_step"):
            try:
                rc = recompute_step(artifact_dir, job_spec, leaf, float(tolerance))
                # Booleans
                for key in ("pre_commit_ok", "post_commit_ok", "optimizer_ok", "batch_ok"):
                    if key in rc:
                        val = bool(rc[key])
                        checks[key] = val
                        ok = ok and val
                # Metrics
                for key in ("max_abs", "max_rel"):
                    if key in rc and rc[key] is not None:
                        metrics[key] = float(rc[key])
                # Notes
                if isinstance(rc.get("notes"), list):
                    notes.extend(str(n) for n in rc["notes"])
            except Exception as e:  # pragma: no cover
                ok = False
                checks["recompute_exception"] = False
                notes.append(f"recompute exception: {e!r}")
        else:
            # Light audit fallback
            pre_c = leaf.get("pre_commit")
            post_c = leaf.get("post_commit")
            tags = leaf.get("tags") or []
            if pre_c is not None and post_c is not None and "noop" in tags:
                same = (str(pre_c) == str(post_c))
                checks["noop_commit_equal"] = same
                ok = ok and same
                if not same:
                    notes.append("noop leaf but pre/post differ")
            else:
                notes.append("light audit only (no trainer.recompute_step)")

        results.append(AuditSampleResult(
            index=int(idx),
            kind=kind,
            ok=bool(ok),
            checks=checks,
            metrics=metrics,
            notes=notes,
        ))

    overall_ok = all(r.ok for r in results)
    rep = AuditReport(
        transcript_path=os.path.realpath(transcript_path),
        artifact_dir=os.path.realpath(artifact_dir),
        artifact_commit=artifact_commit_hex,
        total_leaves=total,
        sampled=len(results),
        seed=int(seed),
        tolerance=float(tolerance),
        samples=results,
        overall_ok=bool(overall_ok),
    )
    return rep
