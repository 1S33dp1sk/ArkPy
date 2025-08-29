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
from ..utils.hashing import sha256_domain_hex
from ..determinism import apply_determinism_profile
from ..commit import compute_commit as compute_artifact_commit


_AUDIT_DOMAIN = b"ARK/AUDIT/REPORT/v1\n"


@dataclass
class AuditSampleResult:
    index: int                # leaf index in transcript
    kind: str                 # e.g., "train_step"
    ok: bool
    checks: Dict[str, bool]   # per-check booleans (e.g., pre_commit, dataset, post_commit, grads)
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
        return sha256_domain_hex(_AUDIT_DOMAIN, self.to_json().encode("utf-8"))


# ---- internal helpers -----------------------------------------------------

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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    max_abs = float(diff.max() if diff.size else 0.0)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom[denom == 0.0] = 1.0
    rel = diff / denom
    max_rel = float(rel.max() if rel.size else 0.0)
    return (max_abs, max_rel)


# ---- public API -----------------------------------------------------------

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
    - If trainer.recompute_step() is not present, performs a *light audit*:
        * checks artifact commit matches leaf,
        * validates dataset commit equality,
        * validates post-commit equals pre-commit when leaf says "no-op".
    Returns a deterministic AuditReport with its own commit hex.
    """
    if apply_determinism:
        apply_determinism_profile(seed=seed)

    tr = _load_json(transcript_path)
    leaves: List[Dict[str, Any]] = list(tr.get("leaves") or [])
    total = len(leaves)

    job_spec: Dict[str, Any] = _load_json(spec_path) if spec_path else (tr.get("job_spec") or {})
    artifact_commit_hex, _ = compute_artifact_commit(artifact_dir)

    # Trainer hook (deep audit) if available
    recompute_step = _import_trainer_recompute(artifact_dir)

    # sample indices
    chosen = _select_indices(total, k, seed)

    results: List[AuditSampleResult] = []
    for idx in chosen:
        leaf = leaves[idx]
        kind = str(leaf.get("kind") or "unknown")
        checks: Dict[str, bool] = {}
        metrics: Dict[str, float] = {}
        notes: List[str] = []
        ok = True

        # Always check artifact commit if present in leaf
        leaf_art_commit = leaf.get("artifact_commit")
        if leaf_art_commit is not None:
            same = (str(leaf_art_commit) == artifact_commit_hex)
            checks["artifact_commit"] = same
            ok = ok and same
            if not same:
                notes.append("artifact_commit mismatch")

        # dataset commit equality if both sides provide it
        if "dataset_commit" in leaf and "dataset_commit" in job_spec:
            dc_same = (str(leaf["dataset_commit"]) == str(job_spec["dataset_commit"]))
            checks["dataset_commit"] = dc_same
            ok = ok and dc_same
            if not dc_same:
                notes.append("dataset_commit mismatch")

        if recompute_step and kind in ("train_step", "opt_step", "sgd_step", "adamw_step"):
            try:
                rc = recompute_step(artifact_dir, job_spec, leaf, float(tolerance))
                # expected keys:
                #   rc["pre_commit_ok"], rc["post_commit_ok"]
                #   rc.get("max_abs"), rc.get("max_rel")
                #   rc.get("optimizer_ok"), rc.get("batch_ok")
                for key in ("pre_commit_ok", "post_commit_ok", "optimizer_ok", "batch_ok"):
                    if key in rc:
                        checks[key] = bool(rc[key])
                        ok = ok and bool(rc[key])
                for key in ("max_abs", "max_rel"):
                    if key in rc and rc[key] is not None:
                        metrics[key] = float(rc[key])
                if "notes" in rc and isinstance(rc["notes"], list):
                    notes.extend([str(n) for n in rc["notes"]])
            except Exception as e:  # pragma: no cover
                ok = False
                checks["recompute_exception"] = False
                notes.append(f"recompute exception: {e!r}")
        else:
            # Light audit fallback
            pre_c = leaf.get("pre_commit")
            post_c = leaf.get("post_commit")
            if pre_c is not None and post_c is not None and "noop" in (leaf.get("tags") or []):
                same = (str(pre_c) == str(post_c))
                checks["noop_commit_equal"] = same
                ok = ok and same
                if not same:
                    notes.append("noop leaf but pre/post differ")
            else:
                notes.append("light audit only (no trainer.recompute_step)")

        results.append(AuditSampleResult(
            index=idx, kind=kind, ok=ok, checks=checks, metrics=metrics, notes=notes
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
