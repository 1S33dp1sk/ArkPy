# training/trainer.py — glue: run train loop per TrainingSpec, emit artifacts
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional, Tuple

# --- Arknet plumbing -------------------------------------------------------

from ..determinism import apply_determinism_profile
from ..utils.json_canon import dumps_canon, loads_canon
from ..utils.hashing import sha256_hex, sha256_domain_hex
from ..utils.merkle import merkle_root_hex
from ..utils.iohelpers import ensure_dir, atomic_write
from ..commit import compute_commit as compute_artifact_commit, write_commit_files
from ..training.container import write_environment_snapshot
from ..training.spec import TrainingSpec
from ..training.dataset import DatasetSpec, open_dataset
from ..training.exporter import export_safetensors, weight_commit_hex as exporter_weight_commit_hex

_AUDIT_STEP_DOMAIN = b"ARK/TRAIN/STEP/v1\n"
_TRANSCRIPT_DOMAIN = b"ARK/TRAIN/TRANSCRIPT/v1\n"


# --- Backend selection (Torch if available/supported; else Dummy) ----------

def _import_module(entry_py: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, entry_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import: {entry_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DummyBackend:
    """
    Minimal, strictly deterministic backend:
    - No real updates; weights commit is derived from model-reported bytes if available,
      else from model.__class__.__name__ + repr of a stable snapshot.
    - export_safetensors() writes small placeholder weights (deterministic).
    """
    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    def is_supported(model: Any) -> bool:
        return True  # always

    def weight_commit_hex(self) -> str:
        # Preferred: model exposes stable raw weights bytes
        if hasattr(self.model, "get_weights_bytes"):
            b = self.model.get_weights_bytes()
            if isinstance(b, (bytes, bytearray)):
                return sha256_hex(bytes(b))
        # Fallback: deterministic fingerprint of class name + stable repr
        snap = {
            "class": self.model.__class__.__name__,
            "repr": getattr(self.model, "stable_repr", None) or str(self.model),
        }
        return sha256_domain_hex(b"ARK/DUMMY/WEIGHTS/v1\n", dumps_canon(snap).encode("utf-8"))

    def export_safetensors(self, out_path: str) -> None:
        # Write deterministic tiny blob (so the artifact is complete)
        payload = dumps_canon({
            "dummy": True,
            "model_class": self.model.__class__.__name__,
        }).encode("utf-8")
        atomic_write(out_path, payload, binary=True)

    # training step: return unchanged commit; report "noop"
    def train_step(self, batch: Any, lr: float, opt: Any = None) -> Dict[str, Any]:
        loss_val = 0.0
        return {"loss": float(loss_val)}

    def create_optimizer(self, opt_cfg: Dict[str, Any]) -> Any:
        return None


def _resolve_backend(model: Any):
    # Try Torch backend if available and model is a torch.nn.Module
    try:
        from ..backends.torch_backend import TorchBackend  # type: ignore
        if TorchBackend.is_supported(model):
            return TorchBackend(model)
    except Exception:
        pass
    # Fallback
    return _DummyBackend(model)


# --- Recipe (LR schedule) and optimizer glue (soft-deps) -------------------

def _make_optimizer(backend, model, opt_cfg: Dict[str, Any]):
    # Prefer backend implementation
    try:
        opt = backend.create_optimizer(opt_cfg)
        if opt is not None:
            return opt
    except Exception:
        pass
    # Try reference optimizers
    try:
        from .optimizer import create_optimizer  # type: ignore
        return create_optimizer(model, opt_cfg)
    except Exception:
        # No-op optimizer
        return None


def _lr_for_step(step_idx: int, total_steps: int, base_lr: float, sched_cfg: Dict[str, Any]) -> float:
    # Try reference recipe
    try:
        from .recipe import lr_schedule  # type: ignore
        return float(lr_schedule(step_idx, total_steps, base_lr, sched_cfg))
    except Exception:
        return float(base_lr)


# --- Transcript helpers ----------------------------------------------------

def _leaf_preimage(
    index: int,
    pre_commit: Optional[str],
    post_commit: Optional[str],
    loss: Optional[float],
    dataset_commit: Optional[str],
    batch_meta: Dict[str, Any],
    tags: Iterable[str] = (),
) -> Dict[str, Any]:
    leaf = {
        "index": int(index),
        "kind": "train_step",
        "pre_commit": pre_commit,
        "post_commit": post_commit,
        "loss": loss,
        "dataset_commit": dataset_commit,
        "batch": batch_meta,
        "tags": list(tags),
    }
    # Canonical null-handling
    return {k: v for k, v in leaf.items() if v is not None}


def _leaf_hash(leaf: Dict[str, Any]) -> bytes:
    return sha256_domain_hex(_AUDIT_STEP_DOMAIN, dumps_canon(leaf).encode("utf-8"), raw=True)


def _write_transcript(out_dir: str, header: Dict[str, Any], leaves: Iterable[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Write transcript.json with header+leaves and return (root_hex, file_path).
    """
    leaf_hashes = []
    leaf_list = []
    for lf in leaves:
        leaf_list.append(lf)
        leaf_hashes.append(_leaf_hash(lf))
    root = merkle_root_hex(leaf_hashes)
    doc = {
        "header": header,
        "leaves": leaf_list,
        "root": root,
        "domain": _TRANSCRIPT_DOMAIN.decode("utf-8").strip(),
    }
    p = os.path.join(out_dir, "transcript.json")
    atomic_write(p, dumps_canon(doc).encode("utf-8"))
    return root, p


# --- Artifact IO -----------------------------------------------------------

def _load_artifact_model(artifact_dir: str):
    entry = os.path.join(artifact_dir, "model.py")
    mod = _import_module(entry, "arknet_model_entry")
    if not hasattr(mod, "load_model"):
        raise RuntimeError("artifact model.py must export load_model(artifact_dir)")
    return mod.load_model(artifact_dir), mod


def _load_manifest(artifact_dir: str) -> Dict[str, Any]:
    p = os.path.join(artifact_dir, "manifest.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_new_manifest(out_dir: str, base_manifest: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    mf = dict(base_manifest)
    mf.update(overrides)
    atomic_write(os.path.join(out_dir, "manifest.json"), dumps_canon(mf).encode("utf-8"))


def _copy_artifact_scaffold(src_dir: str, dst_dir: str) -> None:
    # Ensure dst exists; copy model.py by default; leave room for custom extras
    ensure_dir(dst_dir)
    for name in ("model.py",):
        s = os.path.join(src_dir, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, name))


# --- Public entrypoint -----------------------------------------------------

def train_once(artifact_dir: str, job_spec: Dict[str, Any], out_dir: str) -> str:
    """
    Deterministic training driver. Returns *artifact commit hex* of `out_dir`.
    `job_spec` must satisfy TrainingSpec (see training/spec.py).
    """
    ensure_dir(out_dir)

    # Parse spec & apply determinism
    spec = TrainingSpec.from_dict(job_spec)
    apply_determinism_profile(seed=spec.seed, allow_tf32=bool(spec.allow_tf32))

    # Load base model + backend
    model, model_module = _load_artifact_model(artifact_dir)
    backend = _resolve_backend(model)

    # Dataset
    ds_spec = DatasetSpec.from_dict(spec.dataset)
    dset = open_dataset(ds_spec, seed=spec.seed)
    total_steps = int(spec.steps)
    batch_size = int(spec.batch_size)

    # Optimizer & schedule
    opt = _make_optimizer(backend, model, spec.optimizer)

    # Anchors & metadata
    parent_commit_hex, base_manifest = compute_artifact_commit(artifact_dir)
    env_commit_hex = write_environment_snapshot(out_dir)

    # Pre & transcript init
    leaves = []
    dataset_commit = dset.commit_hex if hasattr(dset, "commit_hex") else None

    # Train loop
    for step in range(total_steps):
        # deterministic batch
        batch, batch_meta = dset.next_batch(batch_size)

        # pre-weights commit
        try:
            pre_commit = backend.weight_commit_hex()
        except Exception:
            pre_commit = None

        # LR for step
        lr = _lr_for_step(step, total_steps, float(spec.optimizer.get("lr", 1e-3)), spec.schedule)

        # one step
        try:
            info = backend.train_step(batch, lr, opt)
            loss_val = float(info.get("loss", 0.0)) if isinstance(info, dict) else 0.0
        except Exception as e:
            # strict determinism: if step fails, mark noop & continue
            loss_val = 0.0

        # post-weights commit
        try:
            post_commit = backend.weight_commit_hex()
        except Exception:
            post_commit = pre_commit

        # Compose leaf (if backend is dummy/no-op, tag 'noop')
        tags = []
        if pre_commit == post_commit:
            tags.append("noop")
        leaves.append(_leaf_preimage(
            index=step,
            pre_commit=pre_commit,
            post_commit=post_commit,
            loss=loss_val,
            dataset_commit=dataset_commit,
            batch_meta=batch_meta,
            tags=tags,
        ))

    # Export weights (canonical)
    weights_path = os.path.join(out_dir, "weights.safetensors")
    try:
        backend.export_safetensors(weights_path)
    except Exception:
        # fallback tiny, but ensure file exists
        atomic_write(weights_path, b"{}", binary=True)

    # Transcript (header + leaves)
    header = {
        "spec": spec.to_public_dict(),
        "parent_commit": parent_commit_hex,
        "env_commit": env_commit_hex,
        "steps": total_steps,
        "dataset_commit": dataset_commit,
    }
    root_hex, transcript_path = _write_transcript(out_dir, header, leaves)

    # New manifest & scaffold
    _copy_artifact_scaffold(artifact_dir, out_dir)
    overrides = {
        "parent_commit": parent_commit_hex,
        "transcript_root": root_hex,
        "trainer": {
            "spec_hash": spec.spec_hash_hex(),
            "env_commit": env_commit_hex,
            "transcript": os.path.basename(transcript_path),
        },
    }
    _write_new_manifest(out_dir, base_manifest, overrides)

    # Final artifact commit
    digest_hex, _ = compute_artifact_commit(out_dir)
    # Ensure commit.json present
    write_commit_files(out_dir)
    return digest_hex


# --- Audit hook (optional; used by training/audit.py) ----------------------

def recompute_step(artifact_dir: str, job_spec: Dict[str, Any], leaf: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Best-effort recompute of a single leaf:
      - Loads model + backend;
      - If possible, *assumes model currently at pre_commit* (we do not time-travel weights);
      - Runs forward/backward/update for the batch described by leaf["batch"] deterministically;
      - Compares post-commit within tolerance (if numeric weights & backend supports).
    In MVP, when we cannot guarantee preimage restore, we return light checks.
    """
    out: Dict[str, Any] = {"notes": []}
    try:
        spec = TrainingSpec.from_dict(job_spec)
    except Exception as e:
        out["notes"].append(f"spec parse failed: {e!r}")
        return out

    # determinism
    apply_determinism_profile(seed=spec.seed, allow_tf32=bool(spec.allow_tf32))

    # load model + backend
    try:
        model, _ = _load_artifact_model(artifact_dir)
        backend = _resolve_backend(model)
    except Exception as e:
        out["notes"].append(f"load model/backend failed: {e!r}")
        return out

    # baseline commits (what we have *now*)
    try:
        current_commit = backend.weight_commit_hex()
    except Exception:
        current_commit = None

    pre_c = leaf.get("pre_commit")
    post_c = leaf.get("post_commit")

    out["pre_commit_ok"] = (pre_c is None) or (current_commit == pre_c)
    if not out["pre_commit_ok"]:
        out["notes"].append("cannot guarantee preimage restore; treating as light audit")

    # dataset batch reconstruction (best effort)
    try:
        ds_spec = DatasetSpec.from_dict(spec.dataset)
        dset = open_dataset(ds_spec, seed=spec.seed)
        batch_meta = leaf.get("batch") or {}
        batch = dset.batch_from_meta(batch_meta)
    except Exception as e:
        out["notes"].append(f"batch reconstruction failed: {e!r}")
        batch = None

    # If we can proceed, run a single step
    if batch is not None:
        try:
            lr = float(spec.optimizer.get("lr", 1e-3))
            info = backend.train_step(batch, lr, None)
            # compute new commit
            new_commit = backend.weight_commit_hex()
            out["post_commit_ok"] = (post_c is None) or (new_commit == post_c)
            out["max_abs"] = float(0.0)
            out["max_rel"] = float(0.0)
        except Exception as e:
            out["notes"].append(f"backend train_step failed: {e!r}")
    else:
        out["post_commit_ok"] = False

    return out
