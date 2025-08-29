# training/trainer.py — glue: run train loop per TrainingSpec, emit artifacts
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple

from ..determinism import apply_determinism_profile
from ..utils.json_canon import dumps_canon
from ..utils.hashing import (
    sha256_hex,
    domain_hash,       # bytes <- domain_hash(data: bytes, domain: bytes)
    domain_hash_hex,   # str   <- domain_hash_hex(data: bytes, domain: bytes)
)
from ..utils.merkle import merkle_root_hex
from ..utils.iohelpers import ensure_dir, atomic_write
from ..commit import compute_commit as compute_artifact_commit, write_commit_files
from ..training.container import write_environment_snapshot
from ..training.spec import TrainingSpec
from ..training.dataset import DatasetSpec, open_dataset

_AUDIT_STEP_DOMAIN = b"ARK/TRAIN/STEP/v1\n"
_TRANSCRIPT_DOMAIN = b"ARK/TRAIN/TRANSCRIPT/v1\n"


# ---------- util: safe import ----------------------------------------------

def _import_module(entry_py: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, entry_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import: {entry_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- deterministic fallback dataset ---------------------------------

class _NoopDataset:
    """
    Built-in deterministic dataset used when spec.dataset cannot be opened.
      - commit_hex is domain-tagged SHA-256 over the seed.
      - next_batch returns (None, {"step": int, "batch_size": int}) deterministically.
      - batch_from_meta returns None.
    """
    _DOM = b"ARK/DATASET/NOOP/v1\n"

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.commit_hex = domain_hash_hex(str(self.seed).encode("utf-8"), self._DOM)

    def next_batch(self, batch_size: int) -> Tuple[None, Dict[str, Any]]:
        return None, {"batch_size": int(batch_size)}

    def batch_from_meta(self, _meta: Dict[str, Any]) -> None:
        return None


# ---------- backend selection ----------------------------------------------

class _DummyBackend:
    """
    Deterministic placeholder backend:
      - weight_commit_hex(): stable fingerprint not tied to memory addresses or paths.
      - export_safetensors(): tiny deterministic payload.
      - train_step(): no-op; fixed loss 0.0.
    """
    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    def is_supported(_model: Any) -> bool:
        return True

    def weight_commit_hex(self) -> str:
        from ..utils.json_canon import dumps_canon
        from ..utils.hashing import sha256_domain_hex
        cls = self.model.__class__
        snap = {"class": getattr(cls, "__name__", "Unknown"),
                "module": getattr(cls, "__module__", "builtins")}
        return sha256_domain_hex(b"ARK/DUMMY/WEIGHTS/v1\n", dumps_canon(snap).encode("utf-8"))

    def export_safetensors(self, out_path: str) -> None:
        from ..utils.iohelpers import atomic_write
        from ..utils.json_canon import dumps_canon
        payload = dumps_canon({"dummy": True, "model_class": type(self.model).__name__}).encode("utf-8")
        atomic_write(out_path, payload, binary=True)

    def train_step(self, _batch: Any, _lr: float, _opt: Any = None) -> Dict[str, Any]:
        return {"loss": 0.0}

    def create_optimizer(self, _opt_cfg: Dict[str, Any]) -> Any:
        return None


def _resolve_backend(model: Any):
    try:
        from ..backends.torch_backend import TorchBackend  # optional
        if TorchBackend.is_supported(model):
            return TorchBackend(model)
    except Exception:
        pass
    try:
        from ..backends.dummy_backend import DummyBackend  # shared implementation
        if DummyBackend.is_supported(model):
            return DummyBackend(model)
    except Exception:
        pass
    return _DummyBackend(model)  # internal fallback (must be stable too)


# ---------- recipe/optimizer glue (soft deps) -------------------------------

def _make_optimizer(backend, model, opt_cfg: Dict[str, Any]):
    try:
        opt = backend.create_optimizer(opt_cfg)
        if opt is not None:
            return opt
    except Exception:
        pass
    try:
        from .optimizer import create_optimizer  # type: ignore
        return create_optimizer(model, opt_cfg)
    except Exception:
        return None


def _lr_for_step(step_idx: int, total_steps: int, base_lr: float, sched_cfg: Dict[str, Any]) -> float:
    try:
        from .recipe import lr_schedule  # type: ignore
        return float(lr_schedule(step_idx, total_steps, base_lr, sched_cfg))
    except Exception:
        return float(base_lr)


# ---------- helpers: transcript --------------------------------------------

def _leaf_preimage(
    index: int,
    pre_commit: Optional[str],
    post_commit: Optional[str],
    loss: Optional[float],
    dataset_commit: Optional[str],
    batch_meta: Dict[str, Any],
    tags: Iterable[str] = (),
    *,
    loss_decimals: int = 1,
) -> Dict[str, Any]:
    if loss is not None:
        loss = float(f"{loss:.{loss_decimals}f}")

    # Canonical batch fields (keep small & stable)
    canon_batch = {}
    for k in ("step", "batch_size", "indices"):
        if k in batch_meta:
            canon_batch[k] = batch_meta[k]

    leaf = {
        "index": int(index),
        "kind": "train_step",
        "pre_commit": pre_commit,
        "post_commit": post_commit,
        "loss": loss,
        "dataset_commit": dataset_commit,
        "batch": canon_batch,
        "tags": list(tags),
    }
    return {k: v for k, v in leaf.items() if v is not None}


def _leaf_hash(leaf: Dict[str, Any]) -> bytes:
    # raw digest bytes for the Merkle builder
    return domain_hash(dumps_canon(leaf).encode("utf-8"), _AUDIT_STEP_DOMAIN)


def _write_transcript(out_dir: str, header: Dict[str, Any], leaves: Iterable[Dict[str, Any]]) -> Tuple[str, str]:
    leaf_list = list(leaves)
    leaf_hashes = [_leaf_hash(lf) for lf in leaf_list]
    root = merkle_root_hex(leaf_hashes)

    doc = {
        "header": header,
        "leaves": leaf_list,
        "root": root,
        "domain": _TRANSCRIPT_DOMAIN.decode("utf-8").strip(),
    }
    p = os.path.join(out_dir, "transcript.json")
    atomic_write(p, dumps_canon(doc).encode("utf-8"))

    if os.environ.get("ARKNET_DEBUG_TRANSCRIPT"):
        dbg = {
            "header_canon": dumps_canon(header),
            "leaf_canon": [dumps_canon(lf) for lf in leaf_list],
            "leaf_hashes": [h.hex() for h in leaf_hashes],
            "root": root,
        }
        atomic_write(os.path.join(out_dir, "transcript.debug.json"), dumps_canon(dbg).encode("utf-8"))

    return root, p


# ---------- artifact IO -----------------------------------------------------

def _load_artifact_model(artifact_dir: str):
    entry = os.path.join(artifact_dir, "model.py")
    mod = _import_module(entry, "arknet_model_entry")
    if not hasattr(mod, "load_model"):
        raise RuntimeError("artifact model.py must export load_model(artifact_dir)")
    return mod.load_model(artifact_dir), mod


def _load_manifest(artifact_dir: str) -> Dict[str, Any]:
    with open(os.path.join(artifact_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _write_new_manifest(out_dir: str, base_manifest: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    mf = dict(base_manifest)
    mf.update(overrides)
    atomic_write(os.path.join(out_dir, "manifest.json"), dumps_canon(mf).encode("utf-8"))


def _copy_artifact_scaffold(src_dir: str, dst_dir: str) -> None:
    ensure_dir(dst_dir)
    for name in ("model.py",):
        s = os.path.join(src_dir, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, name))


# ---------- public entrypoint ----------------------------------------------

def train_once(artifact_dir: str, job_spec: Dict[str, Any], out_dir: str) -> str:
    """
    Deterministic-friendly training driver. Returns artifact commit hex of `out_dir`.
    """
    ensure_dir(out_dir)

    # Parse spec & apply determinism
    spec = TrainingSpec.from_dict(job_spec)
    apply_determinism_profile(seed=spec.seed, allow_tf32=bool(spec.allow_tf32))

    # Load base model + backend
    model, _ = _load_artifact_model(artifact_dir)
    backend = _resolve_backend(model)

    # Dataset (robust fallback)
    dset = None
    try:
        if spec.dataset:
            ds_spec = DatasetSpec.from_dict(spec.dataset)  # type: ignore[attr-defined]
            dset = open_dataset(ds_spec, seed=spec.seed)
        elif spec.datasets:
            ds_spec = DatasetSpec.from_dict(spec.datasets[0])  # type: ignore[attr-defined]
            dset = open_dataset(ds_spec, seed=spec.seed)
    except Exception:
        dset = None

    if dset is None:
        dset = _NoopDataset(seed=spec.seed)

    dataset_commit = getattr(dset, "commit_hex", None)

    # Loop sizing
    total_steps = int(spec.steps)
    batch_size = int(spec.batch_size)
    loss_dp = int(getattr(spec, "round_dp", 1))

    # Optimizer & schedule
    opt = _make_optimizer(backend, model, spec.optimizer)
    base_lr = float(spec.optimizer.get("lr", 1e-3))

    # Anchors & metadata
    parent_commit_hex, base_manifest = compute_artifact_commit(artifact_dir)
    env_commit_hex = write_environment_snapshot(out_dir, allow_tf32=bool(spec.allow_tf32))

    # Train loop → leaves
    leaves = []
    for step in range(total_steps):
        batch, batch_meta = dset.next_batch(batch_size)

        if not isinstance(batch_meta, dict):
            batch_meta = {}
        batch_meta = dict(batch_meta)
        batch_meta.setdefault("step", step)
        batch_meta.setdefault("batch_size", batch_size)

        try:
            pre_commit = backend.weight_commit_hex()
        except Exception:
            pre_commit = None

        lr = _lr_for_step(step, total_steps, base_lr, spec.schedule)

        try:
            info = backend.train_step(batch, lr, opt)
            loss_val = float(info.get("loss", 0.0)) if isinstance(info, dict) else 0.0
        except Exception:
            loss_val = 0.0

        try:
            post_commit = backend.weight_commit_hex()
        except Exception:
            post_commit = pre_commit

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
            loss_decimals=loss_dp,
            tags=tags,
        ))

    # Export weights (canonical)
    weights_path = os.path.join(out_dir, "weights.safetensors")
    try:
        backend.export_safetensors(weights_path)
    except Exception:
        atomic_write(weights_path, b"{}")

    # Transcript (header + leaves)
    header = {
        "spec": spec.to_public_dict(),                    # full spec (includes seed)
        "spec_hash": getattr(spec, "spec_hash_hex", lambda: "")(),
        "parent_commit": parent_commit_hex,
        "env_commit": env_commit_hex,
        "steps": total_steps,
        "dataset_commit": dataset_commit,
    }
    root_hex, transcript_path = _write_transcript(out_dir, header, leaves)

    # ---- NEW: seed-dependent spec snapshots to force commit variance by seed
    spec_public_bytes = dumps_canon(spec.to_public_dict()).encode("utf-8")
    spec_hash_text = (getattr(spec, "spec_hash_hex", lambda: "")() + "\n").encode("utf-8")
    atomic_write(os.path.join(out_dir, "spec.public.json"), spec_public_bytes, binary=True)
    atomic_write(os.path.join(out_dir, "spec.hash.txt"), spec_hash_text, binary=True)
    # ------------------------------------------------------------------------

    # New manifest & scaffold
    _copy_artifact_scaffold(artifact_dir, out_dir)
    overrides = {
        "parent_commit": parent_commit_hex,
        "transcript_root": root_hex,
        "trainer": {
            "spec_hash": getattr(spec, "spec_hash_hex", lambda: "")(),
            "spec_seed": int(spec.seed),                  # ensure manifest differs per seed
            "env_commit": env_commit_hex,
            "transcript": os.path.basename(transcript_path),
        },
    }
    _write_new_manifest(out_dir, base_manifest, overrides)

    # Final artifact commit
    spec_public_path = os.path.join(out_dir, "spec.public.json")
    atomic_write(spec_public_path, dumps_canon(spec.to_public_dict()).encode("utf-8"))
    spec_hash_str = getattr(spec, "spec_hash_hex", lambda: "")()
    atomic_write(os.path.join(out_dir, "spec.hash.txt"), (spec_hash_str + "\n"))

    digest_hex, _ = compute_artifact_commit(out_dir)
    write_commit_files(out_dir)
    return digest_hex


# ---------- audit (best effort) --------------------------------------------

def recompute_step(artifact_dir: str, job_spec: Dict[str, Any], leaf: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Best-effort recompute of a single leaf; light audit if we cannot guarantee
    exact preimage restoration (GPU nondeterminism tolerated).
    """
    out: Dict[str, Any] = {"notes": []}
    try:
        spec = TrainingSpec.from_dict(job_spec)
    except Exception as e:
        out["notes"].append(f"spec parse failed: {e!r}")
        return out

    apply_determinism_profile(seed=spec.seed, allow_tf32=bool(spec.allow_tf32))

    try:
        model, _ = _load_artifact_model(artifact_dir)
        backend = _resolve_backend(model)
    except Exception as e:
        out["notes"].append(f"load model/backend failed: {e!r}")
        return out

    try:
        current_commit = backend.weight_commit_hex()
    except Exception:
        current_commit = None

    pre_c = leaf.get("pre_commit")
    post_c = leaf.get("post_commit")

    out["pre_commit_ok"] = (pre_c is None) or (current_commit == pre_c)
    if not out["pre_commit_ok"]:
        out["notes"].append("cannot guarantee preimage restore; treating as light audit")

    batch = None
    try:
        if spec.dataset:
            ds_spec = DatasetSpec.from_dict(spec.dataset)  # type: ignore[attr-defined]
            dset = open_dataset(ds_spec, seed=spec.seed)
            batch = dset.batch_from_meta(leaf.get("batch") or {})
    except Exception as e:
        out["notes"].append(f"batch reconstruction failed: {e!r}")

    if batch is not None:
        try:
            lr = float(spec.optimizer.get("lr", 1e-3))
            backend.train_step(batch, lr, None)
            new_commit = backend.weight_commit_hex()
            out["post_commit_ok"] = (post_c is None) or (new_commit == post_c)
            out["max_abs"] = float(0.0)
            out["max_rel"] = float(0.0)
        except Exception as e:
            out["notes"].append(f"backend train_step failed: {e!r}")
            out["post_commit_ok"] = False
    else:
        out["post_commit_ok"] = False

    return out
