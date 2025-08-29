import importlib.util
import json
import os
from typing import Dict, Tuple

from .commit import write_commit_files
from .determinism import apply_determinism_profile

def _import_trainer(artifact_dir: str):
    entry = os.path.join(artifact_dir, "trainer.py")
    if not os.path.exists(entry):
        raise FileNotFoundError("trainer.py not found in artifact directory")
    spec = importlib.util.spec_from_file_location("arknet_trainer_entry", entry)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import trainer entry")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "train"):
        raise RuntimeError("trainer.py must expose train(job_spec, out_dir)")
    return mod

def train_once(artifact_dir: str, job_spec: Dict, out_dir: str) -> str:
    """
    - Applies determinism profile from job_spec (seed, allow_tf32)
    - Calls artifact's trainer.train(job_spec, out_dir)
    - Writes commit.json and returns digest hex
    """
    seed = int(job_spec.get("seed", 0))
    allow_tf32 = bool(job_spec.get("allow_tf32", False))
    apply_determinism_profile(seed=seed, allow_tf32=allow_tf32)

    os.makedirs(out_dir, exist_ok=True)
    trainer = _import_trainer(artifact_dir)
    trainer.train(job_spec, out_dir)

    # Ensure a manifest.json exists in out_dir (trainer must produce/patch it)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("trainer did not create manifest.json in out_dir")

    # Canonicalize + compute commit
    digest_hex = write_commit_files(out_dir)
    return digest_hex
