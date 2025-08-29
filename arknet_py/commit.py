# arknet_py/commit.py — canonical artifact commit helpers (identity-first)
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .utils.tar_canon import build_canonical_tar_bytes, DEFAULT_EXCLUDES
from .utils.hashing import sha256_hex, domain_hash_hex
from .utils.iohelpers import atomic_write
from .constants import DOM_MODEL_COMMIT, domain_bytes

# Files that define model identity (hash preimage) — only ones we tar by default.
# We include only those that actually exist in the artifact directory.
_IDENTITY_CANDIDATES: Sequence[str] = (
    "weights.safetensors",
    "model.py",
    "tokenizer.json",
    "tokenizer.model",
    "config.json",
)

# Additional volatile outputs we never want to include (belt-and-suspenders).
_VOLATILE: Sequence[str] = (
    "commit.json",
    "env.json",
    "transcript.json",
)

def _load_manifest(path: str) -> Dict[str, Any]:
    mp = os.path.join(path, "manifest.json")
    if not os.path.exists(mp):
        raise FileNotFoundError("manifest.json missing in artifact directory")
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_includes(artifact_dir: str, include: Optional[Sequence[str]], profile: str) -> Optional[Sequence[str]]:
    """
    Decide which paths to include in the canonical tar:
    - if 'include' is provided, use it verbatim (relative paths to artifact_dir)
    - else if profile == 'identity' (default): include only identity candidates that exist
    - else if profile == 'full': return None to mean "everything under root"
    """
    if include:
        return list(include)
    if profile == "full":
        return None
    # identity
    present = [p for p in _IDENTITY_CANDIDATES if os.path.exists(os.path.join(artifact_dir, p))]
    # If nothing found (very bare artifact), fall back to just model.py if present
    if not present and os.path.exists(os.path.join(artifact_dir, "model.py")):
        present = ["model.py"]
    return present or None  # None => full, but our excludes still trim volatility

def compute_commit(
    artifact_dir: str,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    domain: Optional[bytes] = None,
    profile: str = "identity",  # 'identity' (default) | 'full'
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute deterministic commit hex for an artifact directory and return
    (digest_hex, manifest_dict_with_commit).

    Behavior:
      - Profile 'identity' (default): hash only stable identity files
        (weights/model/tokenizer/config) if present; ignore transcript/env/etc.
      - Profile 'full': hash the entire directory tree (minus excludes).

    Args:
      include: explicit list of relative paths to include; overrides 'profile'
      exclude: extra names to exclude (basenames). DEFAULT_EXCLUDES + _VOLATILE always applied.
      domain: optional domain-separation bytes; defaults to DOM_MODEL_COMMIT
      profile: 'identity' | 'full'
    """
    artifact_dir = os.path.realpath(artifact_dir)
    manifest = _load_manifest(artifact_dir)

    # includes / excludes
    includes = _resolve_includes(artifact_dir, include, profile)
    ex = set(DEFAULT_EXCLUDES)
    ex.update(_VOLATILE)
    if exclude:
        ex.update(exclude)

    tar_bytes = build_canonical_tar_bytes(
        artifact_dir,
        includes=includes,
        excludes=ex,
        filter_fn=None,
    )

    dom = domain if domain is not None else domain_bytes(DOM_MODEL_COMMIT)
    digest_hex = domain_hash_hex(tar_bytes, dom) if dom else sha256_hex(tar_bytes)

    out_manifest = dict(manifest)
    out_manifest["format_version"] = int(out_manifest.get("format_version", 1))
    out_manifest["commit"] = digest_hex
    return digest_hex, out_manifest

def write_commit_files(
    artifact_dir: str,
    out_dir: Optional[str] = None,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    domain: Optional[bytes] = None,
    profile: str = "identity",
) -> str:
    """Compute commit and write `commit.json` (pretty, sorted). Returns digest hex."""
    digest_hex, manifest = compute_commit(
        artifact_dir,
        include=include,
        exclude=exclude,
        domain=domain,
        profile=profile,
    )
    target_dir = out_dir or artifact_dir
    os.makedirs(target_dir, exist_ok=True)
    atomic_write(
        os.path.join(target_dir, "commit.json"),
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return digest_hex

def load_commit_manifest(path: str) -> Dict[str, Any]:
    """Load commit.json if present; otherwise fall back to manifest.json."""
    p1 = os.path.join(path, "commit.json")
    p2 = os.path.join(path, "manifest.json")
    target = p1 if os.path.exists(p1) else p2
    if not os.path.exists(target):
        raise FileNotFoundError("commit.json or manifest.json not found")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_commit(
    artifact_dir: str,
    expected_hex: str,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    domain: Optional[bytes] = None,
    profile: str = "identity",
) -> bool:
    """Recompute commit and compare against expected hex (case-insensitive)."""
    actual, _ = compute_commit(
        artifact_dir,
        include=include,
        exclude=exclude,
        domain=domain,
        profile=profile,
    )
    return actual.lower() == (expected_hex or "").strip().lower()

__all__ = [
    "compute_commit",
    "write_commit_files",
    "load_commit_manifest",
    "verify_commit",
]
