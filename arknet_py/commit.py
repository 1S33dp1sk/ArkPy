"""
arknet_py.commit — canonical artifact commit helpers.

We compute a content-addressed commit for a *model artifact directory*:
- Build a deterministic (USTAR) tar from the directory using utils.tar_canon
  (normalized metadata, stable ordering, posix arcnames).
- Hash the tar bytes (optionally with domain separation).
- Return the digest hex and a manifest dict. Optionally write commit.json.

Contract:
- The artifact directory SHOULD contain a `manifest.json`. It is *included*
  in the tar and *not modified* during hashing.
- `commit.json` is written *after* hashing and is not part of the commit.

Public API:
- compute_commit(artifact_dir, *, include=None, exclude=None, domain=None)
- write_commit_files(artifact_dir, out_dir=None, **kwargs)
- load_commit_manifest(path)   # load commit.json or manifest.json
- verify_commit(artifact_dir, expected_hex, **kwargs) -> bool
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .utils.tar_canon import canonical_tar_bytes, DEFAULT_INCLUDE, DEFAULT_EXCLUDE
from .utils.hashing import sha256_hex, sha256_domain_hex
from .utils.json_canon import loads_canon
from .utils.iohelpers import atomic_write
from .constants import DOM_MODEL_COMMIT, domain_bytes


def _load_manifest(path: str) -> Dict[str, Any]:
    mp = os.path.join(path, "manifest.json")
    if not os.path.exists(mp):
        raise FileNotFoundError("manifest.json missing in artifact directory")
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_commit(
    artifact_dir: str,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    domain: Optional[bytes] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute deterministic commit hex for an artifact directory and return
    (digest_hex, manifest_dict_with_commit).

    - `include`: optional list of relative paths to include (default: DEFAULT_INCLUDE)
    - `exclude`: optional iterable of path patterns to exclude (default: DEFAULT_EXCLUDE)
    - `domain`: optional domain-separation bytes; if None, uses DOM_MODEL_COMMIT
    """
    artifact_dir = os.path.realpath(artifact_dir)
    manifest = _load_manifest(artifact_dir)

    tar_bytes = canonical_tar_bytes(
        artifact_dir,
        include=include or DEFAULT_INCLUDE,
        exclude=exclude or DEFAULT_EXCLUDE,
    )

    dom = domain if domain is not None else domain_bytes(DOM_MODEL_COMMIT)
    if dom:
        digest_hex = sha256_domain_hex(dom, tar_bytes)
    else:
        digest_hex = sha256_hex(tar_bytes)

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
) -> str:
    """
    Compute commit and write `commit.json` (pretty, sorted). Returns digest hex.
    If out_dir is None, writes next to the artifact (same directory).
    """
    digest_hex, manifest = compute_commit(
        artifact_dir,
        include=include,
        exclude=exclude,
        domain=domain,
    )
    target_dir = out_dir or artifact_dir
    os.makedirs(target_dir, exist_ok=True)
    atomic_write(
        os.path.join(target_dir, "commit.json"),
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )
    return digest_hex


def load_commit_manifest(path: str) -> Dict[str, Any]:
    """
    Load commit.json if present; otherwise fall back to manifest.json.
    (Useful for cross-checking.)
    """
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
) -> bool:
    """
    Recompute commit and compare against expected hex.
    Returns True if they match (case-insensitive), False otherwise.
    """
    actual, _ = compute_commit(
        artifact_dir,
        include=include,
        exclude=exclude,
        domain=domain,
    )
    return actual.lower() == (expected_hex or "").strip().lower()


__all__ = [
    "compute_commit",
    "write_commit_files",
    "load_commit_manifest",
    "verify_commit",
]
