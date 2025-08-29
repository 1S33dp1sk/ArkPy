# arknet_py/utils/tar_canon.py — Canonical TAR (USTAR) builder for deterministic hashing
#
# - Walks a root directory (or a set of include roots)
# - Excludes common transient dirs/files
# - Sorts entries by POSIX arcname (forward slashes)
# - Writes USTAR format with:
#     uid=gid=0, uname=gname="", mode=0o644, mtime=0
# - Regular files only (symlinks, devices, dirs are skipped)
#
# NOTE: Keep this in sync with C-side canonicalization if you add fields.

from __future__ import annotations

import io
import os
import tarfile
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    # defaults (compat)
    "DEFAULT_INCLUDE",
    "DEFAULT_EXCLUDE",
    "DEFAULT_EXCLUDES",
    # core
    "iter_files",
    "canonical_arcname",
    "build_canonical_tar_bytes",
    "canonical_tar_bytes",   # compat alias
    "write_canonical_tar",
]

# Reasonable defaults (tuned to typical repos)
DEFAULT_EXCLUDES: Set[str] = {
    ".git", ".gitignore", ".gitattributes",
    ".DS_Store", "__pycache__", ".pytest_cache",
    ".ipynb_checkpoints",
    "venv", ".venv", "env", ".mypy_cache",
    "node_modules", "commit.json",
    # Arknet-specific: do not include metadata in identity
    "env.json",        # machine snapshot – varies across runs/machines
    "commit.json",     # written after hashing; not part of the commit preimage
}
# Compat names used elsewhere in the project
DEFAULT_INCLUDE: Sequence[str] = (".",)
DEFAULT_EXCLUDE: Set[str] = set(DEFAULT_EXCLUDES)

# Optional: allow callers to further filter paths (return True to KEEP file)
PathFilter = Callable[[str], bool]


def _is_excluded(name: str, excludes: Set[str]) -> bool:
    base = os.path.basename(name)
    return base in excludes


def canonical_arcname(root: str, full: str) -> str:
    """Relative POSIX path (forward slashes), no leading './'."""
    rel = os.path.relpath(full, start=root)
    return rel.replace(os.sep, "/")


def iter_files(
    roots: Sequence[str],
    *,
    excludes: Optional[Set[str]] = None,
    filter_fn: Optional[PathFilter] = None,
) -> Iterable[Tuple[str, str]]:
    """
    Yield (root, fullpath) tuples for regular files under each root.
    - Excludes dir/file basenames found in 'excludes'
    - Skips non-regular files (dirs, symlinks, devices)
    - Applies filter_fn(fullpath) if provided (True -> keep)
    """
    ex = set(DEFAULT_EXCLUDES)
    if excludes:
        ex |= set(excludes)

    for root in roots:
        r = os.path.realpath(root)
        for dirpath, dirnames, filenames in os.walk(r):
            # prune excluded dirs in-place
            dirnames[:] = [d for d in dirnames if not _is_excluded(d, ex)]
            for fn in filenames:
                if _is_excluded(fn, ex):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                # regular files only; dereference symlinks by skipping them
                if not os.path.isfile(full) or not stat_is_regular(st.st_mode):
                    continue
                if filter_fn and not filter_fn(full):
                    continue
                yield (r, os.path.realpath(full))


def stat_is_regular(mode: int) -> bool:
    # Avoid importing stat module for a single check
    S_IFMT = 0o170000
    S_IFREG = 0o100000
    return (mode & S_IFMT) == S_IFREG


def _tarinfo_for(full: str, arc: str) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(arc)
    st = os.stat(full)
    ti.size = int(st.st_size)
    ti.mode = 0o644
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    ti.mtime = 0
    ti.type = tarfile.REGTYPE
    return ti


def build_canonical_tar_bytes(
    root: str,
    *,
    includes: Optional[Sequence[str]] = None,
    excludes: Optional[Set[str]] = None,
    filter_fn: Optional[PathFilter] = None,
) -> bytes:
    """
    Return USTAR bytes of a canonical archive containing files under root (or includes).
    """
    roots = [os.path.join(root, p) if includes else root for p in (includes or (".",))]
    # Collect and sort by arcname
    files: List[Tuple[str, str, str]] = []  # (root, full, arc)
    for r, full in iter_files(roots, excludes=excludes, filter_fn=filter_fn):
        arc = canonical_arcname(r, full)
        files.append((r, full, arc))
    files.sort(key=lambda t: t[2])

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.USTAR_FORMAT) as tf:
        for _, full, arc in files:
            ti = _tarinfo_for(full, arc)
            with open(full, "rb") as f:
                tf.addfile(ti, f)
    return buf.getvalue()


# ---------------- compatibility alias ----------------

def canonical_tar_bytes(
    root: str,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    filter_fn: Optional[PathFilter] = None,
) -> bytes:
    """
    Compat wrapper for older call sites:
      canonical_tar_bytes(root, include=..., exclude=..., filter_fn=...)
    maps to build_canonical_tar_bytes(root, includes=..., excludes=...).
    """
    inc = include if include is not None else DEFAULT_INCLUDE
    exc = set(exclude) if exclude is not None else DEFAULT_EXCLUDE
    return build_canonical_tar_bytes(root, includes=inc, excludes=exc, filter_fn=filter_fn)


def write_canonical_tar(
    root: str,
    out_path: str,
    *,
    includes: Optional[Sequence[str]] = None,
    excludes: Optional[Set[str]] = None,
    filter_fn: Optional[PathFilter] = None,
) -> None:
    """Write canonical TAR bytes to a file."""
    data = build_canonical_tar_bytes(
        root, includes=includes, excludes=excludes, filter_fn=filter_fn
    )
    with open(out_path, "wb") as w:
        w.write(data)
